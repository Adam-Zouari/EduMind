"""Model-free regression coverage for video audio preparation and reuse gates."""

from __future__ import annotations

import copy
import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from experiments.benchmarks.common.process import WorkerExecutionError
from experiments.benchmarks.extraction.audio.adapters import Transcript, profiles
from experiments.benchmarks.extraction.audio.evaluate import score_speech
from experiments.benchmarks.extraction.audio.protocol import load_protocol as load_audio
from experiments.benchmarks.extraction.scoring import bootstrap_sources
from experiments.benchmarks.extraction.video import frozen_asr, validation
from experiments.benchmarks.extraction.video import frozen_asr_worker as worker
from experiments.benchmarks.extraction.video.protocol import (
    load_protocol,
    protocol_from_mapping,
)

VIDEO, AUDIO = load_protocol(), load_audio()
CANDIDATE = "whisper-small-en-control"


def _wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\0\0" * round(seconds * rate))


def _item(identifier="one", duration=65.0):
    starts = worker._window_starts(duration, 30.0, 2.0)
    return {
        "id": identifier,
        "kind": "video",
        "source_path": identifier + ".mp4",
        "duration_seconds": duration,
        "reference_transcript": " ".join("alpha" for _ in starts),
        "reference_segments": [
            {"text": "alpha", "start": start, "end": start + 1} for start in starts
        ],
        "independent_source_id": "group-" + identifier,
    }


def _identity():
    return {
        "audio_candidate": CANDIDATE,
        "model_path": "fixture-model",
        "model_revision": "fixture-revision",
        "selection_revision": "fixture-selection",
        "model_cache_manifest_sha256": "fixture-cache",
        "submodels": [],
        "model_lock_fingerprint": stable_hash({"fixture": "lock"}),
        "model_decision_fingerprint": stable_hash({"fixture": "decision"}),
    }


@pytest.fixture
def environment(monkeypatch):
    state = SimpleNamespace(
        fail=None,
        requests=[],
        decodes=[],
        closed=False,
        monitoring=False,
        transcript=None,
        placement_calls=0,
    )

    class Monitor:
        def __init__(self, *, require_vram, device, **kwargs):
            self.device = device
            self.gpu_identity = {"uuid": "fixture-gpu"} if device == "cuda" else {}
            self.vram_measurement_method = (
                "nvml-device-total" if device == "cuda" else "not-applicable"
            )

        def __enter__(self):
            state.monitoring = True
            return self

        def __exit__(self, *_):
            state.monitoring = False

        def metrics(self):
            if state.fail == "telemetry":
                error = RuntimeError("injected telemetry failure")
                error.resource_evidence = {
                    "peak_process_tree_ram_mb": 7.0,
                    "peak_vram_mb": 9000.0,
                }
                raise error
            return {
                "peak_process_tree_ram_mb": 7.0,
                "peak_vram_mb": 9000.0 if self.device == "cuda" else None,
            }

        def samples(self):
            return [
                {
                    "process_tree_ram_mb": 7.0,
                    "vram_mb": 9000.0 if self.device == "cuda" else None,
                }
            ]

    class Runtime:
        def load(self):
            assert state.monitoring
            if state.fail in {"load", "load_and_close"}:
                raise RuntimeError("injected load failure")

        def transcribe(self, path):
            assert state.monitoring
            state.requests.append(path.stem)
            if state.fail == path.stem:
                raise RuntimeError("injected request failure")
            if state.fail == "invalid_warmup" and path.stem == "warmup":
                return Transcript("alpha", ())
            return state.transcript or Transcript(
                "alpha",
                ({"text": "alpha", "start": 0.0, "end": 1.0},),
                ("native warning",),
            )

        def close(self):
            state.closed = True
            assert not state.monitoring
            if state.fail in {"close", "load_and_close"}:
                raise RuntimeError("injected cleanup failure")

        def placement_models(self):
            return (object(),)

        def parameters(self):
            if state.fail == "parameters":
                raise RuntimeError("injected parameters failure")
            profile = profiles(AUDIO)[CANDIDATE]
            return {
                **{
                    name: value
                    for name, value in profile.decoding.items()
                    if name not in {"model_id", "cpu_dtype", "cuda_dtype"}
                },
                "num_beams": 1,
                **{
                    name: value
                    for name, value in _identity().items()
                    if name
                    in {
                        "model_path",
                        "model_revision",
                        "selection_revision",
                        "model_cache_manifest_sha256",
                    }
                },
                "candidate": CANDIDATE,
                "model": profile.model,
                "backend": profile.backend,
                "device": state.device,
                "dtype": AUDIO.dtype(CANDIDATE, state.device),
                "decoder": profile.decoder,
                "timestamp_method": profile.timestamp_method,
                "language": profile.decoding["language"],
                "batch_size": AUDIO.batch_size,
                "sample_rate_hz": 16000,
                "runtime_version": frozen_asr.execution_identity()["package_versions"][
                    "transformers"
                ],
                "package_versions": {
                    name: value
                    for name, value in frozen_asr.execution_identity()[
                        "package_versions"
                    ].items()
                    if name != "numpy"
                },
            }

    def decode(item, destination, _protocol):
        assert state.monitoring
        state.decodes.append((item["id"], destination.name))
        if state.fail == "decode" and destination.name == "full.wav":
            raise subprocess.CalledProcessError(
                1, ["ffmpeg", "-i", item["source_path"]]
            )
        _wav(destination, item["duration_seconds"])
        return ["ffmpeg", "-i", item["source_path"], str(destination)]

    def placement(*_, **__):
        state.placement_calls += 1
        return {
            "status": "cpu_offload_detected"
            if state.fail == f"placement_{state.placement_calls}"
            else "qualified"
        }

    monkeypatch.setattr(worker, "ResourceMonitor", Monitor)
    monkeypatch.setattr(worker, "build_runtime", lambda *_: Runtime())
    monkeypatch.setattr(worker, "_decode", decode)
    monkeypatch.setattr(worker, "inspect_model_placement", placement)
    monkeypatch.setattr(worker, "seed_deterministically", lambda *_: None)
    monkeypatch.setattr(worker, "synchronize", lambda *_: None)

    def run(*, items=None, device="cpu", protocol=VIDEO, mode=None):
        state.device = device
        return worker.execute(
            {
                "candidate": CANDIDATE,
                "model_lock": {},
                "device": device,
                "profile": "smoke",
                "mode": mode,
                "items": items or [_item()],
                "protocol": protocol.meta.worker_payload(),
                "audio_protocol": AUDIO.meta.worker_payload(),
            }
        )

    state.run = run
    return state


@pytest.mark.parametrize(
    "duration,expected",
    [
        (0.01, [0.0]),
        (30.0, [0.0]),
        (30.01, [0.0, 28.0]),
        (58.0, [0.0, 28.0]),
        (65.0, [0.0, 28.0, 56.0]),
    ],
)
def test_deterministic_required_windows(duration, expected):
    assert worker._window_starts(duration, 30.0, 2.0) == expected


def test_single_warmup_and_unique_measured_decodes(environment):
    result = environment.run(items=[_item("one"), _item("two")])
    assert environment.requests == ["warmup", "0", "1", "2", "0", "1", "2"]
    assert environment.decodes == [
        ("one", "warmup-full.wav"),
        ("one", "full.wav"),
        ("two", "full.wav"),
    ]
    assert result["execution_error"] is None and environment.closed
    assert result["metrics"]["word_error_rate"] == 0
    assert result["metrics"]["peak_vram_mb"] is None
    assert all(len(row["warnings"]) == 3 for row in result["videos"])
    assert all(
        window["executed"] for row in result["videos"] for window in row["windows"]
    )
    assert "transcript_repeatability_success_rate" not in result["metrics"]


def test_lifecycle_support_uses_one_observation_not_video_count(environment):
    result = environment.run(items=[_item("one"), _item("two")])
    for name in ("cold_model_load_seconds", "peak_process_tree_ram_mb"):
        report = result["metric_statuses"][name]
        assert report["value"] is not None
        assert report["support"]["scheduled_count"] == 1
        assert report["support"]["contributing_count"] == 1
        assert report["support"]["sample_count"] == 1
        assert name + ".sample_count" not in result["metrics"]
    cpu_vram = result["metric_statuses"]["peak_vram_mb"]
    assert cpu_vram["status"] == "inapplicable"
    assert cpu_vram["support"]["eligible_count"] == 0
    assert cpu_vram["support"]["contributing_count"] == 0


def test_speed_support_counts_completed_independent_sources(environment):
    items = [
        {**_item(name), "independent_source_id": "lecture"} for name in ("one", "two")
    ]
    result = environment.run(items=items)
    for name in (
        "word_error_rate",
        "real_time_factor",
        "p50_warm_audio_processing_latency_seconds",
        "p95_warm_audio_processing_latency_seconds",
    ):
        support = result["metric_statuses"][name]["support"]
        assert support["contributing_count"] == 2
        assert support["contributing_independent_source_count"] == 1
        assert support["sample_count"] == 1
        assert result["metrics"][name + ".sample_count"] == 1


def test_unexecuted_quality_is_incomplete_not_observed_failure(environment):
    environment.fail = "load"
    result = environment.run()
    for name in (
        "word_error_rate",
        "real_time_factor",
        "p50_warm_audio_processing_latency_seconds",
    ):
        assert result["metric_statuses"][name]["status"] == "incomplete"
        assert result["metric_statuses"][name]["support"]["sample_count"] == 0


def test_video_untimed_speech_has_recognition_but_inapplicable_timing(environment):
    item = _item(duration=1.0)
    item["reference_segments"] = []
    result = environment.run(items=[item])
    assert result["metrics"]["word_error_rate"] == 0
    assert result["metrics"]["character_error_rate"] == 0
    for name in ("timestamp_boundary_mae_seconds", "timestamp_alignment_coverage"):
        assert result["metrics"][name] is None
        assert result["metric_statuses"][name]["status"] == "inapplicable"
    assert result["videos"][0]["planned_reference_timed_segment_count"] == 0


def test_standalone_asr_scoring_still_requires_timed_speech_reference():
    item = {
        "id": "speech",
        "reference": "alpha",
        "reference_segments": [],
        "duration_seconds": 1.0,
    }
    with pytest.raises(ValueError, match="reference timestamp segments are missing"):
        score_speech(
            item,
            "alpha",
            [{"text": "alpha", "start": 0.0, "end": 1.0}],
            quality_latency_seconds=0.1,
            alignment_threshold=0.5,
            timestamp_tolerance_seconds=0.1,
        )


@pytest.mark.parametrize(
    "mode", ["untimed_allowed", "timed_mismatch", "untimed_nonspeech"]
)
def test_video_validation_distinguishes_optional_timing_from_missing_review(
    tmp_path, monkeypatch, mode
):
    protocol = VIDEO
    if mode == "untimed_nonspeech":
        raw = copy.deepcopy(VIDEO.meta.worker_payload()["resolved"])
        raw["datasets"].update(
            nonspeech_unit_type="asr_window", required_nonspeech_units=1
        )
        protocol = protocol_from_mapping(raw)
    items = [_item(name, 1.0) for name in ("one", "two")]
    for index, item in enumerate(items):
        source = tmp_path / (item["id"] + ".mp4")
        _wav(source, 1.0 + index * 0.001)
        item.update(
            source_path=str(source),
            asset_sha256=sha256_file(source),
            reference_visual_text=[],
            visual_occurrences=[],
            reference_segments=[],
        )
    if mode == "timed_mismatch":
        items[0]["reference_segments"] = [{"text": "beta", "start": 0.0, "end": 1.0}]
    elif mode == "untimed_nonspeech":
        items[0]["nonspeech_window_indices"] = [0]
    monkeypatch.setattr(
        validation, "load_manifest", lambda *_: SimpleNamespace(samples=items)
    )
    monkeypatch.setattr(validation, "media_duration", lambda *_: 1.0)
    monkeypatch.setattr(
        validation,
        "_media_metadata",
        lambda *_: {"width": 640, "height": 360, "canonical_sha256": "fixture"},
    )
    if mode == "untimed_allowed":
        assert (
            len(
                validation.validate(tmp_path / "manifest", "smoke", protocol)["samples"]
            )
            == 2
        )
    else:
        with pytest.raises(
            ValueError,
            match="does not match its timed segments"
            if mode == "timed_mismatch"
            else "cannot be verified against untimed speech",
        ):
            validation.validate(tmp_path / "manifest", "smoke", protocol)


def test_duplicate_video_ids_rejected_before_loading(environment):
    with pytest.raises(ValueError, match="unique video"):
        environment.run(items=[_item(), _item()])
    assert not environment.requests


@pytest.mark.parametrize(
    "failure,phase",
    [
        ("load", "load"),
        ("invalid_warmup", "warmup"),
        ("warmup", "warmup"),
        ("parameters", "parameters"),
        ("close", "close"),
        ("telemetry", "telemetry"),
    ],
)
def test_lifecycle_failure_keeps_readiness_resources_and_inventory(
    environment, failure, phase
):
    environment.fail = failure
    result = environment.run()
    assert result["execution_error"]["phase"] == phase
    assert environment.closed
    assert result["resource_samples"]
    assert len(result["videos"][0]["windows"]) == 3
    assert (
        result["metrics"]["cold_model_load_seconds"] is None
        if failure == "load"
        else result["metrics"]["cold_model_load_seconds"] >= 0
    )
    if phase in {"load", "warmup"}:
        assert all(
            window["success"] is None and not window["executed"]
            for window in result["videos"][0]["windows"]
        )
        assert result["metrics"]["attempt_failure_rate"] is None
        assert result["metrics"]["word_error_rate"] is None
        assert result["videos"][0]["planned_reference_word_count"] == 3
    else:
        assert result["metrics"]["word_error_rate"] == 0


def test_runtime_construction_failure_retains_unexecuted_windows(
    environment, monkeypatch
):
    def fail(*_):
        raise RuntimeError("pinned snapshot absent")

    monkeypatch.setattr(worker, "build_runtime", fail)
    result = environment.run()
    assert result["execution_error"]["phase"] == "load"
    assert result["videos"][0]["unexecuted_window_count"] == 3


def test_cleanup_failure_does_not_replace_load_error(environment):
    environment.fail = "load_and_close"
    result = environment.run()
    assert result["execution_error"]["phase"] == "load"
    assert "cleanup failure" in result["execution_error"]["close_error"]


def test_preprocessing_failure_has_no_measured_window_latency(environment, monkeypatch):
    slice_wav = worker._slice_wav

    def fail_first(source, destination, **kwargs):
        if destination.stem == "0":
            raise ValueError("injected PCM slicing failure")
        return slice_wav(source, destination, **kwargs)

    monkeypatch.setattr(worker, "_slice_wav", fail_first)
    row = environment.run()["videos"][0]
    assert row["windows"][0]["success"] is None
    assert row["windows"][0]["latency_seconds"] is None
    assert row["windows"][0]["preprocessing_elapsed_seconds"] >= 0
    assert row["failed_window_count"] == 0
    assert environment.requests == ["warmup", "1", "2"]


def test_preflight_error_retains_actual_request_cause(environment):
    environment.fail = "1"
    with pytest.raises(WorkerExecutionError, match="injected request failure") as error:
        environment.run(device="cuda", mode="preflight")
    assert len(error.value.resource_evidence["videos"][0]["windows"]) == 3


def test_decoded_checksum_mismatch_is_preprocessing_not_request_failure(environment):
    item = _item()
    item["canonical_sha256"] = "changed"
    result = environment.run(items=[item])
    assert result["videos"][0]["unexecuted_window_count"] == 3
    assert result["videos"][0]["failed_window_count"] == 0
    assert "checksum differs" in result["videos"][0]["error"]


def test_manifest_duration_tolerance_does_not_invent_pcm_or_fail_valid_tail(
    environment, monkeypatch
):
    decode = worker._decode

    def short(item, destination, protocol):
        command = decode(item, destination, protocol)
        _wav(destination, 0.97)
        return command

    monkeypatch.setattr(worker, "_decode", short)
    row = environment.run(items=[_item(duration=1.0)])["videos"][0]
    assert row["success"]
    assert row["duration_seconds"] == 1.0
    assert row["canonical_duration_seconds"] == 0.97
    assert row["windows"][0]["input_duration_seconds"] == 0.97


def test_decode_failure_is_not_three_failed_requests(environment):
    environment.fail = "decode"
    result = environment.run()
    row = result["videos"][0]
    assert row["failed_window_count"] == 0 and row["unexecuted_window_count"] == 3
    assert result["metrics"]["attempt_failure_rate"] is None
    assert result["metric_statuses"]["attempt_failure_rate"]["status"] == "incomplete"
    assert environment.requests == ["warmup"]
    assert result["ffmpeg_commands"][-1]["error"]


def test_request_failure_keeps_partial_output_but_no_complete_quality(environment):
    environment.fail = "1"
    result = environment.run()
    row = result["videos"][0]
    assert environment.requests == ["warmup", "0", "1", "2"]
    assert row["transcript"] == "alpha alpha" and not row["success"]
    assert row["executed_window_count"] == 3 and row["failed_window_count"] == 1
    assert result["metrics"]["word_error_rate"] is None
    assert result["metrics"]["real_time_factor"] is None
    assert result["metrics"]["attempt_failure_rate"] == pytest.approx(1 / 3)
    assert (
        result["metric_statuses"]["attempt_failure_rate"]["comparison_status"]
        == "complete"
    )


@pytest.mark.parametrize("failure", ["placement_1", "placement_2"])
def test_cuda_placement_failures_are_strict_even_in_preflight(environment, failure):
    environment.fail = failure
    with pytest.raises(WorkerExecutionError) as error:
        environment.run(device="cuda", mode="preflight")
    evidence = error.value.resource_evidence
    assert evidence["placement"]["status"] == "cpu_offload_detected"
    assert evidence["peak_vram_mb"] == 9000.0  # no artificial cap
    assert evidence["resource_samples"] and environment.closed


def test_offline_failure_retains_all_raw_window_outputs(environment, monkeypatch):
    def fail(*_, **__):
        assert not environment.monitoring and environment.closed
        raise RuntimeError("injected evaluator failure")

    monkeypatch.setattr(worker, "score_speech", fail)
    with pytest.raises(WorkerExecutionError) as error:
        environment.run()
    evidence = error.value.resource_evidence
    assert evidence["execution_error"]["phase"] == "offline_evaluation"
    assert evidence["videos"][0]["success"]
    assert len(evidence["videos"][0]["windows"]) == 3
    assert evidence["cold_model_load_seconds"] >= 0


def test_native_boundaries_survive_shift_without_clamping(environment):
    environment.transcript = Transcript(
        "alpha", ({"text": "alpha", "start": 0.0, "end": 1.05},)
    )
    row = environment.run(items=[_item(duration=1.0)])["videos"][0]
    assert row["success"] and row["segments"][0]["end"] == 1.05


def test_stitching_caps_tokens_and_preserves_native_tie_order():
    left = [{"text": " ".join(["ALPHA!"] * 65), "start": 28.0, "end": 30.0}]
    right = [{"text": " ".join(["alpha"] * 65), "start": 28.0, "end": 29.0}]
    stitched = worker._stitch_segments(
        left, right, 64, overlap_start=28.0, overlap_end=30.0
    )
    assert len(stitched) == 66
    assert [row["end"] for row in stitched] == [30.0] * 65 + [29.0]
    repeated = worker._stitch_segments(
        [{"text": "again", "start": 0, "end": 1}],
        [{"text": "again", "start": 28, "end": 29}],
        64,
        overlap_start=28,
        overlap_end=30,
    )
    assert len(repeated) == 2


def test_slice_rejects_truncation_instead_of_silently_shortening(tmp_path):
    source = tmp_path / "source.wav"
    _wav(source, 1.0)
    with pytest.raises(ValueError, match="truncated PCM"):
        worker._slice_wav(
            source,
            tmp_path / "window.wav",
            start=0.75,
            duration=0.5,
            sample_rate_hz=16000,
            channels=1,
            sample_width_bytes=2,
        )


def _artifact(tmp_path, environment, monkeypatch):
    result = environment.run(items=[_item(duration=1.0)])
    monkeypatch.setattr(frozen_asr, "model_identity", lambda *_: ({}, _identity()))
    monkeypatch.setattr(frozen_asr, "run_json_worker", lambda *_args, **_kwargs: result)
    path = tmp_path / "frozen.json"
    frozen_asr.create_frozen_asr_artifact(
        path,
        items=[_item(duration=1.0)],
        manifest_checksum="manifest",
        protocol=VIDEO,
        audio_protocol=AUDIO,
        audio_candidate=CANDIDATE,
        audio_decision_path=tmp_path / "decision",
        device="cpu",
        ffmpeg_version="ffmpeg fixture",
        profile_name="smoke",
        producer_run_id="producer",
    )
    return path


def _load(path, **kwargs):
    return frozen_asr.load_frozen_asr_artifact(
        path,
        manifest_checksum="manifest",
        protocol_checksum=VIDEO.meta.checksum,
        audio_protocol_checksum=AUDIO.meta.checksum,
        timestamp_tolerance_seconds=AUDIO.timestamp_tolerance_seconds,
        sample_ids=["one"],
        expected_identity=_identity(),
        expected_device="cpu",
        expected_profile="smoke",
        expected_durations={"one": 1.0},
        expected_ffmpeg_version="ffmpeg fixture",
        **kwargs,
    )


def _reseal(path, payload):
    payload.pop("artifact_content_sha256", None)
    payload["artifact_content_sha256"] = stable_hash(payload)
    atomic_write_json(path, payload)


def test_complete_artifact_reuse_and_immutable_creation(
    tmp_path, environment, monkeypatch
):
    path = _artifact(tmp_path, environment, monkeypatch)
    artifact, checksum = _load(path, expected_checksum=sha256_file(path))
    assert artifact["run_id"] == "producer" and len(checksum) == 64
    original = path.read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        _artifact(tmp_path, environment, monkeypatch)
    assert path.read_bytes() == original


def test_reuse_matches_sealed_canonical_checksum_by_sample_id(
    tmp_path, environment, monkeypatch
):
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    expected = {row["sample_id"]: row["canonical_sha256"] for row in payload["videos"]}
    artifact, _ = _load(path, expected_canonical_checksums=expected)
    assert artifact["videos"][0]["canonical_sha256"] == expected["one"]


@pytest.mark.parametrize(
    "expected", [{"one": "0" * 64}, {}, {"another": "0" * 64}, {"one": None}, []]
)
def test_reuse_rejects_wrong_missing_or_invalid_sealed_canonical_checksums(
    tmp_path, environment, monkeypatch, expected
):
    path = _artifact(tmp_path, environment, monkeypatch)
    with pytest.raises(ValueError, match="canonical checksum"):
        _load(path, expected_canonical_checksums=expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("canonical_sha256", None),
        ("canonical_sha256", "not-a-digest"),
        ("canonical_duration_seconds", None),
        ("canonical_duration_seconds", 0),
        ("canonical_duration_seconds", 1.5),
    ],
)
def test_reusable_rows_require_complete_canonical_provenance_without_expected_map(
    tmp_path, environment, monkeypatch, field, value
):
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    payload["videos"][0][field] = value
    _reseal(path, payload)
    with pytest.raises(ValueError, match="complete canonical checksum/duration"):
        _load(path)


@pytest.mark.parametrize("field", ["canonical_sha256", "canonical_duration_seconds"])
def test_missing_canonical_fields_are_not_a_legacy_compatibility_branch(
    tmp_path, environment, monkeypatch, field
):
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    payload["videos"][0].pop(field)
    _reseal(path, payload)
    with pytest.raises(ValueError, match="complete execution/support records"):
        _load(path)


@pytest.mark.parametrize("failure", ["load", "warmup", "decode"])
def test_canonical_map_does_not_erase_fatal_or_preprocessing_evidence(
    tmp_path, environment, monkeypatch, failure
):
    environment.fail = failure
    path = _artifact(tmp_path, environment, monkeypatch)
    with pytest.raises(WorkerExecutionError) as error:
        _load(path, expected_canonical_checksums={"one": "0" * 64})
    row = error.value.resource_evidence["videos"][0]
    assert row["canonical_sha256"] is None
    assert row["canonical_duration_seconds"] is None
    assert row["unexecuted_window_count"] == 1
    assert row["failed_window_count"] == 0
    assert row["windows"][0]["success"] is None


def test_request_failures_retain_and_verify_decoded_checksum(
    tmp_path, environment, monkeypatch
):
    environment.fail = "0"
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    canonical = payload["videos"][0]["canonical_sha256"]
    artifact, _ = _load(path, expected_canonical_checksums={"one": canonical})
    assert artifact["metrics"]["attempt_failure_rate"] == 1
    assert artifact["metrics"]["word_error_rate"] is None
    assert artifact["videos"][0]["windows"][0]["error"]
    with pytest.raises(
        ValueError, match="canonical checksum differs from sealed inputs"
    ):
        _load(path, expected_canonical_checksums={"one": "0" * 64})


def test_artifact_content_mutation_is_detected_without_external_checksum(
    tmp_path, environment, monkeypatch
):
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    payload["metrics"]["word_error_rate"] = 0.9
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="content checksum"):
        _load(path)


def test_artifact_checksum_is_mandatory_not_a_legacy_optional_gate(
    tmp_path, environment, monkeypatch
):
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    payload.pop("artifact_content_sha256")
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="content checksum"):
        _load(path)


def test_fully_accounted_window_failures_can_be_reused_as_reliability_evidence(
    tmp_path, environment, monkeypatch
):
    environment.fail = "0"
    path = _artifact(tmp_path, environment, monkeypatch)
    artifact, _ = _load(path)
    assert artifact["metrics"]["attempt_failure_rate"] == 1
    assert artifact["metrics"]["word_error_rate"] is None
    assert artifact["videos"][0]["segments"] == []


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p["parameters"].update(dtype="float16"), "runtime parameters"),
        (
            lambda p: p["parameters"].update(model_revision="different"),
            "runtime parameters",
        ),
        (lambda p: p["metrics"].update(peak_vram_mb=0.0), "CPU VRAM"),
        (
            lambda p: p["videos"][0]["windows"][0].update(latency_seconds=True),
            "window latency",
        ),
        (
            lambda p: p["videos"][0].update(
                transcript="beta", segments=[{"text": "beta", "start": 0.0, "end": 1.0}]
            ),
            "deterministic window stitching",
        ),
        (
            lambda p: p.update(model_lock_fingerprint="different"),
            "model lock fingerprint",
        ),
        (
            lambda p: p.update(submodels=[{"revision": "different"}]),
            "submodels mismatch",
        ),
    ],
)
def test_reuse_rejects_full_identity_and_output_drift(
    tmp_path, environment, monkeypatch, mutation, match
):
    path = _artifact(tmp_path, environment, monkeypatch)
    payload = json.loads(path.read_text())
    mutation(payload)
    _reseal(path, payload)
    with pytest.raises(ValueError, match=match):
        _load(path)


def test_fatal_artifact_reuse_keeps_error_envelope(tmp_path, environment, monkeypatch):
    environment.fail = "warmup"
    path = _artifact(tmp_path, environment, monkeypatch)
    with pytest.raises(WorkerExecutionError) as error:
        _load(path)
    assert error.value.resource_evidence["execution_error"]["phase"] == "warmup"
    assert error.value.resource_evidence["videos"][0]["unexecuted_window_count"] == 1


def test_creation_preserves_worker_transport_failure_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen_asr, "model_identity", lambda *_: ({}, _identity()))
    evidence = {
        "videos": [{"sample_id": "one"}],
        "worker_result": {"windows": ["retained"]},
    }

    def fail(*_, **__):
        raise WorkerExecutionError("offline failure", evidence)

    monkeypatch.setattr(frozen_asr, "run_json_worker", fail)
    with pytest.raises(WorkerExecutionError) as error:
        frozen_asr.create_frozen_asr_artifact(
            tmp_path / "frozen.json",
            items=[_item()],
            manifest_checksum="manifest",
            protocol=VIDEO,
            audio_protocol=AUDIO,
            audio_candidate=CANDIDATE,
            audio_decision_path=tmp_path / "decision",
            device="cpu",
            ffmpeg_version="fixture",
            profile_name="smoke",
        )
    assert error.value.resource_evidence["worker_result"] == evidence["worker_result"]
    assert error.value.resource_evidence["frozen_asr_identity"] == _identity()


def test_reviewed_nonspeech_windows_keep_failed_units_and_labels(environment):
    raw = copy.deepcopy(VIDEO.meta.worker_payload()["resolved"])
    raw["datasets"].update(nonspeech_unit_type="asr_window", required_nonspeech_units=1)
    protocol = protocol_from_mapping(raw)
    item = _item()
    item.update(nonspeech_window_indices=[0, 1], conditions=["environmental_sound"])
    environment.fail = "1"
    result = environment.run(items=[item], protocol=protocol)
    units = result["videos"][0]["nonspeech_units"]
    assert len(units) == 2 and units[1]["false_transcription"] is None
    assert units[0]["labels"] == ["environmental_sound"]
    assert result["metrics"]["nonspeech_false_transcription_rate"] == 1
    assert (
        result["metric_statuses"]["nonspeech_false_transcription_rate"]["support"][
            "scheduled_count"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "failure,expected_success,failed,unavailable,incomplete",
    [
        ("load", None, 0, 0, 1),
        ("warmup", None, 0, 0, 1),
        ("0", False, 1, 1, 0),
        (None, True, 0, 0, 0),
    ],
)
def test_whole_video_nonspeech_support_distinguishes_unexecuted_and_failed(
    environment, failure, expected_success, failed, unavailable, incomplete
):
    raw = copy.deepcopy(VIDEO.meta.worker_payload()["resolved"])
    raw["datasets"].update(
        nonspeech_unit_type="whole_video", required_nonspeech_units=1
    )
    protocol = protocol_from_mapping(raw)
    item = _item(duration=1.0)
    item.update(reference_transcript="", reference_segments=[], nonspeech_reviewed=True)
    environment.fail = failure
    result = environment.run(items=[item], protocol=protocol)
    unit = result["videos"][0]["nonspeech_units"][0]
    assert unit["success"] is expected_success
    assert unit["executed"] is (expected_success is not None)
    assert unit["false_transcription"] is (True if expected_success is True else None)
    support = result["metric_statuses"]["nonspeech_false_transcription_rate"]["support"]
    assert support["failed_count"] == failed
    assert support["unavailable_count"] == unavailable
    assert support["incomplete_count"] == incomplete
    assert support["contributing_count"] == int(expected_success is True)
    assert result["metrics"]["nonspeech_false_transcription_rate"] == (
        1 if expected_success is True else None
    )


def test_unexecuted_nonspeech_artifact_load_keeps_null_unit_outcome(
    tmp_path, environment, monkeypatch
):
    raw = copy.deepcopy(VIDEO.meta.worker_payload()["resolved"])
    raw["datasets"].update(
        nonspeech_unit_type="whole_video", required_nonspeech_units=1
    )
    protocol = protocol_from_mapping(raw)
    item = _item(duration=1.0)
    item.update(reference_transcript="", reference_segments=[], nonspeech_reviewed=True)
    environment.fail = "load"
    result = environment.run(items=[item], protocol=protocol)
    monkeypatch.setattr(frozen_asr, "model_identity", lambda *_: ({}, _identity()))
    monkeypatch.setattr(frozen_asr, "run_json_worker", lambda *_args, **_kwargs: result)
    path = tmp_path / "frozen.json"
    frozen_asr.create_frozen_asr_artifact(
        path,
        items=[item],
        manifest_checksum="manifest",
        protocol=protocol,
        audio_protocol=AUDIO,
        audio_candidate=CANDIDATE,
        audio_decision_path=tmp_path / "decision",
        device="cpu",
        ffmpeg_version="ffmpeg fixture",
        profile_name="smoke",
    )
    with pytest.raises(WorkerExecutionError) as error:
        frozen_asr.load_frozen_asr_artifact(
            path,
            manifest_checksum="manifest",
            protocol_checksum=protocol.meta.checksum,
            audio_protocol_checksum=AUDIO.meta.checksum,
            timestamp_tolerance_seconds=AUDIO.timestamp_tolerance_seconds,
            sample_ids=["one"],
            expected_canonical_checksums={"one": "0" * 64},
        )
    unit = error.value.resource_evidence["videos"][0]["nonspeech_units"][0]
    assert unit["executed"] is False and unit["success"] is None
    assert error.value.resource_evidence["execution_error"]["phase"] == "load"


def test_source_bootstrap_keeps_failed_video_windows_inside_group(environment):
    result = environment.run(items=[_item("one"), _item("two")])
    rows = result["videos"]
    rows[1]["success"] = rows[1]["first_attempt_success"] = False
    rows[1]["windows"][1].update(success=False, error="retained failure")
    rows[1]["source_group_id"] = rows[0]["source_group_id"]
    seen = []

    def statistic(draw):
        seen.append([row["sample_id"] for row in draw])
        return worker._statistics(draw)

    intervals = bootstrap_sources(
        rows, statistic, resamples=8, seed=42, confidence=0.95, minimum_sources=1
    )
    assert all(draw == ["one", "two"] for draw in seen)
    assert intervals["word_error_rate"]["independent_source_count"] == 1
    assert "cold_model_load_seconds" not in intervals


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("reference_transcript", None, "reference_transcript"),
        ("reference_segments", None, "reference_segments"),
        ("nonspeech_reviewed", "true", "review flag must be boolean"),
        ("nonspeech_window_indices", [0, 0], "actual unique ASR windows"),
        ("nonspeech_window_indices", [{}], "actual unique ASR windows"),
        ("nonspeech_window_indices", [0], "overlaps spoken evidence"),
    ],
)
def test_validation_rejects_missing_annotations_and_unreviewable_nonspeech(
    tmp_path, monkeypatch, field, value, match
):
    raw = copy.deepcopy(VIDEO.meta.worker_payload()["resolved"])
    raw["datasets"].update(nonspeech_unit_type="asr_window", required_nonspeech_units=1)
    protocol = protocol_from_mapping(raw)
    items = [_item(name, 1.0) for name in ("one", "two")]
    for item in items:
        source = tmp_path / (item["id"] + ".mp4")
        _wav(source, 1.0)
        item.update(
            source_path=str(source),
            asset_sha256=sha256_file(source),
            reference_visual_text=[],
            visual_occurrences=[],
        )
    items[0][field] = value
    monkeypatch.setattr(
        validation, "load_manifest", lambda *_: SimpleNamespace(samples=items)
    )
    monkeypatch.setattr(validation, "media_duration", lambda *_: 1.0)
    with pytest.raises(ValueError, match=match):
        validation.validate(tmp_path / "manifest", "smoke", protocol)


def test_media_metadata_rejects_missing_audio_not_reviewed_silence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(
                {"streams": [{"codec_type": "video", "width": 640, "height": 360}]}
            )
        ),
    )
    with pytest.raises(ValueError, match="missing audio is not reviewed silence"):
        validation._media_metadata(tmp_path / "source.mp4", 1.0, VIDEO)


@pytest.mark.parametrize("extra", ["video", "audio"])
def test_media_validation_rejects_ambiguous_stream_selection(
    tmp_path, monkeypatch, extra
):
    streams = [
        {"codec_type": "video", "width": 640, "height": 360},
        {"codec_type": "audio"},
        {"codec_type": extra, "width": 1920, "height": 1080},
    ]
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps({"streams": streams})
        ),
    )
    with pytest.raises(ValueError, match="one visual and one audio track"):
        validation._media_metadata(tmp_path / "source.mp4", 1.0, VIDEO)


def test_silent_canonical_audio_has_dimensions_duration_and_checksum(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validation.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "width": 640, "height": 360},
                        {"codec_type": "audio"},
                    ]
                }
            )
        ),
    )

    def decode(source, destination, **kwargs):
        _wav(destination, 1.0)
        return ["ffmpeg", str(source), str(destination)]

    monkeypatch.setattr(validation, "decode_canonical_audio", decode)
    metadata = validation._media_metadata(tmp_path / "source.mp4", 1.0, VIDEO)
    assert metadata["width"] == 640 and metadata["height"] == 360
    assert metadata["canonical_duration_seconds"] == 1.0
    assert (
        len(metadata["canonical_sha256"]) == 64 and metadata["audio_stream_count"] == 1
    )


def test_validation_seal_binds_media_properties_and_annotation_changes(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    fixture = root / "data/benchmarks/extraction/smoke.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    samples = [row for row in payload["samples"] if row["kind"] == "video"]
    payload["samples"] = samples
    payload["checksum"] = stable_hash(samples)
    # Manifest checksums use UTF-8 with ensure_ascii=False, identical for these fixtures.
    path = tmp_path / "manifest.json"
    atomic_write_json(path, payload)
    monkeypatch.setattr(validation, "ffmpeg_version", lambda: "fixture version")
    monkeypatch.setattr(
        validation,
        "media_duration",
        lambda source: next(
            row["duration_seconds"]
            for row in samples
            if source == root / row["source_path"]
        ),
    )
    monkeypatch.setattr(
        validation,
        "_media_metadata",
        lambda *_: {
            "width": 640,
            "height": 360,
            "canonical_sha256": "fixture-hash",
            "canonical_duration_seconds": 1.0,
        },
    )
    report_root = tmp_path / "reports"
    _, report = validation.prepare_report(path, "smoke", VIDEO, report_root=report_root)
    assert report["status"] == "passed"
    _, verified = validation.verified_inputs(
        path, "smoke", VIDEO, report_root=report_root
    )
    assert verified["metadata"]["samples"][samples[0]["id"]]["width"] == 640
    samples[0]["reference_transcript"] = "changed"
    payload["checksum"] = stable_hash(samples)
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="Missing or stale data validation"):
        validation.verified_inputs(path, "smoke", VIDEO, report_root=report_root)
