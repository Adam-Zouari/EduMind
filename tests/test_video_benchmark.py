from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import atomic_write_json
from experiments.benchmarks.extraction.audio.protocol import (
    load_protocol as load_audio_protocol,
)
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol as load_document_protocol,
)
from experiments.benchmarks.extraction.video import runner as video_runner
from experiments.benchmarks.extraction.video.candidates import (
    all_candidates,
    frame_command,
    parse_candidate,
)
from experiments.benchmarks.extraction.video.frozen_asr import load_frozen_asr_artifact
from experiments.benchmarks.extraction.video.frozen_asr_worker import (
    _slice_wav,
    _stitch_segments,
    _window_starts,
)
from experiments.benchmarks.extraction.video.metrics import (
    aggregate_quality,
    score_video,
)
from experiments.benchmarks.extraction.video.protocol import load_protocol
from experiments.benchmarks.extraction.video.runner import _candidates

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PROTOCOL = load_protocol()
AUDIO_PROTOCOL = load_audio_protocol()
DOCUMENT_PROTOCOL = load_document_protocol()


def test_video_smoke_uses_a_device_specific_frozen_asr_artifact(monkeypatch) -> None:
    commands = []

    def run(command, **_options):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(video_runner.subprocess, "run", run)
    monkeypatch.setattr(
        video_runner.sys,
        "argv",
        ["run.py", "--profile", "smoke", "--phase", "all"],
    )
    arguments = SimpleNamespace(phase="all", frozen_asr=None)
    assert video_runner._run_smoke_devices(arguments, ("cpu", "cuda")) == 0
    assert len(commands) == 2
    assert [command[command.index("--device") + 1] for command in commands] == [
        "cpu",
        "cuda",
    ]
    assert all("--frozen-asr" not in command for command in commands)


def test_all_nine_video_configurations_include_frame_zero_and_vfr() -> None:
    candidates = all_candidates(VIDEO_PROTOCOL, 0.40)
    assert len(candidates) == len(set(candidates)) == 9
    assert candidates[:3] == ("video-fixed-5s", "video-fixed-10s", "video-fixed-20s")
    assert candidates[3:6] == (
        "video-scene-0.30",
        "video-scene-0.40",
        "video-scene-0.50",
    )
    for value in candidates:
        parsed = parse_candidate(value, VIDEO_PROTOCOL)
        assert "eq(n,0)" in parsed.ffmpeg_filter
        command = frame_command(parsed, "video.mp4", "frame-%05d.png")
        assert command[command.index("-fps_mode") + 1] == "vfr"


def test_video_protocol_is_versioned_and_rejects_timestamp_tolerance(tmp_path) -> None:
    assert VIDEO_PROTOCOL.window_length_seconds == 30
    assert VIDEO_PROTOCOL.overlap_seconds == 2
    assert VIDEO_PROTOCOL.stitching["method"] == "normalized_suffix_prefix"
    assert VIDEO_PROTOCOL.hybrid_threshold("smoke") == 0.40
    with pytest.raises(ValueError, match="selected_scene_threshold"):
        VIDEO_PROTOCOL.hybrid_threshold("development")
    bad_path = tmp_path / "bad-tolerance.yaml"
    payload = dict(VIDEO_PROTOCOL.meta.resolved)
    payload["visual_text"] = {
        **payload["visual_text"],
        "frame_timestamp_tolerance_seconds": 0.01,
    }
    atomic_write_json(
        bad_path,
        payload,
    )
    with pytest.raises(ValueError, match="timestamp tolerance"):
        load_protocol(bad_path)


def test_video_metrics_use_distinct_content_and_one_to_one_occurrences() -> None:
    item = {
        "id": "video",
        "duration_seconds": 10.0,
        "reference_visual_text": ["Alpha", "Alpha"],
        "visual_occurrences": [
            {"id": "occ-1", "text": "Alpha", "start": 0.0, "end": 5.0},
            {"id": "occ-2", "text": "Alpha", "start": 0.0, "end": 5.0},
        ],
    }
    settings = {
        "content_f1_threshold": 0.5,
        "frame_timestamp_tolerance_seconds": 0.0,
    }
    scores = score_video(
        item,
        [{"text": "Alpha", "timestamp": 1.0}],
        settings,
    )
    assert scores["visual_content_precision"] == 1.0
    assert scores["visual_content_recall"] == 1.0
    assert scores["duplicate_visual_text_rate"] == 0.0
    assert scores["timed_visual_occurrence_coverage"] == 0.5
    assert scores["mean_visual_first_detection_delay_seconds"] == 1.0

    empty = score_video(item, [], settings)
    assert empty["duplicate_visual_text_rate"] is None
    assert empty["mean_visual_first_detection_delay_seconds"] is None
    assert empty["timed_visual_occurrence_coverage"] == 0.0


def test_video_occurrence_matching_never_leaks_outside_verified_interval() -> None:
    item = {
        "id": "video",
        "duration_seconds": 3.0,
        "reference_visual_text": ["alpha"],
        "visual_occurrences": [
            {"id": "occ-3", "text": "alpha", "start": 1.0, "end": 2.0}
        ],
    }
    settings = {
        "content_f1_threshold": 0.5,
        "frame_timestamp_tolerance_seconds": 0.0,
    }
    scores = score_video(item, [{"text": "alpha", "timestamp": 2.01}], settings)
    assert scores["timed_visual_occurrence_coverage"] == 0.0
    assert scores["mean_visual_first_detection_delay_seconds"] is None


def test_video_occurrence_matching_minimizes_raw_detection_delay() -> None:
    item = {
        "id": "video",
        "duration_seconds": 100.0,
        "reference_visual_text": ["alpha"],
        "visual_occurrences": [
            {"id": "occ-4", "text": "alpha", "start": 0.0, "end": 100.0},
            {"id": "occ-5", "text": "alpha", "start": 9.0, "end": 10.01},
        ],
    }
    settings = {
        "content_f1_threshold": 0.5,
        "frame_timestamp_tolerance_seconds": 0.0,
    }
    scores = score_video(item, [{"text": "alpha", "timestamp": 10.0}], settings)
    assert scores["timed_visual_occurrence_coverage"] == 0.5
    assert scores["mean_visual_first_detection_delay_seconds"] == 1.0


def test_video_window_stitching_and_bounds_are_deterministic() -> None:
    assert _window_starts(65.0, 30.0, 2.0) == [0.0, 28.0, 56.0]
    stitched = _stitch_segments(
        [{"id": "occ-6", "text": "alpha beta", "start": 0.0, "end": 2.0}],
        [{"id": "occ-7", "text": "beta gamma", "start": 1.0, "end": 3.0}],
        8,
    )
    assert [row["text"] for row in stitched] == ["alpha", "beta", "gamma"]


def test_frozen_asr_windows_slice_one_canonical_decode(tmp_path) -> None:
    import wave

    source = ROOT / "data/benchmarks/fixtures/extraction/audio-01.wav"
    destination = tmp_path / "window.wav"
    _slice_wav(
        source,
        destination,
        start=0.25,
        duration=0.5,
        sample_rate_hz=int(AUDIO_PROTOCOL.audio["sample_rate_hz"]),
        channels=int(AUDIO_PROTOCOL.audio["channels"]),
        sample_width_bytes=int(AUDIO_PROTOCOL.audio["sample_width_bytes"]),
    )
    with wave.open(str(destination), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16_000
        assert audio.getsampwidth() == 2
        assert audio.getnframes() == 8_000


def test_video_aggregates_macro_average_video_values() -> None:
    settings = {
        "content_f1_threshold": 0.5,
        "frame_timestamp_tolerance_seconds": 0.0,
    }
    many = score_video(
        {
            "id": "many",
            "duration_seconds": 5.0,
            "reference_visual_text": ["A", "B", "C"],
            "visual_occurrences": [
                {"id": "occ-8", "text": "A", "start": 0.0, "end": 2.0}
            ],
        },
        [{"text": "A\nB\nC", "timestamp": 1.0}],
        settings,
    )
    one = score_video(
        {
            "id": "one",
            "duration_seconds": 5.0,
            "reference_visual_text": ["D"],
            "visual_occurrences": [
                {"id": "occ-9", "text": "D", "start": 0.0, "end": 2.0}
            ],
        },
        [],
        settings,
    )
    aggregate = aggregate_quality([many, one])
    assert aggregate["visual_content_recall"] == 0.5
    assert aggregate["timed_visual_occurrence_coverage"] == 0.5


def test_frozen_asr_artifact_is_reused_only_for_matching_protocol(tmp_path) -> None:
    path = tmp_path / "frozen.json"
    atomic_write_json(path, _frozen_payload("one"))
    artifact, checksum = load_frozen_asr_artifact(
        path,
        manifest_checksum="manifest",
        protocol_checksum=VIDEO_PROTOCOL.meta.checksum,
        audio_protocol_checksum=AUDIO_PROTOCOL.meta.checksum,
        timestamp_tolerance_seconds=0.1,
        sample_ids=["one"],
    )
    assert artifact["run_id"] == "asr-run"
    assert len(checksum) == 64
    with pytest.raises(ValueError, match="protocol checksum"):
        load_frozen_asr_artifact(
            path,
            manifest_checksum="manifest",
            protocol_checksum="different",
            audio_protocol_checksum=AUDIO_PROTOCOL.meta.checksum,
            timestamp_tolerance_seconds=0.1,
            sample_ids=["one"],
        )


def test_video_profiles_validate_candidate_sources() -> None:
    smoke = SimpleNamespace(profile="smoke", phase="all", shortlist=None)
    candidates, decisions = _candidates(smoke, VIDEO_PROTOCOL)
    assert candidates == all_candidates(VIDEO_PROTOCOL, 0.40)
    assert decisions == {}
    development_hybrid = SimpleNamespace(
        profile="development", phase="hybrid", shortlist=None
    )
    with pytest.raises(ValueError, match="selected_scene_threshold"):
        _candidates(development_hybrid, VIDEO_PROTOCOL)
    locked = SimpleNamespace(profile="locked", phase="all", shortlist=None)
    with pytest.raises(ValueError, match="shortlist"):
        _candidates(locked, VIDEO_PROTOCOL)


def test_visual_worker_has_no_asr_execution_dependency() -> None:
    source = (
        ROOT / "experiments/benchmarks/extraction/video/visual_worker.py"
    ).read_text(encoding="utf-8")
    assert "build_runtime" not in source
    assert "audio_candidate" not in source


def test_application_video_defaults_and_request_overrides_use_shared_selector() -> None:
    from edumind.common.config import load_settings
    from edumind.extraction import SourceKind
    from edumind.extraction.extractors.video import VideoExtractor
    from edumind.extraction.pipeline import build_default_registry
    from edumind.extraction.video_policy import frame_command as application_command

    configured = build_default_registry(
        load_settings(ROOT / "config/base.yaml")
    ).create("video-hybrid", SourceKind.VIDEO)
    configured_policy = configured._keyframe_policy({})
    assert configured_policy.strategy == "hybrid"
    assert configured_policy.scene_threshold == 0.35
    assert configured_policy.maximum_gap_seconds == 10

    extractor = VideoExtractor(
        "hybrid",
        fixed_interval_seconds=10,
        scene_threshold=0.35,
        maximum_hybrid_gap_seconds=10,
    )
    default = extractor._keyframe_policy({})
    assert default.strategy == "hybrid"
    assert default.scene_threshold == 0.35
    overridden = extractor._keyframe_policy(
        {
            "keyframe_strategy": "fixed",
            "fixed_interval_seconds": 5,
        }
    )
    command = application_command(overridden, "video.mp4", "frame-%05d.png")
    assert "eq(n,0)" in command[command.index("-vf") + 1]
    assert command[command.index("-fps_mode") + 1] == "vfr"
    assert "gte(t-prev_selected_t,5)" in command[command.index("-vf") + 1]


def test_visual_worker_uses_current_candidate_temp_root(monkeypatch, tmp_path) -> None:
    from experiments.benchmarks.common import process
    from experiments.benchmarks.extraction.video import runner

    current = tmp_path / "current-candidate"
    stale = tmp_path / "removed-previous-candidate"
    current.mkdir()
    monkeypatch.setenv("TEMP", str(current))
    monkeypatch.setattr(process.tempfile, "tempdir", str(stale))

    def fake_popen(command, **_kwargs):
        output = Path(command[-1])
        assert current in output.parents
        output.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, communicate=lambda **_kwargs: ("", ""))

    monkeypatch.setattr(process.subprocess, "Popen", fake_popen)
    assert (
        runner._run_visual_worker(
            "video-fixed-5s",
            [],
            device="cpu",
            protocol=VIDEO_PROTOCOL.meta.worker_payload(),
        )
        == {}
    )


def test_mocked_visual_phase_runs_through_artifact_writer(
    monkeypatch, tmp_path
) -> None:
    from experiments.benchmarks.common.runner import run_benchmark as real_run_benchmark
    from experiments.benchmarks.extraction.video import runner

    manifest_path = tmp_path / "manifest.json"
    frozen_path = tmp_path / "frozen.json"
    for path in (manifest_path, frozen_path):
        atomic_write_json(path, {"name": path.stem})
    monkeypatch.setattr(
        runner,
        "load_selected_model_lock",
        lambda *_args, **_kwargs: {
            "docling-standard": {
                "revision": "2.117.0",
                "selection_revision": "2.117.0",
                "model_path": str(tmp_path),
                "model_cache_manifest_sha256": "cache",
                "prepared_components": ["layout", "tableformer", "rapidocr"],
            }
        },
    )
    atomic_write_json(frozen_path, _frozen_payload("video"))
    item = {
        "id": "video",
        "duration_seconds": 1.0,
        "kind": "video",
        "source_path": str(tmp_path / "video.mp4"),
    }
    monkeypatch.setattr(runner, "verified_inputs", lambda *args: (manifest_path, {}))
    monkeypatch.setattr(
        runner,
        "load_manifest",
        lambda *args: SimpleNamespace(
            name="smoke", fingerprint="manifest", samples=(item,)
        ),
    )
    monkeypatch.setattr(runner, "model_identity", lambda *args: ({}, {}))
    monkeypatch.setattr(
        runner,
        "_run_visual_worker",
        lambda *_args, **_kwargs: {
            "samples": [
                {
                    "sample_id": "video",
                    "visual_content_precision": 1.0,
                    "visual_content_recall": 1.0,
                    "visual_content_f1": 1.0,
                    "mean_visual_first_detection_delay_seconds": 0.0,
                    "timed_visual_occurrence_coverage": 1.0,
                    "duplicate_visual_text_rate": 0.0,
                    "quality_latency_seconds": 0.1,
                    "selected_frame_count": 1,
                }
            ],
            "operational": {
                "visual_real_time_factor": 0.1,
                "p50_warm_visual_latency_seconds": 0.1,
                "p95_warm_visual_latency_seconds": 0.1,
                "cold_visual_pipeline_load_seconds": 0.1,
                "peak_visual_process_tree_ram_mb": 1.0,
                "peak_visual_vram_mb": 0.0,
                "mean_selected_frames_per_video": 1.0,
            },
            "metrics": {
                "visual_content_precision": 1.0,
                "visual_content_recall": 1.0,
                "visual_content_f1": 1.0,
                "mean_visual_first_detection_delay_seconds": 0.0,
                "timed_visual_occurrence_coverage": 1.0,
                "duplicate_visual_text_rate": 0.0,
                "repeatability_success_rate": None,
                "attempt_failure_rate": 0.0,
            },
            "parameters": {"candidate": "video-fixed-5s"},
            "intervals": {},
            "timings": [{"sample_id": "video", "latency_seconds": 0.1}],
            "ffmpeg_commands": [{"command": ["ffmpeg"]}],
            "outputs": {"attempts": []},
            "resource_samples": [],
            "gpu_identity": {},
        },
    )

    def scoped_run(*args, **kwargs):
        return real_run_benchmark(*args, **kwargs, artifact_root=tmp_path / "artifacts")

    monkeypatch.setattr(runner, "run_benchmark", scoped_run)
    result = runner.run_visual_benchmark(
        "smoke",
        ("video-fixed-5s",),
        items=[item],
        manifest_path=manifest_path,
        manifest_name="smoke",
        manifest_checksum="manifest",
        protocol=VIDEO_PROTOCOL,
        audio_protocol=AUDIO_PROTOCOL,
        document_protocol=DOCUMENT_PROTOCOL,
        frozen_asr_path=frozen_path,
        audio_candidate="whisper-small-en-control",
        audio_decision=manifest_path,
        image_candidate="docling-standard",
        image_decision=None,
        decision_files={},
        device="cpu",
        ffmpeg_version="ffmpeg test",
        no_mlflow=True,
    )
    assert result.complete, result.completion_problems
    assert len(result.candidates) == 2
    assert result.candidates[0].parameters["run_type"] == "preparation"
    assert "word_error_rate" not in result.candidates[1].metrics
    assert (result.artifact_directory / "summary.json").is_file()
    assert (result.artifact_directory / "document_protocol.json").is_file()


def _frozen_payload(sample_id):
    from experiments.benchmarks.extraction.video.frozen_asr import execution_identity
    from experiments.benchmarks.extraction.video.frozen_asr_worker import (
        METRIC_DIRECTIONS,
    )

    return {
        "schema_version": 2,
        "artifact_type": "FrozenASRArtifact",
        "execution_identity": execution_identity(),
        "run_id": "asr-run",
        "manifest_checksum": "manifest",
        "protocol_checksum": VIDEO_PROTOCOL.meta.checksum,
        "audio_protocol_checksum": AUDIO_PROTOCOL.meta.checksum,
        "protocol_version": VIDEO_PROTOCOL.meta.version,
        "audio_protocol_version": AUDIO_PROTOCOL.meta.version,
        "protocol": VIDEO_PROTOCOL.meta.worker_payload(),
        "audio_protocol": AUDIO_PROTOCOL.meta.worker_payload(),
        "model_decision_fingerprint": "decision",
        "audio_candidate": "asr",
        "model_path": "model",
        "model_revision": "revision",
        "selection_revision": "selection",
        "model_cache_manifest_sha256": "cache",
        "ffmpeg_version": "ffmpeg test",
        "ffmpeg_commands": [],
        "device": "cpu",
        "profile": "smoke",
        "metrics": {
            name: None if name == "peak_vram_mb" else 0.0 for name in METRIC_DIRECTIONS
        },
        "intervals": {},
        "parameters": {"device": "cpu"},
        "resource_samples": [],
        "gpu_identity": {},
        "videos": [
            {
                "sample_id": sample_id,
                "duration_seconds": 1.0,
                "latency_seconds": 0.1,
                "success": True,
                "error": None,
                "scheduled_window_count": 1,
                "transcript": "alpha",
                "segments": [{"text": "alpha", "start": 0.0, "end": 1.0}],
                "windows": [
                    {
                        "index": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "success": True,
                        "error": None,
                        "latency_seconds": 0.1,
                        "transcript": "alpha",
                        "segments": [{"text": "alpha", "start": 0.0, "end": 1.0}],
                    }
                ],
            }
        ],
    }


def test_duplicate_text_is_exact_and_scoped_to_visibility_occurrence():
    item = {
        "id": "chapter",
        "duration_seconds": 30,
        "reference_visual_text": ["Chapter 1"],
        "visual_occurrences": [
            {"id": "first", "text": "Chapter 1", "start": 10, "end": 20},
            {"id": "again", "text": "Chapter 1", "start": 20, "end": 30},
        ],
    }
    settings = VIDEO_PROTOCOL.occurrence_matching
    row = score_video(
        item,
        [
            {"timestamp": 10, "text": "Chapter"},
            {"timestamp": 12, "text": "Chapter 1"},
            {"timestamp": 14, "text": "Chapter 1"},
        ],
        settings,
    )
    assert row["duplicate_visual_text_rate"] == pytest.approx(1 / 3)
    reappearance = score_video(
        item,
        [
            {"timestamp": 12, "text": "Chapter 1"},
            {"timestamp": 22, "text": "Chapter 1"},
        ],
        settings,
    )
    assert reappearance["duplicate_visual_text_rate"] == 0
    assert reappearance["timed_visual_occurrence_coverage"] == 1
    unassigned = score_video(item, [{"timestamp": 1, "text": "Chapter 1"}], settings)
    assert unassigned["duplicate_visual_text_rate"] is None


def test_text_free_video_has_perfect_content_but_no_timed_task():
    row = score_video(
        {
            "id": "blank",
            "duration_seconds": 1,
            "reference_visual_text": [],
            "visual_occurrences": [],
        },
        [],
        VIDEO_PROTOCOL.occurrence_matching,
    )
    assert row["visual_content_f1"] == 1
    assert row["timed_visual_occurrence_coverage"] is None
    assert row["duplicate_visual_text_rate"] is None


@pytest.mark.parametrize("failure", [False, True])
def test_frozen_asr_decodes_each_video_once_and_preserves_window_failures(
    tmp_path, monkeypatch, failure
):
    import wave

    from experiments.benchmarks.extraction.audio.adapters import Transcript
    from experiments.benchmarks.extraction.video import frozen_asr_worker as worker

    decoded, requests = [], []

    def decode(item, destination, protocol):
        decoded.append((item["id"], destination))
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"\0\0" * (65 * 16000))
        return ["ffmpeg", str(item["id"])]

    class Runtime:
        def load(self):
            pass

        def close(self):
            pass

        def parameters(self):
            return {}

        def transcribe(self, path):
            requests.append(path.stem)
            if failure and path.stem == "1":
                raise RuntimeError("injected window failure")
            return Transcript("alpha", ({"text": "alpha", "start": 0, "end": 1},))

    monkeypatch.setattr(worker, "build_runtime", lambda *_args: Runtime())
    monkeypatch.setattr(worker, "_decode", decode)
    monkeypatch.setattr(worker, "seed_deterministically", lambda *_args: None)
    items = [
        {
            "id": name,
            "duration_seconds": 65.0,
            "reference_transcript": "alpha alpha alpha",
            "reference_segments": [
                {"text": "alpha", "start": start, "end": start + 1}
                for start in (0, 28, 56)
            ],
        }
        for name in ("one", "two")
    ]
    output = worker.execute(
        {
            "protocol": VIDEO_PROTOCOL.meta.worker_payload(),
            "audio_protocol": AUDIO_PROTOCOL.meta.worker_payload(),
            "profile": "smoke",
            "device": "cpu",
            "candidate": "mock",
            "model_lock": {},
            "items": items,
        }
    )
    assert sum(path.name == "full.wav" for _, path in decoded) == 2
    assert requests == ["warmup", "0", "1", "2", "0", "1", "2"]
    assert all(len(row["windows"]) == 3 for row in output["videos"])
    assert output["metrics"]["word_error_rate"] == (None if failure else 0)
    assert output["metrics"]["attempt_failure_rate"] == pytest.approx(
        1 / 3 if failure else 0
    )
    assert output["metrics"]["peak_vram_mb"] is None
    assert all(row["success"] != failure for row in output["videos"])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("device", "cuda", "device mismatch"),
        ("model_revision", "another", "model revision mismatch"),
        ("duration_seconds", 2.0, "duration differs"),
        ("scheduled_window_count", 2, "scheduled window count"),
        ("execution_identity", {}, "execution/software identity mismatch"),
    ],
)
def test_frozen_asr_rejects_execution_identity_and_inventory_drift(
    tmp_path, field, value, match
):
    payload = _frozen_payload("one")
    if field in {"duration_seconds", "scheduled_window_count"}:
        payload["videos"][0][field] = value
    else:
        payload[field] = value
    path = tmp_path / "frozen.json"
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match=match):
        load_frozen_asr_artifact(
            path,
            manifest_checksum="manifest",
            protocol_checksum=VIDEO_PROTOCOL.meta.checksum,
            audio_protocol_checksum=AUDIO_PROTOCOL.meta.checksum,
            timestamp_tolerance_seconds=0.1,
            sample_ids=["one"],
            expected_device="cpu",
            expected_identity={"model_revision": "revision"},
            expected_durations={"one": 1.0},
        )


def test_overlap_stitching_retains_legitimate_reappearance_and_native_timestamp_order():
    result = _stitch_segments(
        [{"text": "again", "start": 1, "end": 2}],
        [{"text": "again", "start": 28, "end": 29}],
        64,
        overlap_start=28,
        overlap_end=30,
    )
    assert [row["text"] for row in result] == ["again", "again"]
    result = _stitch_segments(
        [{"text": "left", "start": 29, "end": 30}],
        [{"text": "right", "start": 28, "end": 29}],
        64,
        overlap_start=28,
        overlap_end=30,
    )
    assert [row["start"] for row in result] == [28, 29]
