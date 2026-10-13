"""ASR lifecycle, preprocessing seals, and frozen-contract regressions."""

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.benchmarks.common.preflight import _probe_candidate
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
)
from experiments.benchmarks.extraction.audio import validation, worker
from experiments.benchmarks.extraction.audio.adapters import Transcript
from experiments.benchmarks.extraction.audio.protocol import (
    load_protocol,
    protocol_from_mapping,
)
from experiments.benchmarks.extraction.audio.runner import (
    _validate_output,
    _verify_input_envelope,
)


@pytest.fixture
def lifecycle(monkeypatch):
    protocol = load_protocol()
    state = {
        "calls": 0,
        "closed": False,
        "load_error": False,
        "close_error": False,
        "fail_calls": set(),
        "telemetry_error": False,
    }

    class Runtime:
        def load(self):
            if state.get("load_exception") is not None:
                raise state["load_exception"]
            if state["load_error"]:
                raise RuntimeError("load failed")

        def transcribe(self, _path):
            state["calls"] += 1
            if state["calls"] in state["fail_calls"]:
                raise RuntimeError("request failed")
            return Transcript("hello", ({"text": "hello", "start": 0.0, "end": 1.0},))

        def parameters(self):
            if state.get("parameters_error"):
                raise RuntimeError("parameters failed")
            return {"device": "cpu"}

        def placement_models(self):
            return (self,)

        def close(self):
            state["closed"] = True
            if state["close_error"]:
                raise RuntimeError("close failed")

    class Monitor:
        gpu_identity = None
        vram_measurement_method = None

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            if state["telemetry_error"]:
                raise RuntimeError("telemetry failed")

        def metrics(self):
            return {"peak_process_tree_ram_mb": 12.0, "peak_vram_mb": state.get("vram")}

        def samples(self):
            return [{"ram_mb": 12.0}]

    monkeypatch.setattr(worker, "build_runtime", lambda *_args: Runtime())
    monkeypatch.setattr(worker, "ResourceMonitor", Monitor)
    monkeypatch.setattr(worker, "seed_deterministically", lambda *_args: None)
    monkeypatch.setattr(worker, "synchronize", lambda *_args: None)
    monkeypatch.setattr(
        worker,
        "inspect_model_placement",
        lambda *_args, **_kwargs: {"status": state.get("placement", "qualified")},
    )
    speech = [
        {
            "id": "speech",
            "canonical_path": "speech.wav",
            "reference": "hello",
            "reference_segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
            "duration_seconds": 1.0,
            "document_family": "source",
        }
    ]
    payload = {
        "candidate": "whisper-small-en-control",
        "model_lock": {},
        "profile": "smoke",
        "device": "cpu",
        "speech": speech,
        "reliability": [],
        "protocol": protocol.meta.worker_payload(),
    }
    return state, payload


def _preflight(lifecycle):
    state, payload = lifecycle
    state["vram"] = 12.0
    payload.update(device="cuda", mode="preflight")
    return state, payload


def test_preflight_shares_lifecycle_without_warmup_or_quality(lifecycle):
    state, payload = _preflight(lifecycle)
    result = worker.execute(payload)
    assert state["calls"] == 1
    assert state["closed"]
    assert result["placement"]["status"] == "qualified"
    assert "metrics" not in result


def test_preflight_stops_before_inference_on_offload(lifecycle):
    state, payload = _preflight(lifecycle)
    state["placement"] = "offload_detected"
    result = worker.execute(payload)
    assert result["placement"]["status"] == "offload_detected"
    assert state["calls"] == 0
    assert state["closed"]


def test_preflight_preserves_inference_error_and_cleanup(lifecycle):
    state, payload = _preflight(lifecycle)
    state["fail_calls"] = {1}
    with pytest.raises(WorkerExecutionError, match="request failed"):
        worker.execute(payload)
    assert state["closed"]


@pytest.mark.parametrize("oom", [False, True])
def test_preflight_cleanup_cannot_mask_a_failed_request(lifecycle, monkeypatch, oom):
    state, payload = _preflight(lifecycle)
    state.update(fail_calls={1}, close_error=True)
    if oom:

        def fail(*_args, **_kwargs):
            raise RuntimeError("CUDA out of memory")

        state["fail_calls"] = set()
        monkeypatch.setattr(worker, "validate_prediction", fail)
    result = _probe_candidate("asr", lambda _candidate: worker.execute(payload))
    assert result["outcome"] == ("gpu_oom" if oom else "execution_error")
    assert "close failed" in result["execution_error"]["error"]
    assert (
        "out of memory" in result["timings"][0]["error"]
        if oom
        else "request failed" in result["timings"][0]["error"]
    )


@pytest.mark.parametrize("operation", ["score_speech", "aggregate"])
def test_offline_failure_retains_completed_attempts_and_partial_scores(
    lifecycle, monkeypatch, operation
):
    state, payload = lifecycle

    def fail(*_args, **_kwargs):
        raise RuntimeError("offline evaluation failed")

    monkeypatch.setattr(worker, operation, fail)
    with pytest.raises(
        WorkerExecutionError, match="offline evaluation failed"
    ) as caught:
        worker.execute(payload)
    evidence = caught.value.resource_evidence
    assert evidence["timings"][0]["executed"] is True
    assert evidence["timings"][0]["success"] is True
    assert len(evidence["samples"]) == (1 if operation == "aggregate" else 0)
    assert evidence["execution_error"]["phase"] == "offline_evaluation"
    assert state["closed"]


def test_offline_failure_retains_cold_load_and_preceding_lifecycle_error(
    lifecycle, monkeypatch
):
    state, payload = lifecycle
    state["close_error"] = True

    def fail(*_args, **_kwargs):
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(worker, "aggregate", fail)
    with pytest.raises(WorkerExecutionError) as caught:
        worker.execute(payload)
    evidence = caught.value.resource_evidence
    assert evidence["cold_model_load_seconds"] >= 0
    assert evidence["execution_error"]["preceding_execution_error"]["phase"] == "close"
    assert (
        "close failed"
        in evidence["execution_error"]["preceding_execution_error"]["error"]
    )


def test_preflight_missing_telemetry_is_not_hardware_exclusion(lifecycle):
    state, payload = _preflight(lifecycle)
    state["vram"] = None
    with pytest.raises(WorkerMeasurementError):
        worker.execute(payload)
    assert state["closed"]


@pytest.mark.parametrize(
    "kind,outcome,status",
    [
        ("offload", "offload_detected", "excluded"),
        ("oom", "gpu_oom", "excluded"),
        ("dtype", "execution_error", "blocked"),
    ],
)
def test_preflight_distinguishes_load_time_hardware_and_execution_failures(
    lifecycle, kind, outcome, status
):
    state, payload = _preflight(lifecycle)
    error = RuntimeError(
        "CUDA out of memory" if kind == "oom" else "load contract failed"
    )
    if kind == "offload":
        error.placement_report = {"status": "offload_detected"}
    state["load_exception"] = error
    result = _probe_candidate("asr", lambda _candidate: worker.execute(payload))
    assert result["outcome"] == outcome
    assert result["status"] == status
    assert state["closed"]
    assert state["calls"] == 0


@pytest.mark.parametrize(
    "phase", ["load", "warmup", "close", "telemetry", "parameters"]
)
def test_lifecycle_failures_retain_attempt_inventory(lifecycle, phase):
    state, payload = lifecycle
    if phase == "warmup":
        state["fail_calls"] = {1}
    else:
        state[phase + "_error"] = True
    result = worker.execute(payload)
    assert result["execution_error"]["phase"] == phase
    assert state["closed"]
    _validate_output(result, payload["speech"], [], 1, "cpu")
    executed = phase in {"close", "telemetry", "parameters"}
    assert result["timings"][0]["executed"] is executed
    assert result["metrics"]["word_error_rate"] == (0 if executed else None)
    assert result["metrics"]["attempt_failure_rate"] == (0 if executed else None)
    assert result["metrics"]["peak_vram_mb"] is None
    if phase != "parameters":
        assert result["parameters"]["device"] == "cpu"


def test_cleanup_does_not_mask_original_error(lifecycle):
    state, payload = lifecycle
    state.update(load_error=True, close_error=True)
    result = worker.execute(payload)
    assert result["execution_error"]["error"] == "RuntimeError: load failed"
    assert result["execution_error"]["cleanup_error"] == "RuntimeError: close failed"


def test_failed_first_attempt_never_uses_later_output(lifecycle):
    state, payload = lifecycle
    root = deepcopy(load_protocol().meta.resolved)
    root["profiles"]["smoke"]["repetitions"] = 3
    protocol = protocol_from_mapping(root, source_path=Path("test-protocol.yaml"))
    payload["protocol"] = protocol.meta.worker_payload()
    state["fail_calls"] = {2}
    result = worker.execute(payload)
    assert state["calls"] == 4
    assert [row["success"] for row in result["timings"]] == [False, True, True]
    assert result["metrics"]["word_error_rate"] is None
    assert result["metrics"]["attempt_failure_rate"] == pytest.approx(1 / 3)
    assert result["metrics"]["transcript_repeatability_success_rate"] == pytest.approx(
        1 / 3
    )
    assert result["execution_error"] is None


@pytest.mark.parametrize("mutation", ["hash", "duration", "missing"])
def test_canonical_execution_must_match_sealed_validation(mutation):
    sample = {"id": "speech", "canonical_sha256": "a" * 64, "duration_seconds": 1.0}
    expected = {"canonical_sha256": "a" * 64, "actual_duration_seconds": 1.0}
    metadata = {"samples": {"speech": expected}}
    validation.verify_canonical_inputs([sample], metadata)
    if mutation == "hash":
        expected["canonical_sha256"] = "b" * 64
    elif mutation == "duration":
        expected["actual_duration_seconds"] = 2.0
    else:
        metadata["samples"].clear()
    with pytest.raises(ValueError, match="sealed validation"):
        validation.verify_canonical_inputs([sample], metadata)


def test_qualification_reuse_rejects_longer_speech_or_controls():
    _verify_input_envelope([{"duration_seconds": 2.0}], 2.0)
    with pytest.raises(ValueError, match="duration envelope"):
        _verify_input_envelope([{"duration_seconds": 2.1}], 2.0)


def test_control_inventory_rejects_unknown_split():
    with pytest.raises(ValueError, match="Unknown ASR reliability split"):
        validation.validate_reliability_split_isolation(
            [
                {
                    "id": "control",
                    "kind": "audio_reliability",
                    "split": "typo",
                    "asset_sha256": "a" * 64,
                }
            ]
        )


def test_canonicalization_rejects_zero_frames_even_within_manifest_tolerance(
    tmp_path, monkeypatch
):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(validation, "sha256_file", lambda _path: "checksum")
    monkeypatch.setattr(
        validation, "decode_canonical_audio", lambda *_args, **_kwargs: ["ffmpeg"]
    )
    monkeypatch.setattr(
        validation, "canonical_wav_duration", lambda *_args, **_kwargs: 0.0
    )
    with pytest.raises(ValueError, match="positive duration"):
        validation.canonicalize(
            [
                {
                    "id": "empty",
                    "source_path": str(source),
                    "asset_sha256": "checksum",
                    "duration_seconds": 0.05,
                }
            ],
            tmp_path / "canonical",
            load_protocol(),
        )


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_reference_boundaries_are_strict_numbers(value):
    item = {
        "id": "speech",
        "reference": "hello",
        "reference_segments": [{"text": "hello", "start": value, "end": 1.0}],
    }
    with pytest.raises(ValueError):
        validation.validate_reference_segments(item, 1.0)


@pytest.mark.parametrize(
    "alias,field,value",
    [
        ("whisper-small-en-control", "do_sample", True),
        ("canary-180m", "beam_size", 2),
        ("canary-180m", "timestamps", False),
        ("canary-180m", "punctuation_and_capitalization", False),
        ("moss-transcribe-diarize", "do_sample", True),
    ],
)
def test_protocol_rejects_contradictory_decoder_settings(alias, field, value):
    root = deepcopy(load_protocol().meta.resolved)
    root["decoding"][alias][field] = value
    with pytest.raises(ValueError):
        protocol_from_mapping(root, source_path=Path("test-protocol.yaml"))
