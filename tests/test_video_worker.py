"""Visual lifecycle, first-attempt ownership, and strict frame output regressions."""

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.extraction import ExtractedDocument, ExtractionProfile, SourceKind
from experiments.benchmarks.common.process import WorkerExecutionError
from experiments.benchmarks.extraction.audio.protocol import (
    load_protocol as audio_protocol,
)
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol as document_protocol,
)
from experiments.benchmarks.extraction.video import visual_worker as worker
from experiments.benchmarks.extraction.video.protocol import (
    load_protocol,
    protocol_from_mapping,
)
from experiments.benchmarks.extraction.video.runner import _validate_visual_output


@pytest.fixture
def lifecycle(monkeypatch):
    root = deepcopy(load_protocol().meta.resolved)
    root["profiles"]["smoke"]["repetitions"] = 3
    protocol = protocol_from_mapping(root)
    document = document_protocol()
    state = SimpleNamespace(failures=set(), events=[], fail=None, device="cpu")
    item = {
        "id": "one",
        "source_path": "video.mp4",
        "duration_seconds": 10.0,
        "reference_visual_text": ["alpha"],
        "visual_occurrences": [{"id": "a", "text": "alpha", "start": 0, "end": 10}],
    }

    class Monitor:
        vram_measurement_method = "nvml-device-total"

        def __init__(self, **_kwargs):
            self.gpu_identity = {"uuid": "test"}

        def __enter__(self):
            state.events.append("monitor-start")
            return self

        def __exit__(self, *_args):
            state.events.append("monitor-stop")

        def metrics(self):
            return {
                "peak_process_tree_ram_mb": 12.0,
                "peak_vram_mb": 500.0 if state.device == "cuda" else None,
            }

        def samples(self):
            return [{"ram_mb": 12}]

    def process(_extractor, _candidate, _item, directory, *_args):
        name = directory.name
        state.events.append(name)
        if name in state.failures:
            raise worker.VisualAttemptFailure("bad frame", 12, ["ffmpeg"])
        return [{"text": "alpha", "timestamp": 0.0}], ["ffmpeg"], 12, "fingerprint"

    def initialize(_request):
        state.events.append("load")
        if state.fail == "load":
            raise RuntimeError("load failed")

    def placement(*_args, **_kwargs):
        state.events.append("placement")
        if state.fail == "placement_after" and state.events.count("placement") == 2:
            raise RuntimeError("inspection failed")
        return {
            "status": "offload_detected" if state.fail == "offload" else "qualified"
        }

    monkeypatch.setattr(worker, "ResourceMonitor", Monitor)
    monkeypatch.setattr(
        worker, "load_frozen_asr_artifact", lambda *_args, **_kwargs: ({}, "checksum")
    )
    monkeypatch.setattr(
        worker,
        "_extract_frames",
        lambda *_args, **_kwargs: ([(Path("frame.png"), 0.0)], ["ffmpeg"]),
    )
    monkeypatch.setattr(worker, "_image_request", lambda *_args: None)
    monkeypatch.setattr(worker, "_process_video", process)
    monkeypatch.setattr(worker, "synchronize", lambda *_args: None)
    monkeypatch.setattr(worker, "seed_deterministically", lambda *_args: None)
    monkeypatch.setattr(worker, "inspect_model_placement", placement)
    monkeypatch.setattr(worker, "pipeline_parameters", lambda *_args: {})
    monkeypatch.setattr(worker, "package_versions", lambda *_args: {})
    monkeypatch.setattr(
        worker,
        "build_experiment_registry",
        lambda: SimpleNamespace(
            create=lambda *_args: SimpleNamespace(initialize_image_pipeline=initialize)
        ),
    )

    def run(**overrides):
        return worker.execute(
            {
                "candidate": "video-fixed-5s",
                "device": state.device,
                "profile": "smoke",
                "protocol": protocol.meta.worker_payload(),
                "document_protocol": document.meta.worker_payload(),
                "audio_protocol": audio_protocol().meta.worker_payload(),
                "items": [item],
                "image_candidate": "docling-standard",
                "image_engine": "docling-standard",
                "image_revision": "test",
                "image_options": document.parser_options("docling-standard"),
                "frozen_asr_path": "frozen.json",
                "manifest_checksum": "manifest",
                "frozen_asr_identity": {},
                "frozen_asr_checksum": "checksum",
                "canonical_audio_checksums": {"one": "checksum"},
                "ffmpeg_version": "test",
                **overrides,
            }
        )

    state.run = run
    return state


def test_failed_first_attempt_stays_null_and_all_repetitions_run(lifecycle):
    lifecycle.failures.add("0-1")
    result = lifecycle.run()
    row = result["samples"][0]
    assert row["visual_content_f1"] is None
    assert row["selected_frame_count"] == 12
    assert row["attempt_failure_rate"] == pytest.approx(1 / 3)
    assert row["repeatability_success_rate"] == pytest.approx(1 / 3)
    assert [timing["success"] for timing in result["timings"]] == [False, True, True]
    assert result["operational"]["mean_selected_frames_per_video"] == 12
    assert result["operational"]["peak_visual_vram_mb"] is None


def test_later_failure_preserves_first_quality(lifecycle):
    lifecycle.failures.add("0-2")
    result = lifecycle.run()
    assert result["samples"][0]["visual_content_f1"] == 1
    assert result["samples"][0]["failed_attempt_count"] == 1
    assert result["execution_error"] is None


def test_lifecycle_observations_do_not_receive_statistical_suffixes(lifecycle):
    result = lifecycle.run()
    for name in (
        "cold_visual_pipeline_load_seconds",
        "peak_visual_process_tree_ram_mb",
        "peak_visual_vram_mb",
    ):
        assert name + ".sample_count" not in result["metrics"]
    assert result["metrics"]["visual_real_time_factor.sample_count"] == 1


def test_synchronization_failure_keeps_completed_frame_selection(
    lifecycle, monkeypatch
):
    def synchronize(*_args):
        if lifecycle.events[-1] == "0-1":
            raise RuntimeError("synchronization failed")

    monkeypatch.setattr(worker, "synchronize", synchronize)
    result = lifecycle.run()
    assert result["timings"][0]["selected_frame_count"] == 12
    assert result["timings"][0]["success"] is False
    assert any(
        command.get("repetition") == 1 and command.get("command") == ["ffmpeg"]
        for command in result["ffmpeg_commands"]
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda result: result["timings"][0].update(success=False, error="failure"),
            "success fingerprint",
        ),
        (lambda result: result["timings"].pop(), "inventory"),
        (
            lambda result: result["timings"][0].update(latency_seconds=float("nan")),
            "latency",
        ),
        (lambda result: result["timings"][0].update(duration_seconds=9), "manifest"),
        (
            lambda result: result["outputs"]["attempts"]["one"][1].update(
                fingerprint="wrong"
            ),
            "disagree",
        ),
        (lambda result: result["metrics"].update(visual_content_f1=0.5), "aggregate"),
        (
            lambda result: result["samples"][0].update(attempt_failure_rate=1),
            "reliability",
        ),
        (
            lambda result: result["operational"].update(peak_visual_vram_mb=0),
            "CPU visual VRAM",
        ),
    ],
)
def test_visual_transport_gate_rejects_contradictory_results(
    lifecycle, mutation, match
):
    result = lifecycle.run()
    items = [{"id": "one", "duration_seconds": 10.0}]
    _validate_visual_output(result, items, 3, "cpu")
    mutation(result)
    with pytest.raises(ValueError, match=match):
        _validate_visual_output(result, items, 3, "cpu")


def test_load_failure_is_unexecuted_not_measured_failure(lifecycle):
    lifecycle.fail = "load"
    result = lifecycle.run()
    assert result["execution_error"]["phase"] == "load"
    assert all(
        not row["executed"] and row["success"] is None for row in result["timings"]
    )
    assert result["metrics"]["attempt_failure_rate"] is None
    assert result["metric_statuses"]["attempt_failure_rate"]["status"] == "incomplete"


def test_placement_after_monitor_preserves_completed_quality(lifecycle):
    lifecycle.device, lifecycle.fail = "cuda", "placement_after"
    result = lifecycle.run()
    assert result["execution_error"]["phase"] == "placement_after"
    assert result["samples"][0]["visual_content_f1"] == 1
    assert lifecycle.events[-2:] == ["monitor-stop", "placement"]


def test_preflight_stops_before_inference_on_offload(lifecycle):
    lifecycle.device, lifecycle.fail = "cuda", "offload"
    result = lifecycle.run(mode="preflight")
    assert result["placement"]["status"] == "offload_detected"
    assert all(not row["executed"] for row in result["timings"])


def test_preflight_keeps_original_request_failure(lifecycle):
    lifecycle.device = "cuda"
    lifecycle.failures.add("0-1")
    with pytest.raises(WorkerExecutionError, match="bad frame") as caught:
        lifecycle.run(mode="preflight")
    assert caught.value.resource_evidence["timings"][0]["selected_frame_count"] == 12


def test_frame_conversion_failure_preserves_completed_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(
        worker,
        "_extract_frames",
        lambda *_args: ([(tmp_path / "frame.png", 0)], ["ffmpeg"]),
    )
    monkeypatch.setattr(
        worker, "_image_request", lambda *_args: SimpleNamespace(checksum="checksum")
    )
    extractor = SimpleNamespace(
        extract=lambda *_args: SimpleNamespace(to_dict=lambda: {"text": "bad"})
    )
    evidence = {}
    with pytest.raises(worker.VisualAttemptFailure) as caught:
        worker._process_video(
            extractor,
            None,
            {"source_path": "test", "duration_seconds": 10},
            tmp_path,
            "engine",
            "revision",
            "cpu",
            {},
            evidence,
        )
    assert caught.value.selected_frame_count == 1
    assert caught.value.command == ["ffmpeg"]
    assert evidence["selected_frame_timestamps"] == [0]
    assert evidence["frames"] == [
        {
            "frame_index": 0,
            "timestamp": 0,
            "input_sha256": "checksum",
            "document": {"text": "bad"},
        }
    ]


def test_frame_times_and_structured_outputs_survive_temporary_cleanup(
    monkeypatch, tmp_path
):
    evidence = {}
    with tempfile.TemporaryDirectory(dir=tmp_path) as raw:
        directory = Path(raw)
        monkeypatch.setattr(
            worker,
            "_extract_frames",
            lambda *_args: (
                [(directory / "zero.png", 0.0), (directory / "five.png", 5.0)],
                ["ffmpeg"],
            ),
        )
        monkeypatch.setattr(
            worker,
            "_image_request",
            lambda *_args: SimpleNamespace(checksum="checksum"),
        )
        document = ExtractedDocument(
            "id",
            "frame.png",
            SourceKind.IMAGE,
            "checksum",
            None,
            "alpha",
            (),
            ExtractionProfile("test", "test", "1"),
        )
        extractor = SimpleNamespace(extract=lambda *_args: document)
        predictions, _, count, fingerprint = worker._process_video(
            extractor,
            None,
            {"source_path": "video.mp4", "duration_seconds": 10},
            directory,
            "engine",
            "revision",
            "cpu",
            {},
            evidence,
        )
    assert not directory.exists()
    saved = json.loads(json.dumps(evidence))
    assert saved["selected_frame_timestamps"] == [0, 5]
    assert [frame["document"]["text"] for frame in saved["frames"]] == [
        "alpha",
        "alpha",
    ]
    assert [prediction["timestamp"] for prediction in predictions] == [0, 5]
    assert count == 2 and fingerprint
