"""Video protocol ownership, qualification envelopes, and preparation tracking."""

import json
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import sha256_file
from experiments.benchmarks.common import preflight as common_preflight
from experiments.benchmarks.common import runner as common_runner
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
)
from experiments.benchmarks.common.tracking import NoTracking
from experiments.benchmarks.extraction.video import preflight, runner
from experiments.benchmarks.extraction.video.protocol import (
    load_protocol,
    protocol_from_mapping,
)


def _validation(items):
    return {
        "metadata": {
            "samples": {
                item["id"]: {
                    "width": item["width"],
                    "height": item["height"],
                    "canonical_duration_seconds": item["duration_seconds"],
                    "actual_duration_seconds": item["duration_seconds"],
                }
                for item in items
            }
        }
    }


def test_stress_selection_covers_nondominated_shapes_and_durations():
    items = [
        {
            "id": "small",
            "kind": "video",
            "width": 640,
            "height": 480,
            "duration_seconds": 10,
        },
        {
            "id": "long",
            "kind": "video",
            "width": 640,
            "height": 480,
            "duration_seconds": 60,
        },
        {
            "id": "wide",
            "kind": "video",
            "width": 1920,
            "height": 1080,
            "duration_seconds": 30,
        },
    ]
    validation = _validation(items)
    selected = preflight.qualification_inputs(
        SimpleNamespace(samples=items), validation
    )
    assert [item["id"] for item in selected] == ["long", "wide"]
    qualification = {
        "context": {
            "input_envelope": {
                "tested_video_envelopes": [
                    preflight.video_envelope(item, validation) for item in selected
                ]
            }
        }
    }
    preflight.validate_input_envelope(items, validation, qualification)
    bigger = {**items[-1], "id": "larger", "duration_seconds": 60}
    with pytest.raises(ValueError, match="exceeds the qualified"):
        preflight.validate_input_envelope(
            [bigger], _validation([bigger]), qualification
        )


def test_visual_stream_duration_is_part_of_the_tested_envelope():
    item = {"id": "video", "width": 640, "height": 480, "duration_seconds": 10}
    validation = _validation([item])
    validation["metadata"]["samples"]["video"]["actual_duration_seconds"] = 10.09
    assert preflight.video_envelope(item, validation)["duration_seconds"] == 10.09


@pytest.mark.parametrize("value", [0.304, 0.301, 0.4001])
def test_scene_threshold_cannot_silently_round_to_another_candidate(value):
    root = deepcopy(load_protocol().meta.resolved)
    root["keyframes"]["scene_thresholds"][0] = value
    root["keyframes"]["scene_thresholds"].sort()
    with pytest.raises(ValueError, match="two-decimal"):
        protocol_from_mapping(root)


def test_validation_decision_is_bound_to_all_protocols_and_qualification(
    monkeypatch, tmp_path
):
    root = deepcopy(load_protocol().meta.resolved)
    root["keyframes"].update(
        selected_scene_threshold=0.4, selected_scene_source_run_id="scene-run"
    )
    protocol = protocol_from_mapping(root)
    captured = {}

    def load(_path, **settings):
        captured.update(settings)
        return SimpleNamespace(selected_candidates=("video-fixed-5s",))

    monkeypatch.setattr(runner, "load_engineer_decision", load)
    arguments = SimpleNamespace(
        profile="validation", phase="all", shortlist=tmp_path / "decision.json"
    )
    expected = {
        "video": protocol.meta.checksum,
        "audio": "audio",
        "document": "document",
    }
    runner._candidates(
        arguments,
        protocol,
        expected_protocols=expected,
        expected_qualification="qualification",
    )
    assert captured["expected_protocols"] == expected
    assert captured["expected_qualification"] == "qualification"


def test_input_preparation_is_a_sibling_not_a_comparison_candidate(
    monkeypatch, tmp_path
):
    tags = []

    class Tracking:
        @contextmanager
        def run(self, name, *, nested=False):
            yield name

        def tags(self, values):
            tags.append(values)

        def parameters(self, values):
            pass

        def metrics(self, values):
            pass

        def artifact(self, path, artifact_path=None):
            pass

        def mark_failed(self):
            pass

    monkeypatch.setattr(common_runner, "tracker", lambda **_kwargs: Tracking())
    plan = BenchmarkPlan(
        "extraction",
        "video-smoke",
        "smoke",
        "fixture",
        ("frozen", "visual"),
        seed=42,
        repetitions=1,
        warmups=1,
        bootstrap_resamples=0,
    )

    def evaluate(candidate):
        field = "word_error_rate" if candidate == "frozen" else "visual_content_f1"
        return [SampleResult("one", {field: 1.0}, 0.1)], {}, {}

    result = common_runner.run_benchmark(
        plan,
        evaluate,
        dataset_checksum="checksum",
        directions={"word_error_rate": "min", "visual_content_f1": "max"},
        primary_metric="visual_content_f1",
        candidate_run_types={"frozen": "input_preparation"},
        candidate_metric_contracts={
            "frozen": ("word_error_rate",),
            "visual": ("visual_content_f1",),
        },
        monitor_resources=False,
        paired_comparisons=False,
        shuffle_candidates=False,
        artifact_root=tmp_path,
    )
    assert result.complete, result.completion_problems
    assert [tag["run_type"] for tag in tags if "candidate" in tag] == [
        "input_preparation",
        "quality-candidate",
    ]
    summary = json.loads(
        (result.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["completion"]["input_preparation_ready"]
    assert summary["completion"]["comparison_candidates"] == 1


@pytest.mark.parametrize("error_type", [WorkerExecutionError, WorkerMeasurementError])
def test_preparation_failure_does_not_overwrite_mlflow_candidate(
    monkeypatch, error_type
):
    def failed(*_args, **_kwargs):
        raise error_type(
            "original preparation failure",
            {
                "parameters": {"candidate": "whisper-small-en-control"},
                "metrics": {"word_error_rate": 0.25, "invalid_value": float("nan")},
                "intervals": {"word_error_rate": {"lower": None, "upper": None}},
                "metric_statuses": {
                    "word_error_rate": {"value": 0.25, "status": "scored"}
                },
            },
        )

    monkeypatch.setattr(runner, "create_frozen_asr_artifact", failed)
    with pytest.raises(error_type, match="original preparation failure") as caught:
        runner._preserve_worker_failure(failed)
    assert "candidate" not in caught.value.parameters
    assert caught.value.metrics["word_error_rate"] == 0.25
    assert "invalid_value" not in caught.value.metrics
    assert (
        caught.value.artifacts["metric_statuses"]["word_error_rate"][
            "comparison_status"
        ]
        == "incomplete"
    )
    assert (
        caught.value.artifacts["worker_error"]["parameters"]["candidate"]
        == "whisper-small-en-control"
    )


@pytest.mark.parametrize("supervisor_peak", [999.0, None])
def test_raw_preparation_failure_keeps_worker_lifecycle_scope(
    monkeypatch, supervisor_peak
):
    def failed(*_args, **_kwargs):
        raise WorkerExecutionError(
            "offline scoring failed",
            {
                "peak_process_tree_ram_mb": supervisor_peak,
                "peak_vram_mb": supervisor_peak,
                "supervisor_resource_evidence": {
                    "peak_process_tree_ram_mb": supervisor_peak
                },
                "worker_result": {
                    "worker_error": True,
                    "resource_evidence": {
                        "cold_model_load_seconds": 0.2,
                        "peak_process_tree_ram_mb": 7.0,
                        "peak_vram_mb": None,
                    },
                },
            },
        )

    monkeypatch.setattr(runner, "create_frozen_asr_artifact", failed)
    with pytest.raises(WorkerExecutionError) as caught:
        runner._preserve_worker_failure(failed)
    assert caught.value.operational["cold_model_load_seconds"] == 0.2
    assert caught.value.operational["peak_process_tree_ram_mb"] == 7.0
    assert caught.value.operational["peak_vram_mb"] is None
    assert (
        caught.value.artifacts["supervisor_resource_evidence"][
            "peak_process_tree_ram_mb"
        ]
        == supervisor_peak
    )


@pytest.mark.parametrize("worker_result", [None, {}])
def test_missing_worker_output_does_not_promote_supervisor_peaks(
    monkeypatch, worker_result
):
    def failed(*_args, **_kwargs):
        raise WorkerExecutionError(
            "worker returned no JSON",
            {
                "worker_result": worker_result,
                "peak_process_tree_ram_mb": 999.0,
                "peak_vram_mb": 888.0,
                "supervisor_resource_evidence": {"peak_process_tree_ram_mb": 999.0},
            },
        )

    monkeypatch.setattr(runner, "create_frozen_asr_artifact", failed)
    with pytest.raises(WorkerExecutionError) as caught:
        runner._preserve_worker_failure(failed)
    assert caught.value.operational["peak_process_tree_ram_mb"] is None
    assert caught.value.operational["peak_vram_mb"] is None
    assert (
        caught.value.artifacts["supervisor_resource_evidence"][
            "peak_process_tree_ram_mb"
        ]
        == 999.0
    )


def test_unreadable_worker_output_keeps_supervisor_scope(monkeypatch):
    def failed(*_args, **_kwargs):
        raise WorkerExecutionError(
            "unreadable JSON",
            {"peak_process_tree_ram_mb": 999.0, "peak_vram_mb": 888.0},
        )

    monkeypatch.setattr(runner, "create_frozen_asr_artifact", failed)
    with pytest.raises(WorkerExecutionError) as caught:
        runner._preserve_worker_failure(failed)
    assert caught.value.operational["peak_process_tree_ram_mb"] is None
    assert caught.value.operational["peak_vram_mb"] is None
    assert (
        caught.value.artifacts["supervisor_resource_evidence"][
            "peak_process_tree_ram_mb"
        ]
        == 999.0
    )


def test_native_underscore_cuda_oom_is_a_definitive_hardware_exclusion():
    def probe(_candidate):
        raise WorkerExecutionError("CUDA out_of_memory", {})

    result = common_preflight._probe_candidate("visual|video-fixed-5s", probe)
    assert result["status"] == "excluded"
    assert result["outcome"] == "gpu_oom"


def test_ambiguous_host_memory_error_is_not_a_gpu_exclusion():
    def probe(_candidate):
        raise MemoryError("out of memory while decoding media")

    result = common_preflight._probe_candidate("visual|video-fixed-5s", probe)
    assert result["status"] == "blocked"
    assert result["outcome"] == "execution_error"


@pytest.mark.parametrize(
    "child",
    [
        {
            "resource_evidence": {
                "videos": [{"windows": [{"error": "CUDA out_of_memory"}]}]
            }
        },
        {
            "worker_error": "CUDA out of memory",
            "worker_error_type": "WorkerExecutionError",
        },
    ],
)
def test_telemetry_failure_does_not_hide_recorded_child_gpu_oom(child):
    evidence = {"worker_result": child, "vram_measurement_method": "unavailable"}

    def probe(_candidate):
        raise WorkerMeasurementError("NVML measurement failed", evidence)

    result = common_preflight._probe_candidate(
        "frozen-asr|whisper-small-en-control", probe
    )
    assert result["status"] == "excluded"
    assert result["outcome"] == "gpu_oom"
    assert result["worker_result"] == child
    assert result["vram_measurement_method"] == "unavailable"


def test_failed_candidate_logs_independently_valid_retained_intervals(
    monkeypatch, tmp_path
):
    tracking, recorded = NoTracking(), []
    monkeypatch.setattr(tracking, "metrics", lambda values: recorded.append(values))
    monkeypatch.setattr(common_runner, "tracker", lambda **_kwargs: tracking)

    def late_failure(_candidate):
        raise CandidateExecutionError(
            "late cleanup failure",
            metrics={"score": 0.5},
            intervals={"score": {"lower": 0.4, "upper": 0.6, "confidence": 0.95}},
        )

    result = common_runner.run_benchmark(
        BenchmarkPlan(
            "extraction",
            "video-development",
            "development",
            "fixture",
            ("visual",),
            seed=42,
            repetitions=3,
            warmups=1,
            bootstrap_resamples=10000,
        ),
        late_failure,
        dataset_checksum="checksum",
        directions={"score": "max"},
        primary_metric="score",
        monitor_resources=False,
        paired_comparisons=False,
        artifact_root=tmp_path,
    )
    assert not result.complete
    assert any(
        values.get("score.ci_lower") == 0.4 and values.get("score.ci_upper") == 0.6
        for values in recorded
    )


def test_frozen_failure_uploads_only_references_not_nested_raw_outputs(
    monkeypatch, tmp_path
):
    def failed(*_args, **_kwargs):
        raise WorkerExecutionError(
            "offline failure",
            {
                "worker_result": {
                    "resource_evidence": {
                        "videos": [
                            {
                                "transcript": "private prediction",
                                "segments": [
                                    {"text": "private prediction", "start": 0, "end": 1}
                                ],
                                "windows": [
                                    {
                                        "transcript": "private prediction",
                                        "raw_prediction": {"text": "private native"},
                                    }
                                ],
                            }
                        ]
                    }
                }
            },
        )

    monkeypatch.setattr(runner, "create_frozen_asr_artifact", failed)
    with pytest.raises(WorkerExecutionError) as caught:
        runner._preserve_worker_failure(failed, tmp_path / "frozen.json")
    reference = caught.value.parameters["prediction_evidence"]
    from pathlib import Path

    path = Path(reference["path"])
    assert reference["uploaded"] is False and reference["sha256"] == sha256_file(path)
    assert "private prediction" in path.read_text(encoding="utf-8")
    assert "private" not in json.dumps(caught.value.artifacts)
