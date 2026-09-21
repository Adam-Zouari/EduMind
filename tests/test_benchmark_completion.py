from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common import process as benchmark_process
from experiments.benchmarks.common.runner import run_benchmark
import experiments.benchmarks.common.runner as benchmark_runner
from experiments.benchmarks.extraction.document import runner as document_runner
from experiments.benchmarks.rag.generation.protocol import (
    load_protocol as load_generation_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_worker_launcher_uses_repository_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched = []

    def fake_run(command, **options):
        launched.append((command, options))
        Path(command[-1]).write_text('{"status":"success"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(benchmark_process.subprocess, "run", fake_run)
    script = ROOT / "experiments/benchmarks/extraction/audio/worker.py"
    result = benchmark_process.run_json_worker(
        script, {}, device="cpu", prefix="worker-test-", error_label="test",
        temporary_root=tmp_path,
    )
    assert result == {"status": "success"}
    command, options = launched[0]
    assert command[:3] == [
        sys.executable, "-m", "experiments.benchmarks.extraction.audio.worker"
    ]
    assert options["cwd"] == ROOT


def test_document_cold_worker_launches_as_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched = []

    def fake_run(command, **options):
        launched.append((command, options))
        return SimpleNamespace(stdout="EDUMIND_FIRST_ITEM_COMPLETE", stderr="")

    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setattr(document_runner.subprocess, "run", fake_run)
    protocol = SimpleNamespace(meta=SimpleNamespace(worker_payload=lambda: {}))
    assert document_runner._cold_latency("candidate", {"id": "sample"}, {}, {}, protocol) >= 0
    command, options = launched[0]
    assert command[:3] == [
        sys.executable, "-m", "experiments.benchmarks.extraction.document.cold_worker"
    ]
    assert options["cwd"] == ROOT


@pytest.mark.parametrize(
    "module",
    (
        "experiments.benchmarks.prepare",
        "experiments.benchmarks.review",
        "experiments.benchmarks.extraction.audio.run",
        "experiments.benchmarks.extraction.document.run",
        "experiments.benchmarks.extraction.video.run",
        "experiments.benchmarks.rag.chunking_embedding.run",
        "experiments.benchmarks.rag.retrieval_reranking.run",
        "experiments.benchmarks.rag.generation.run",
        "experiments.benchmarks.rag.final.run",
        "experiments.benchmarks.rag.final.confirm_extraction",
        "experiments.benchmarks.vectordb.run",
        "experiments.benchmarks.vectordb.retrieval_run",
    ),
)
def test_benchmark_entrypoint_runs_as_module(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize(
    "module",
    (
        "experiments.benchmarks.rag.chunking_embedding.run",
        "experiments.benchmarks.rag.generation.run",
        "experiments.benchmarks.rag.final.run",
    ),
)
def test_rag_entrypoint_import_does_not_parse_arguments(module: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, sys; sys.argv = ['test', '--unexpected']; "
            f"importlib.import_module({module!r})",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "module",
    (
        "experiments.benchmarks.rag.chunking_embedding.worker",
        "experiments.benchmarks.rag.retrieval_reranking.worker",
    ),
)
def test_rag_workers_import_as_modules(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module], cwd=ROOT, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "usage:" in completed.stderr.lower()
    assert "ModuleNotFoundError" not in completed.stderr


def _plan(*candidates: str) -> BenchmarkPlan:
    return BenchmarkPlan(
        "test-suite",
        "completion",
        "development",
        "fixed-test-data",
        candidates,
        seed=42,
        repetitions=1,
        bootstrap_resamples=50,
        warmups=0,
    )


def _run(tmp_path: Path, plan: BenchmarkPlan, evaluator):
    return run_benchmark(
        plan,
        evaluator,
        dataset_checksum="fixed-checksum",
        directions={"quality": "max", "operational.p95_latency_seconds": "min"},
        primary_metric="quality",
        no_mlflow=True,
        artifact_root=tmp_path,
    )


def test_temporary_environment_does_not_reuse_deleted_candidate_directory(
    tmp_path: Path,
) -> None:
    original = tempfile.tempdir
    tempfile.tempdir = None
    try:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        with benchmark_runner._temporary_environment(first):
            assert Path(tempfile.gettempdir()) == first.resolve()
        first.rmdir()
        with benchmark_runner._temporary_environment(second):
            assert Path(tempfile.gettempdir()) == second.resolve()
    finally:
        tempfile.tempdir = original


def test_resource_monitor_failure_does_not_mask_candidate_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Monitor:
        vram_measurement_method = "unavailable"

        def __init__(self, **_options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def metrics(self):
            raise RuntimeError("monitor failure")

        def samples(self):
            return []

    monkeypatch.setattr(benchmark_runner, "ResourceMonitor", Monitor)

    def evaluator(_candidate: str):
        raise CandidateExecutionError("original candidate failure")

    result = _run(tmp_path, _plan("failed"), evaluator)

    assert result.candidates[0].status == "failed"
    assert result.candidates[0].error == "original candidate failure"


def test_bad_score_is_complete_and_runner_makes_no_selection(tmp_path: Path) -> None:
    def evaluator(candidate: str):
        score = 0.0 if candidate == "poor" else 1.0
        return [SampleResult("same-sample", {"quality": score}, 0.01)], {
            "p95_latency_seconds": 0.01
        }

    result = _run(tmp_path, _plan("poor", "strong"), evaluator)

    assert result.complete is True
    summary = json.loads(
        (result.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["metric_contract"]["primary_metrics"] == ["quality"]
    assert summary["selection"]["made_by_runner"] is False
    assert "pareto_candidates" not in summary
    assert "gate_failures" not in summary
    assert "authoritative" not in summary


def test_missing_required_metric_fails_candidate_but_keeps_samples(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _plan("incomplete"),
        lambda _candidate: ([SampleResult("sample-1", {}, 0.01)], {"p95_latency_seconds": 0.01}),
    )

    assert result.complete is False
    assert result.candidates[0].status == "failed"
    assert "quality" in str(result.candidates[0].error)
    assert result.candidates[0].samples[0].sample_id == "sample-1"
    assert (result.artifact_directory / "samples" / "incomplete.parquet").is_file()


def test_different_sample_sets_make_comparison_incomplete(tmp_path: Path) -> None:
    def evaluator(candidate: str):
        sample_id = "sample-a" if candidate == "left" else "sample-b"
        return [SampleResult(sample_id, {"quality": 0.5}, 0.01)], {
            "p95_latency_seconds": 0.01
        }

    result = _run(tmp_path, _plan("left", "right"), evaluator)

    assert all(candidate.status == "success" for candidate in result.candidates)
    assert result.complete is False
    assert any("different sample set" in problem for problem in result.completion_problems)


def test_input_limit_failure_is_audited_and_fails_parent(
    tmp_path: Path,
) -> None:
    def evaluator(_candidate: str):
        raise CandidateExecutionError(
            "input limit",
            parameters={"maximum_length": 256},
            artifacts={
                "validation_report": {
                    "status": "failed",
                    "reason_code": "input_length_exceeded",
                }
            },
        )

    result = _run(tmp_path, _plan("oversized"), evaluator)
    assert result.complete is False
    assert result.candidates[0].status == "failed"
    summary = json.loads(
        (result.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["completion"]["failed_candidates"] == 1
    assert "incompatible_candidates" not in summary["completion"]
    assert not (result.artifact_directory / "compatibility_matrix.json").exists()
    assert (
        result.artifact_directory
        / "candidates"
        / "oversized"
        / "validation_report.json"
    ).is_file()


def test_failed_candidate_preserves_empty_audit_table(tmp_path: Path) -> None:
    def evaluator(_candidate: str):
        raise CandidateExecutionError(
            "all queries failed",
            artifacts={"query_metrics": []},
        )

    result = _run(tmp_path, _plan("failed"), evaluator)

    assert result.candidates[0].status == "failed"
    assert (
        result.artifact_directory
        / "candidates"
        / "failed"
        / "query_metrics.parquet"
    ).is_file()


def test_parent_artifact_validation_failure_marks_run_incomplete(
    tmp_path: Path,
) -> None:
    def fail_parent(*_args):
        raise ValueError("mirror mismatch")

    result = run_benchmark(
        _plan("candidate"),
        lambda _candidate: (
            [SampleResult("sample", {"quality": 1.0}, 0.01)],
            {"p95_latency_seconds": 0.01},
        ),
        dataset_checksum="fixed-checksum",
        directions={
            "quality": "max",
            "operational.p95_latency_seconds": "min",
        },
        primary_metric="quality",
        no_mlflow=True,
        artifact_root=tmp_path,
        parent_artifact_builder=fail_parent,
    )

    assert result.complete is False
    assert any(
        "parent artifact validation failed" in problem
        for problem in result.completion_problems
    )
    summary = json.loads(
        (result.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is False


def test_paired_comparisons_resample_document_groups(tmp_path: Path) -> None:
    plan = _plan("left", "right")

    def evaluator(candidate: str):
        values = (1.0, 1.0, 0.0) if candidate == "left" else (0.0, 0.0, 0.0)
        samples = [
            SampleResult(
                sample_id,
                {"quality": value},
                0.01,
                {"document_id": document_id},
            )
            for sample_id, document_id, value in zip(
                ("q1", "q2", "q3"), ("a", "a", "b"), values, strict=True
            )
        ]
        return samples, {"p95_latency_seconds": 0.01}

    result = run_benchmark(
        plan,
        evaluator,
        dataset_checksum="fixed-checksum",
        directions={"quality": "max", "operational.p95_latency_seconds": "min"},
        primary_metric="quality",
        paired_metrics=("quality",),
        paired_group_key="document_id",
        no_mlflow=True,
        artifact_root=tmp_path,
    )
    summary = json.loads(
        (result.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    comparison = summary["paired_comparisons"][0]["metrics"]["quality"]
    expected = 0.5 if summary["paired_comparisons"][0]["left"] == "left" else -0.5
    assert comparison["left_minus_right"] == expected
    assert comparison["paired_samples"] == 3
    assert comparison["paired_resampling_units"] == 2


def test_engineer_decision_requires_a_complete_non_smoke_run(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _plan("chosen", "other"),
        lambda candidate: ([SampleResult("sample", {"quality": float(candidate == "chosen")}, 0.01)], {"p95_latency_seconds": 0.01}),
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_summary": str(result.artifact_directory / "summary.json"),
                "source_run_id": result.run_id,
                "selected_candidates": ["chosen"],
                "selected_by": "benchmark engineer",
                "selected_date": "2026-08-26",
                "reason": "Best fit after reviewing quality and operational evidence.",
            }
        ),
        encoding="utf-8",
    )

    decision = load_engineer_decision(
        decision_path,
        exact=1,
        expected_source=("test-suite", "completion", "development"),
    )
    assert decision.selected_candidates == ("chosen",)
    with pytest.raises(ValueError, match="stage 'other'"):
        load_engineer_decision(
            decision_path, expected_source=("test-suite", "other", "development")
        )
    with pytest.raises(ValueError, match="suite 'other'"):
        load_engineer_decision(
            decision_path, expected_source=("other", "completion", "development")
        )
    with pytest.raises(ValueError, match="profile 'validation'"):
        load_engineer_decision(
            decision_path, expected_source=("test-suite", "completion", "validation")
        )

    summary_path = result.artifact_directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["complete"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_engineer_decision(decision_path)


def test_incomplete_candidate_and_parent_are_marked_failed_in_tracking(
    tmp_path: Path, monkeypatch
) -> None:
    class RecordingTracker:
        def __init__(self) -> None:
            self.active: list[str] = []
            self.failed: list[str] = []

        @contextmanager
        def run(self, name: str, *, nested: bool = False):
            del nested
            self.active.append(name)
            try:
                yield name
            finally:
                if self.active and self.active[-1] == name:
                    self.active.pop()

        def parameters(self, values) -> None:
            del values

        def metrics(self, values) -> None:
            del values

        def tags(self, values) -> None:
            del values

        def artifact(self, path, artifact_path=None) -> None:
            del path, artifact_path

        def mark_failed(self) -> None:
            self.failed.append(self.active.pop())

    tracking = RecordingTracker()
    monkeypatch.setattr(benchmark_runner, "tracker", lambda **_kwargs: tracking)

    result = _run(
        tmp_path,
        _plan("incomplete"),
        lambda _candidate: ([SampleResult("sample", {}, 0.01)], {}),
    )

    assert not result.complete
    assert tracking.failed[0] == "incomplete"
    assert tracking.failed[1].startswith("test-suite-completion-")


def test_protocol_identity_and_resolved_settings_are_recorded_for_parent_and_child(
    tmp_path: Path, monkeypatch
) -> None:
    class RecordingTracker:
        def __init__(self) -> None:
            self.active: list[str] = []
            self.parameter_calls: list[tuple[str, dict[str, object]]] = []
            self.artifacts: list[tuple[str, Path, str | None]] = []

        @contextmanager
        def run(self, name: str, *, nested: bool = False):
            del nested
            self.active.append(name)
            try:
                yield name
            finally:
                self.active.pop()

        def parameters(self, values) -> None:
            self.parameter_calls.append((self.active[-1], dict(values)))

        def metrics(self, values) -> None:
            del values

        def tags(self, values) -> None:
            del values

        def artifact(self, path, artifact_path=None) -> None:
            self.artifacts.append((self.active[-1], Path(path), artifact_path))

        def mark_failed(self) -> None:
            raise AssertionError("The protocol-recording fixture must remain complete")

    protocol = load_generation_protocol()
    tracking = RecordingTracker()
    monkeypatch.setattr(benchmark_runner, "tracker", lambda **_kwargs: tracking)
    result = run_benchmark(
        _plan("candidate"),
        lambda _candidate: (
            [SampleResult("sample", {"quality": 1.0}, 0.01)],
            {"p95_latency_seconds": 0.01},
        ),
        dataset_checksum="fixed-checksum",
        directions={"quality": "max", "operational.p95_latency_seconds": "min"},
        primary_metric="quality",
        artifact_root=tmp_path,
        protocols={"generation": protocol.meta},
    )

    protocol_artifact = json.loads(
        (result.artifact_directory / "generation_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol_artifact["checksum"] == protocol.meta.checksum
    candidate_artifact = json.loads(
        (result.artifact_directory / "candidates" / "candidate.json").read_text(
            encoding="utf-8"
        )
    )
    recorded = candidate_artifact["parameters"]["protocols"]["generation"]
    assert recorded["version"] == protocol.meta.version
    assert recorded["resolved"]["generation"]["maximum_answer_tokens"] == 256
    assert any(
        values.get("protocol.generation.checksum") == protocol.meta.checksum
        for _run_name, values in tracking.parameter_calls
    )
    assert any(
        path == protocol.meta.source_path
        and artifact_path == "inputs/protocols/generation"
        for _run_name, path, artifact_path in tracking.artifacts
    )
