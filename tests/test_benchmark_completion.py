from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.benchmarks.common.runner as benchmark_runner
from experiments.benchmarks.common import process as benchmark_process
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.rag.generation.protocol import (
    load_protocol as load_generation_protocol,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "key", ["peak_vram_mb", "peak_visual_vram_mb", "peak_vram_mib"]
)
def test_supervisor_recognizes_each_suite_vram_field(key):
    assert (
        benchmark_process._reported_peak_vram_mb({"operational": {key: 3600.0}})
        == 3600.0
    )


@pytest.mark.parametrize("domain", ("audio", "document"))
def test_worker_launcher_uses_repository_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, domain: str
) -> None:
    launched = []

    def fake_popen(command, **options):
        launched.append((command, options))
        Path(command[-1]).write_text('{"status":"success"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, communicate=lambda **_kwargs: ("", ""))

    monkeypatch.setattr(benchmark_process.subprocess, "Popen", fake_popen)
    script = ROOT / f"experiments/benchmarks/extraction/{domain}/worker.py"
    result = benchmark_process.run_json_worker(
        script,
        {},
        device="cpu",
        prefix="worker-test-",
        error_label="test",
        temporary_root=tmp_path,
    )
    assert result == {"status": "success"}
    command, options = launched[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        f"experiments.benchmarks.extraction.{domain}.worker",
    ]
    assert options["cwd"] == ROOT


@pytest.mark.parametrize("output", ["{", "[]", "null"])
def test_worker_launcher_classifies_unusable_output_as_missing_results(
    tmp_path, monkeypatch, output
):
    def launch(command, **options):
        Path(command[-1]).write_text(output, encoding="utf-8")
        return SimpleNamespace(returncode=0, communicate=lambda **kwargs: ("", ""))

    monkeypatch.setattr(benchmark_process.subprocess, "Popen", launch)
    with pytest.raises(benchmark_process.WorkerExecutionError):
        benchmark_process.run_json_worker(
            ROOT / "experiments/benchmarks/rag/chunking_embedding/worker.py",
            {},
            device="cpu",
            prefix="bad-worker-output-",
            error_label="test",
            temporary_root=tmp_path,
        )


def test_worker_setup_failure_preserves_input_contract_report(tmp_path, monkeypatch):
    from experiments.benchmarks.common.preflight import _probe_candidate
    from experiments.benchmarks.rag.evaluation import InputCompatibilityError

    source = tmp_path / "input.json"
    destination = tmp_path / "output.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker", str(source), str(destination)])
    report = {"reason_code": "input_length_exceeded", "maximum_length": 8192}

    def fail(_payload):
        raise InputCompatibilityError("oversized input", report)

    assert benchmark_process.json_worker_main(fail) == 1
    output = destination.read_text(encoding="utf-8")

    def launch(command, **options):
        Path(command[-1]).write_text(output, encoding="utf-8")
        return SimpleNamespace(returncode=1, communicate=lambda **kwargs: ("", ""))

    monkeypatch.setattr(benchmark_process.subprocess, "Popen", launch)
    result = _probe_candidate(
        "candidate",
        lambda _: benchmark_process.run_json_worker(
            ROOT / "experiments/benchmarks/rag/chunking_embedding/worker.py",
            {},
            device="cpu",
            prefix="input-failure-",
            error_label="test",
            temporary_root=tmp_path,
        ),
    )
    assert result["status"] == "blocked"
    assert result["input_validation"] == report


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
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
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
            (
                "import importlib, sys; sys.argv = ['test', '--unexpected']; "
                f"importlib.import_module({module!r})"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
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
        [sys.executable, "-m", module],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
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


def test_missing_required_metric_fails_candidate_but_keeps_samples(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        _plan("incomplete"),
        lambda _candidate: (
            [SampleResult("sample-1", {}, 0.01)],
            {"p95_latency_seconds": 0.01},
        ),
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
    assert any(
        "different sample set" in problem for problem in result.completion_problems
    )


def test_generic_runner_does_not_publish_surviving_partial_quality(tmp_path):
    result = _run(
        tmp_path,
        _plan("partial"),
        lambda _: (
            [
                SampleResult("scored", {"quality": 1.0}, 0.01),
                SampleResult("unavailable", {"quality": None}, None),
            ],
            {"p95_latency_seconds": 0.01},
        ),
    )
    assert not result.complete
    child = result.candidates[0]
    assert "quality" not in child.metrics
    assert len(child.samples) == 2
    partial = json.loads(
        (
            result.artifact_directory / "candidates/partial/partial_quality.json"
        ).read_text(encoding="utf-8")
    )
    assert partial["quality"]["estimate"] == 1.0
    assert partial["quality"]["contributing"] == 1
    assert partial["quality"]["scheduled"] == 2


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
        result.artifact_directory / "candidates" / "failed" / "query_metrics.parquet"
    ).is_file()


def test_incomplete_execution_report_is_not_relabelled_as_runner_rejection(tmp_path):
    def evaluator(_candidate):
        raise CandidateExecutionError(
            "required attempts were not executed",
            artifacts={
                "validation_report": {
                    "status": "incomplete",
                    "reason_code": "execution_coverage_incomplete",
                    "errors": [],
                }
            },
        )

    result = _run(tmp_path, _plan("interrupted"), evaluator)
    report = json.loads(
        (
            result.artifact_directory / "candidates/interrupted/validation_report.json"
        ).read_text(encoding="utf-8")
    )
    assert not result.complete
    assert report["status"] == "incomplete"
    assert report["reason_code"] == "execution_coverage_incomplete"
    assert report["errors"] == []


def test_mlflow_keeps_child_identity_null_evidence_and_failed_siblings(
    tmp_path, monkeypatch, request
):
    mlflow = pytest.importorskip("mlflow")
    previous_uri = mlflow.get_tracking_uri()
    request.addfinalizer(lambda: mlflow.set_tracking_uri(previous_uri))
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment(
        "EduMind / tracking-check", artifact_location=(tmp_path / "mlruns").as_uri()
    )

    def evaluator(candidate):
        if candidate == "failed":
            raise CandidateExecutionError("recorded test error")
        return (
            [SampleResult("same", {"quality": 1.0}, 0.01)],
            {},
            {"quality": 1.0, "optional": None},
            {},
            {},
            {"availability": {"optional": {"value": None, "status": "inapplicable"}}},
        )

    result = run_benchmark(
        replace(
            _plan("passed", "failed"), suite="tracking-check", stage="tracking-check"
        ),
        evaluator,
        dataset_checksum="synthetic-tracking-test",
        directions={"quality": "max", "optional": "descriptive"},
        primary_metric="quality",
        nullable_metrics=("optional",),
        monitor_resources=False,
        artifact_root=tmp_path / "artifacts",
    )
    assert not result.complete
    runs = client.search_runs([experiment])
    parents = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    assert len(parents) == 1 and parents[0].info.status == "FAILED"
    for child in result.candidates:
        saved = client.get_run(child.mlflow_run_id)
        assert saved.data.tags["mlflow.parentRunId"] == parents[0].info.run_id
        assert saved.info.status == (
            "FINISHED" if child.status == "success" else "FAILED"
        )
        assert "optional" not in saved.data.metrics
    assert any(
        path.name == "availability.json"
        for path in (result.artifact_directory / "candidates/passed").iterdir()
    )


def test_runner_rejection_updates_validation_report_and_preserves_evidence(tmp_path):
    result = run_benchmark(
        _plan("candidate"),
        lambda _: (
            [SampleResult("q", {"quality": 1.0}, 0.01)],
            {"peak_vram_mb": 4000},
            {"quality": 1.0},
            {},
            {},
            {
                "validation_report": {"status": "passed", "errors": []},
                "timings": [{"latency_ms": 10}],
            },
        ),
        dataset_checksum="fixed",
        directions={"quality": "max"},
        primary_metric="quality",
        no_mlflow=True,
        artifact_root=tmp_path,
        monitor_resources=False,
        operational_maximums={"peak_vram_mb": 3584},
    )
    candidate = result.candidates[0]
    report = json.loads(
        (
            result.artifact_directory / "candidates/candidate/validation_report.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate.status == "failed"
    assert candidate.operational["peak_vram_mb"] == 4000
    assert report["status"] == "failed"
    assert "3584" in report["errors"][0]
    assert (
        result.artifact_directory / "candidates/candidate/timings.parquet"
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
        lambda candidate: (
            [SampleResult("sample", {"quality": float(candidate == "chosen")}, 0.01)],
            {"p95_latency_seconds": 0.01},
        ),
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
    assert tracking.failed[1] == "test-suite-completion"


def test_protocol_identity_and_resolved_settings_are_recorded_for_parent_and_child(
    tmp_path: Path, monkeypatch
) -> None:
    class RecordingTracker:
        def __init__(self) -> None:
            self.active: list[str] = []
            self.parameter_calls: list[tuple[str, dict[str, object]]] = []
            self.artifacts: list[tuple[str, Path, str | None]] = []
            self.tag_calls: list[tuple[str, dict[str, object]]] = []

        @contextmanager
        def run(self, name: str, *, nested: bool = False):
            del nested
            self.active.append(name)
            try:
                yield name
            finally:
                self.active.pop()

        def parameters(self, values) -> None:
            for run, previous in self.parameter_calls:
                if run == self.active[-1]:
                    for key in previous.keys() & values.keys():
                        assert str(previous[key]) == str(values[key]), key
            self.parameter_calls.append((self.active[-1], dict(values)))

        def metrics(self, values) -> None:
            del values

        def tags(self, values) -> None:
            self.tag_calls.append((self.active[-1], dict(values)))

        def artifact(self, path, artifact_path=None) -> None:
            self.artifacts.append((self.active[-1], Path(path), artifact_path))

        def mark_failed(self) -> None:
            raise AssertionError("The protocol-recording fixture must remain complete")

    protocol = load_generation_protocol()
    tracking = RecordingTracker()
    monkeypatch.setattr(benchmark_runner, "tracker", lambda **_kwargs: tracking)
    decision = tmp_path / "decision.json"
    decision.write_text("{}", encoding="utf-8")
    result = run_benchmark(
        replace(_plan("candidate"), settings={"device": "cpu", "dtype": "float32"}),
        lambda _candidate: (
            [SampleResult("sample", {"quality": 1.0}, 0.01)],
            {"p95_latency_seconds": 0.01},
        ),
        dataset_checksum="fixed-checksum",
        directions={"quality": "max", "operational.p95_latency_seconds": "min"},
        primary_metric="quality",
        artifact_root=tmp_path,
        protocols={"generation": protocol.meta},
        decision_files={"shortlist": decision},
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
    assert recorded["resolved"]["generation"]["context_tokens"] == 8192
    assert recorded["resolved"]["generation"]["measured_seeds"] == [42, 43, 44]
    assert any(
        values.get("protocol.generation.checksum") == protocol.meta.checksum
        for _run_name, values in tracking.parameter_calls
    )
    assert any(
        path == protocol.meta.source_path
        and artifact_path == "inputs/protocols/generation"
        for _run_name, path, artifact_path in tracking.artifacts
    )
    assert len(tracking.tag_calls) >= 2
    parent_tags = tracking.tag_calls[0][1]
    child_tags = tracking.tag_calls[1][1]
    assert parent_tags["phase"] == child_tags["phase"] == "development"
    assert parent_tags["run_type"] == "quality-comparison"
    assert child_tags["run_type"] == "quality-candidate"
    assert parent_tags["decision.fingerprint"]
    assert parent_tags["decision.fingerprint"] == child_tags["decision.fingerprint"]
