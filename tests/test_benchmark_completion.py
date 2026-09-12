from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark
import experiments.benchmarks.common.runner as benchmark_runner


def _plan(*candidates: str) -> BenchmarkPlan:
    return BenchmarkPlan(
        "test-suite",
        "completion",
        "standard",
        "fixed-test-data",
        candidates,
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
        expected_source=("test-suite", "completion", "standard"),
    )
    assert decision.selected_candidates == ("chosen",)
    with pytest.raises(ValueError, match="stage 'other'"):
        load_engineer_decision(
            decision_path, expected_source=("test-suite", "other", "standard")
        )
    with pytest.raises(ValueError, match="suite 'other'"):
        load_engineer_decision(
            decision_path, expected_source=("other", "completion", "standard")
        )
    with pytest.raises(ValueError, match="profile 'full'"):
        load_engineer_decision(
            decision_path, expected_source=("test-suite", "completion", "full")
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
