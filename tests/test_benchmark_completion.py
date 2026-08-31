from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark


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

    decision = load_engineer_decision(decision_path, exact=1)
    assert decision.selected_candidates == ("chosen",)

    summary_path = result.artifact_directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["complete"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_engineer_decision(decision_path)
