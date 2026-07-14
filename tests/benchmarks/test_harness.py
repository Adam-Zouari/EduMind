from __future__ import annotations

import json

from edumind.benchmarks.contracts import BenchmarkPlan, SampleResult
from edumind.benchmarks.harness import BenchmarkHarness


def test_only_successful_complete_candidates_are_cached(tmp_path) -> None:
    harness = BenchmarkHarness(tmp_path)
    plan = BenchmarkPlan("test", "stage", "smoke", "data", ("bad",), bootstrap_resamples=50)
    calls = 0

    def fail(candidate):
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    first = harness.run(plan, fail, dataset_checksum="data", directions={"quality": "max"})
    second = harness.run(plan, fail, dataset_checksum="data", directions={"quality": "max"})
    assert calls == 2
    assert first.candidates[0].status == second.candidates[0].status == "failed"


def test_successful_candidate_cache_replays_samples(tmp_path) -> None:
    harness = BenchmarkHarness(tmp_path)
    plan = BenchmarkPlan("test", "stage", "smoke", "data", ("good",), bootstrap_resamples=50)
    calls = 0

    def evaluate(candidate):
        nonlocal calls
        calls += 1
        return [SampleResult("one", {"quality": 1.0}, 0.1)], {"latency": 0.1}

    harness.run(plan, evaluate, dataset_checksum="data", directions={"quality": "max"})
    result = harness.run(plan, evaluate, dataset_checksum="data", directions={"quality": "max"})
    assert calls == 1
    assert result.candidates[0].samples[0].sample_id == "one"


def test_hard_gates_run_before_pareto_and_are_reported(tmp_path) -> None:
    harness = BenchmarkHarness(tmp_path)
    plan = BenchmarkPlan("test", "gated", "standard", "data", ("slow",), bootstrap_resamples=50)

    def evaluate(candidate):
        return [SampleResult("one", {"quality": 1.0}, 0.1)], {"p95": 31.0}

    result = harness.run(
        plan,
        evaluate,
        dataset_checksum="data",
        directions={"quality": "max"},
        hard_gates={"operational.p95": ("min", 30.0)},
    )
    assert result.pareto_candidates == () and result.authoritative is False
    summary = json.loads((result.artifact_directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["gate_failures"]["slow"] == ["operational.p95>30.0"]
