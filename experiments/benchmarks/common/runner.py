"""Shared execution shell for direct benchmark scripts."""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from edumind.common.artifacts import (
    atomic_write_json,
    sha256_file,
    stable_hash,
)
from edumind.common.paths import PROJECT_ROOT

from .contracts import BenchmarkPlan, BenchmarkResult, CandidateResult, SampleResult
from .provenance import git_provenance, hardware_summary
from .resources import ResourceMonitor
from .metrics import paired_bootstrap_interval
from .statistics import aggregate_samples
from .tracking import tracker

Evaluator = Callable[
    [str],
    tuple[list[SampleResult], Mapping[str, float]]
    | tuple[list[SampleResult], Mapping[str, float], Mapping[str, float]]
    | tuple[
        list[SampleResult],
        Mapping[str, float],
        Mapping[str, float],
        Mapping[str, object],
    ],
]


def run_benchmark(
    plan: BenchmarkPlan,
    evaluator: Evaluator,
    *,
    dataset_checksum: str,
    directions: Mapping[str, str],
    primary_metric: str,
    revisions: Mapping[str, str] | None = None,
    decision_files: Mapping[str, Path] | None = None,
    no_mlflow: bool = False,
    artifact_root: Path = Path("artifacts/benchmarks"),
) -> BenchmarkResult:
    _validate_metric_contract(directions, primary_metric)
    run_name = f"{plan.suite}-{plan.stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    directory = artifact_root / plan.suite / plan.stage / run_id
    provenance = {
        "dataset_checksum": dataset_checksum,
        "git": git_provenance(PROJECT_ROOT),
        "hardware": hardware_summary(),
        "model_revisions": dict(revisions or {}),
        "dependency_locks": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                PROJECT_ROOT / "requirements/app.lock",
                PROJECT_ROOT / "requirements/benchmarks.lock",
            )
            if path.is_file()
        },
        "seed": plan.seed,
        "engineer_decisions": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (decision_files or {}).items()
        },
    }
    plan_path = directory / "plan.json"
    provenance_path = directory / "provenance.json"
    metric_contract = {
        "primary_metric": primary_metric,
        "directions": dict(directions),
        "required_metrics": list(directions),
    }
    plan_payload = {**asdict(plan), "metric_contract": metric_contract}
    atomic_write_json(plan_path, plan_payload)
    atomic_write_json(provenance_path, provenance)
    tracking = tracker(disabled=no_mlflow, experiment=f"EduMind / {plan.suite}")
    run_fingerprint = stable_hash({"plan": plan_payload, "provenance": provenance})
    results: list[CandidateResult] = []
    order = list(plan.candidates)
    random.Random(plan.seed).shuffle(order)
    with tracking.run(run_name) as mlflow_run_id:
        tracking.parameters(
            {
                "profile": plan.profile,
                "stage": plan.stage,
                "dataset": plan.dataset,
                "dataset_checksum": dataset_checksum,
                "seed": plan.seed,
                "primary_metric": primary_metric,
                "required_metrics": json.dumps(list(directions)),
                "run_fingerprint": run_fingerprint,
                "git_commit": provenance["git"].get("commit"),
                "git_dirty": provenance["git"].get("dirty"),
                "git_dirty_hash": provenance["git"].get("dirty_hash"),
                "hardware": json.dumps(provenance["hardware"], sort_keys=True),
                "model_revisions": json.dumps(provenance["model_revisions"], sort_keys=True),
                "dependency_locks": json.dumps(provenance["dependency_locks"], sort_keys=True),
                "engineer_decisions": json.dumps(
                    provenance["engineer_decisions"], sort_keys=True
                ),
            }
        )
        tracking.artifact(plan_path)
        tracking.artifact(provenance_path)
        for decision_name, decision_path in (decision_files or {}).items():
            tracking.artifact(decision_path, f"engineer-decisions/{decision_name}")
        for candidate in order:
            results.append(
                _run_candidate(
                    plan,
                    candidate,
                    evaluator,
                    directory,
                    tracking,
                    run_fingerprint,
                    tuple(directions),
                )
            )

        successful = [result for result in results if result.status == "success"]
        problems = _completion_problems(plan, results)
        complete = not problems
        summary = {
            "run_id": run_id,
            "fingerprint": run_fingerprint,
            "mlflow_run_id": mlflow_run_id,
            "plan": asdict(plan),
            "metric_contract": metric_contract,
            "provenance": provenance,
            "candidates": [_payload(result, include_samples=False) for result in results],
            "paired_comparisons": _paired_comparisons(
                successful,
                directions,
                resamples=500 if plan.profile == "smoke" else plan.bootstrap_resamples,
                seed=plan.seed,
            ),
            "complete": complete,
            "completion": {
                "planned_candidates": len(plan.candidates),
                "successful_candidates": len(successful),
                "failed_candidates": len(plan.candidates) - len(successful),
                "sample_count": len(successful[0].samples) if successful else 0,
                "problems": problems,
            },
            "selection": {
                "made_by_runner": False,
                "instruction": (
                    "Review the MLflow child runs and artifacts. After a complete standard/full "
                    "run, record any advancement in a separate engineer-decision JSON file."
                ),
            },
        }
        summary_path = directory / "summary.json"
        atomic_write_json(summary_path, summary)
        tracking.artifact(summary_path)
        tracking.metrics(
            {
                "benchmark_complete": float(complete),
                "successful_candidates": float(len(successful)),
                "failed_candidates": float(len(plan.candidates) - len(successful)),
            }
        )
    return BenchmarkResult(
        run_id,
        plan,
        provenance,
        tuple(results),
        complete,
        tuple(problems),
        directory,
    )


def _run_candidate(
    plan,
    candidate,
    evaluator,
    directory,
    tracking,
    run_fingerprint,
    required_metrics,
) -> CandidateResult:
    samples: list[SampleResult] = []
    metrics: dict[str, float] = {}
    intervals: dict[str, dict[str, float]] = {}
    operational: dict[str, float] = {}
    fingerprint = stable_hash({"run": run_fingerprint, "candidate": candidate})
    with tracking.run(candidate, nested=True):
        try:
            tracking.parameters({"candidate": candidate, "profile": plan.profile})
            resources = ResourceMonitor()
            with resources:
                evaluated = evaluator(candidate)
            samples = list(evaluated[0])
            operational = {**dict(evaluated[1]), **resources.metrics()}
            candidate_metrics = dict(evaluated[2]) if len(evaluated) >= 3 else {}
            if len(evaluated) >= 4:
                tracking.parameters(evaluated[3])
            if not samples:
                raise RuntimeError("Candidate produced no samples")
            _validate_sample_ids(samples)
            sample_path = _write_samples(directory, candidate, samples)
            tracking.artifact(sample_path)
            metrics, intervals = aggregate_samples(
                samples,
                resamples=500 if plan.profile == "smoke" else plan.bootstrap_resamples,
                seed=plan.seed,
            )
            metrics = {**metrics, **candidate_metrics}
            _validate_required_metrics(metrics, operational, required_metrics)
            result = CandidateResult(
                candidate,
                "success",
                fingerprint,
                metrics,
                intervals,
                tuple(samples),
                operational,
            )
            tracking.metrics(
                {
                    **metrics,
                    **{
                        f"{name}.ci_{bound}": values[bound]
                        for name, values in intervals.items()
                        for bound in ("lower", "upper")
                    },
                    **{f"operational.{key}": value for key, value in operational.items()},
                }
            )
            tracking.parameters({"candidate_status": "success"})
            candidate_path = directory / "candidates" / f"{_safe(candidate)}.json"
            atomic_write_json(candidate_path, _payload(result, include_samples=False))
            tracking.artifact(candidate_path)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = CandidateResult(
                candidate,
                "failed",
                fingerprint,
                metrics,
                intervals,
                tuple(samples),
                operational,
                error,
            )
            tracking.parameters({"candidate_status": "failed", "candidate_error": error})
            partial_metrics = {
                **metrics,
                **{f"operational.{key}": value for key, value in operational.items()},
            }
            tracking.metrics(
                {
                    key: float(value)
                    for key, value in partial_metrics.items()
                    if _finite_number(value)
                }
            )
            if samples:
                sample_path = directory / "samples" / f"{_safe(candidate)}.parquet"
                if not sample_path.is_file():
                    sample_path = _write_samples(directory, candidate, samples)
                tracking.artifact(sample_path)
            candidate_path = directory / "candidates" / f"{_safe(candidate)}.json"
            atomic_write_json(candidate_path, _payload(result, include_samples=False))
            tracking.artifact(candidate_path)
            return result


def _write_samples(directory: Path, candidate: str, samples: list[SampleResult]) -> Path:
    rows = [
        {
            "sample_id": sample.sample_id,
            "latency_seconds": sample.latency_seconds,
            **{f"metric.{key}": value for key, value in sample.metrics.items()},
            "metadata": json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True),
        }
        for sample in samples
    ]
    path = directory / "samples" / f"{_safe(candidate)}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _payload(result: CandidateResult, *, include_samples: bool) -> dict[str, object]:
    payload = asdict(result)
    payload["samples"] = [asdict(sample) for sample in result.samples] if include_samples else []
    return payload


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _validate_metric_contract(directions: Mapping[str, str], primary_metric: str) -> None:
    if not directions:
        raise ValueError("A benchmark must declare at least one required metric")
    invalid = sorted(name for name, direction in directions.items() if direction not in {"min", "max"})
    if invalid:
        raise ValueError(f"Metrics have invalid directions: {', '.join(invalid)}")
    if primary_metric not in directions:
        raise ValueError("primary_metric must be one of the benchmark's required metrics")


def _validate_sample_ids(samples: list[SampleResult]) -> None:
    identifiers = [sample.sample_id for sample in samples]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every sample result must have a non-empty sample_id")
    duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
    if duplicates:
        raise ValueError(f"Candidate produced duplicate sample IDs: {', '.join(duplicates[:10])}")


def _validate_required_metrics(metrics, operational, required_metrics) -> None:
    values = {
        **metrics,
        **{f"operational.{name}": value for name, value in operational.items()},
    }
    missing = [name for name in required_metrics if name not in values]
    invalid = [
        name for name in required_metrics if name in values and not _finite_number(values[name])
    ]
    problems = []
    if missing:
        problems.append("missing: " + ", ".join(missing))
    if invalid:
        problems.append("non-finite: " + ", ".join(invalid))
    if problems:
        raise ValueError("Required metric contract failed (" + "; ".join(problems) + ")")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _completion_problems(plan: BenchmarkPlan, results: list[CandidateResult]) -> list[str]:
    problems: list[str] = []
    returned = [result.candidate for result in results]
    if returned != list(dict.fromkeys(returned)):
        problems.append("runner returned duplicate candidate results")
    missing_candidates = sorted(set(plan.candidates) - set(returned))
    unexpected_candidates = sorted(set(returned) - set(plan.candidates))
    if missing_candidates:
        problems.append("missing candidate results: " + ", ".join(missing_candidates))
    if unexpected_candidates:
        problems.append("unexpected candidate results: " + ", ".join(unexpected_candidates))

    failed = [result for result in results if result.status != "success"]
    for result in failed:
        problems.append(f"candidate {result.candidate} failed: {result.error or 'unknown error'}")

    successful = [result for result in results if result.status == "success"]
    if successful:
        reference = {sample.sample_id for sample in successful[0].samples}
        for result in successful[1:]:
            observed = {sample.sample_id for sample in result.samples}
            if observed != reference:
                missing = sorted(reference - observed)
                extra = sorted(observed - reference)
                detail = []
                if missing:
                    detail.append("missing " + ", ".join(missing[:10]))
                if extra:
                    detail.append("extra " + ", ".join(extra[:10]))
                problems.append(
                    f"candidate {result.candidate} evaluated a different sample set "
                    f"({' ; '.join(detail)})"
                )
    elif plan.candidates:
        problems.append("no candidate completed successfully")
    return problems


def _paired_comparisons(results, directions, *, resamples: int, seed: int):
    comparisons = []
    quality_metrics = [name for name in directions if not name.startswith("operational.")]
    for left_index, left in enumerate(results):
        left_samples = {sample.sample_id: sample for sample in left.samples}
        for right in results[left_index + 1 :]:
            right_samples = {sample.sample_id: sample for sample in right.samples}
            shared_ids = sorted(left_samples.keys() & right_samples.keys())
            metrics = {}
            for metric in quality_metrics:
                paired_ids = [
                    sample_id
                    for sample_id in shared_ids
                    if metric in left_samples[sample_id].metrics
                    and metric in right_samples[sample_id].metrics
                ]
                if not paired_ids:
                    continue
                interval = paired_bootstrap_interval(
                    [left_samples[sample_id].metrics[metric] for sample_id in paired_ids],
                    [right_samples[sample_id].metrics[metric] for sample_id in paired_ids],
                    resamples=resamples,
                    seed=seed,
                )
                metrics[metric] = {
                    "left_minus_right": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "confidence": interval.confidence,
                    "direction": directions[metric],
                    "paired_samples": len(paired_ids),
                }
            comparisons.append({"left": left.candidate, "right": right.candidate, "metrics": metrics})
    return comparisons
