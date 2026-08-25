"""Shared execution shell for direct benchmark scripts."""

from __future__ import annotations

import json
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
    gates: Mapping[str, tuple[str, float]] | None = None,
    revisions: Mapping[str, str] | None = None,
    no_mlflow: bool = False,
    artifact_root: Path = Path("artifacts/benchmarks"),
) -> BenchmarkResult:
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
    }
    plan_path = directory / "plan.json"
    provenance_path = directory / "provenance.json"
    atomic_write_json(plan_path, asdict(plan))
    atomic_write_json(provenance_path, provenance)
    tracking = tracker(disabled=no_mlflow, experiment=f"EduMind / {plan.suite}")
    run_fingerprint = stable_hash({"plan": asdict(plan), "provenance": provenance})
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
                "run_fingerprint": run_fingerprint,
                "git_commit": provenance["git"].get("commit"),
                "git_dirty": provenance["git"].get("dirty"),
                "git_dirty_hash": provenance["git"].get("dirty_hash"),
                "hardware": json.dumps(provenance["hardware"], sort_keys=True),
                "model_revisions": json.dumps(provenance["model_revisions"], sort_keys=True),
                "dependency_locks": json.dumps(provenance["dependency_locks"], sort_keys=True),
            }
        )
        tracking.artifact(plan_path)
        tracking.artifact(provenance_path)
        for candidate in order:
            results.append(
                _run_candidate(
                    plan, candidate, evaluator, directory, tracking, run_fingerprint
                )
            )

        successful = [result for result in results if result.status == "success"]
        rows = {
            result.candidate: {
                **result.metrics,
                **{f"operational.{key}": value for key, value in result.operational.items()},
            }
            for result in successful
        }
        failures = {
            name: _failed_gates(row, gates or {})
            for name, row in rows.items()
            if _failed_gates(row, gates or {})
        }
        eligible = {name: row for name, row in rows.items() if name not in failures}
        available_directions = {
            name: direction
            for name, direction in directions.items()
            if all(name in row for row in eligible.values())
        }
        pareto = (
            _interval_aware_pareto(
                eligible,
                {result.candidate: result for result in successful},
                available_directions,
            )
            if plan.profile in {"standard", "full"} and eligible
            else []
        )
        summary = {
            "run_id": run_id,
            "fingerprint": run_fingerprint,
            "mlflow_run_id": mlflow_run_id,
            "plan": asdict(plan),
            "provenance": provenance,
            "candidates": [_payload(result, include_samples=False) for result in results],
            "gate_failures": failures,
            "pareto_candidates": pareto,
            "paired_comparisons": _paired_comparisons(
                successful,
                available_directions,
                resamples=500 if plan.profile == "smoke" else plan.bootstrap_resamples,
                seed=plan.seed,
            ),
            "authoritative": plan.profile in {"standard", "full"} and bool(eligible),
        }
        summary_path = directory / "summary.json"
        atomic_write_json(summary_path, summary)
        tracking.artifact(summary_path)
    return BenchmarkResult(
        run_id,
        plan,
        provenance,
        tuple(results),
        tuple(pareto),
        bool(summary["authoritative"]),
        directory,
    )


def _run_candidate(
    plan, candidate, evaluator, directory, tracking, run_fingerprint
) -> CandidateResult:
    try:
        with tracking.run(candidate, nested=True):
            tracking.parameters({"candidate": candidate, "profile": plan.profile})
            try:
                with ResourceMonitor() as resources:
                    evaluated = evaluator(candidate)
                    samples, operational = evaluated[:2]
                    candidate_metrics = evaluated[2] if len(evaluated) == 3 else {}
                    if len(evaluated) >= 4:
                        candidate_metrics = evaluated[2]
                        tracking.parameters(evaluated[3])
            except Exception as exc:
                tracking.parameters(
                    {
                        "candidate_status": "failed",
                        "candidate_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            if not samples:
                raise RuntimeError("Candidate produced no samples")
            metrics, intervals = aggregate_samples(
                samples,
                resamples=500 if plan.profile == "smoke" else plan.bootstrap_resamples,
                seed=plan.seed,
            )
            metrics = {**metrics, **candidate_metrics}
            operational = {**operational, **resources.metrics()}
            result = CandidateResult(
                candidate,
                "success",
                stable_hash({"run": run_fingerprint, "candidate": candidate}),
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
            sample_path = _write_samples(directory, candidate, samples)
            tracking.artifact(sample_path)
            candidate_path = directory / "candidates" / f"{_safe(candidate)}.json"
            atomic_write_json(candidate_path, _payload(result, include_samples=False))
            tracking.artifact(candidate_path)
            return result
    except Exception as exc:
        result = CandidateResult(
            candidate,
            "failed",
            stable_hash({"run": run_fingerprint, "candidate": candidate}),
            {},
            {},
            (),
            {},
            f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(
            directory / "candidates" / f"{_safe(candidate)}.json",
            _payload(result, include_samples=False),
        )
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


def _failed_gates(
    values: Mapping[str, float], gates: Mapping[str, tuple[str, float]]
) -> list[str]:
    failures: list[str] = []
    for metric, (direction, threshold) in gates.items():
        value = values.get(metric)
        if value is None:
            failures.append(f"{metric}:missing")
        elif direction == "max" and value < threshold:
            failures.append(f"{metric}<{threshold}")
        elif direction == "min" and value > threshold:
            failures.append(f"{metric}>{threshold}")
    return failures


def _interval_aware_pareto(rows, results, directions) -> list[str]:
    """Pareto set with overlapping quality intervals treated as ties.

    If every quality objective overlaps for a pair, the documented operational
    tie-break order is applied lexicographically: p95 latency, RAM, then storage.
    """
    if not directions:
        return list(rows)
    names = list(rows)
    selected = []
    for candidate in names:
        if not any(
            other != candidate
            and _dominates(other, candidate, rows, results, directions)
            for other in names
        ):
            selected.append(candidate)
    return selected


def _dominates(left, right, rows, results, directions) -> bool:
    quality = [name for name in directions if not name.startswith("operational.")]
    all_quality_tied = bool(quality) and all(
        _quality_tied(results[left], results[right], name) for name in quality
    )
    if all_quality_tied:
        for metric in (
            "operational.p95_latency_seconds",
            "operational.peak_ram_mb",
            "operational.storage_bytes",
            "operational.persistent_storage_bytes",
        ):
            if metric not in rows[left] or metric not in rows[right]:
                continue
            if rows[left][metric] != rows[right][metric]:
                return rows[left][metric] < rows[right][metric]
        return False

    at_least_as_good = True
    strictly_better = False
    for metric, direction in directions.items():
        if metric in quality and _quality_tied(results[left], results[right], metric):
            continue
        left_value, right_value = rows[left][metric], rows[right][metric]
        better_or_equal = left_value >= right_value if direction == "max" else left_value <= right_value
        better = left_value > right_value if direction == "max" else left_value < right_value
        at_least_as_good &= better_or_equal
        strictly_better |= better
    return at_least_as_good and strictly_better


def _quality_tied(left: CandidateResult, right: CandidateResult, metric: str) -> bool:
    left_interval = left.intervals.get(metric)
    right_interval = right.intervals.get(metric)
    if not left_interval or not right_interval:
        return False
    return not (
        float(left_interval["upper"]) < float(right_interval["lower"])
        or float(right_interval["upper"]) < float(left_interval["lower"])
    )


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
