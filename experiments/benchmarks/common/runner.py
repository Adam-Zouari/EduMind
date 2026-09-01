"""Shared execution shell for direct benchmark scripts."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
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
    ]
    | tuple[
        list[SampleResult],
        Mapping[str, float],
        Mapping[str, float],
        Mapping[str, object],
        Mapping[str, Mapping[str, float]],
        Mapping[str, Sequence[Mapping[str, object]]],
    ],
]


def run_benchmark(
    plan: BenchmarkPlan,
    evaluator: Evaluator,
    *,
    dataset_checksum: str,
    directions: Mapping[str, str],
    primary_metric: str | Sequence[str],
    required_metrics: Sequence[str] | None = None,
    paired_metrics: Sequence[str] | None = None,
    revisions: Mapping[str, str] | None = None,
    decision_files: Mapping[str, Path] | None = None,
    input_artifacts: Mapping[str, Path] | None = None,
    no_mlflow: bool = False,
    artifact_root: Path = Path("artifacts/benchmarks"),
    monitor_resources: bool = True,
    operational_prefix: str = "operational.",
    paired_comparisons: bool = True,
    candidate_artifact_name: str | None = None,
) -> BenchmarkResult:
    if not plan.candidates:
        raise ValueError("A benchmark plan must contain at least one candidate")
    if len(set(plan.candidates)) != len(plan.candidates):
        raise ValueError("A benchmark plan cannot contain duplicate candidates")
    primary_metrics = (
        (primary_metric,) if isinstance(primary_metric, str) else tuple(primary_metric)
    )
    _validate_metric_contract(directions, primary_metrics)
    required = tuple(required_metrics) if required_metrics is not None else tuple(directions)
    paired = tuple(paired_metrics) if paired_metrics is not None else tuple(directions)
    unknown_required = sorted(set(required) - set(directions))
    if unknown_required:
        raise ValueError("Required metrics have no declared direction: " + ", ".join(unknown_required))
    unknown_paired = sorted(set(paired) - set(directions))
    if unknown_paired:
        raise ValueError("Paired metrics have no declared direction: " + ", ".join(unknown_paired))
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
        "input_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (input_artifacts or {}).items()
        },
    }
    plan_path = directory / "plan.json"
    provenance_path = directory / "provenance.json"
    metric_contract = {
        "primary_metrics": list(primary_metrics),
        "directions": dict(directions),
        "required_metrics": list(required),
        "paired_metrics": list(paired),
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
                "primary_metrics": json.dumps(list(primary_metrics)),
                "required_metrics": json.dumps(list(required)),
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
        for input_name, input_path in (input_artifacts or {}).items():
            tracking.artifact(input_path, f"inputs/{input_name}")
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
                    required,
                    monitor_resources,
                    operational_prefix,
                    candidate_artifact_name,
                )
            )

        successful = [result for result in results if result.status == "success"]
        problems = _completion_problems(results)
        complete = not problems
        summary = {
            "run_id": run_id,
            "fingerprint": run_fingerprint,
            "mlflow_run_id": mlflow_run_id,
            "plan": asdict(plan),
            "metric_contract": metric_contract,
            "provenance": provenance,
            "candidates": [_payload(result, include_samples=False) for result in results],
            "paired_comparisons": (
                []
                if plan.profile == "smoke" or not paired_comparisons
                else _paired_comparisons(
                    successful,
                    {name: directions[name] for name in paired},
                    resamples=plan.bootstrap_resamples,
                    seed=plan.seed,
                )
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
        if not complete:
            tracking.mark_failed()
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
    monitor_resources,
    operational_prefix,
    candidate_artifact_name,
) -> CandidateResult:
    samples: list[SampleResult] = []
    metrics: dict[str, float] = {}
    intervals: dict[str, dict[str, float]] = {}
    operational: dict[str, float] = {}
    artifact_names: list[str] = []
    sample_artifact_path: Path | None = None
    fingerprint = stable_hash({"run": run_fingerprint, "candidate": candidate})
    with tracking.run(candidate, nested=True):
        try:
            tracking.parameters({"candidate": candidate, "profile": plan.profile})
            temporary_directory = directory / "temporary" / _safe(candidate)
            temporary_directory.mkdir(parents=True, exist_ok=True)
            if monitor_resources:
                resources = ResourceMonitor(temporary_directory=temporary_directory)
                try:
                    with _temporary_environment(temporary_directory), resources:
                        evaluated = evaluator(candidate)
                finally:
                    operational.update(resources.metrics())
            else:
                with _temporary_environment(temporary_directory):
                    evaluated = evaluator(candidate)
            samples = list(evaluated[0])
            operational = {**dict(evaluated[1]), **operational}
            candidate_metrics = dict(evaluated[2]) if len(evaluated) >= 3 else {}
            if len(evaluated) >= 4:
                tracking.parameters(evaluated[3])
            candidate_intervals = dict(evaluated[4]) if len(evaluated) >= 5 else {}
            artifact_tables = dict(evaluated[5]) if len(evaluated) >= 6 else {}
            if not samples:
                raise RuntimeError("Candidate produced no samples")
            _validate_sample_ids(samples)
            if artifact_tables:
                if "samples" not in artifact_tables:
                    artifact_tables["samples"] = _sample_rows(samples)
                for name, rows in artifact_tables.items():
                    table_path = _write_table(directory, candidate, name, rows)
                    if name == "samples":
                        sample_artifact_path = table_path
                    artifact_names.append(table_path.name)
                    tracking.artifact(table_path)
            else:
                sample_path = _write_samples(directory, candidate, samples)
                tracking.artifact(sample_path)
                artifact_names.append(sample_path.name)
                sample_artifact_path = sample_path
            shutil.rmtree(temporary_directory, ignore_errors=True)
            # Evaluators with grouped/pooled statistics return their own aggregates and CIs.
            if len(evaluated) >= 5:
                metrics, intervals = candidate_metrics, candidate_intervals
            else:
                metrics, intervals = aggregate_samples(
                    samples,
                    resamples=0 if plan.profile == "smoke" else plan.bootstrap_resamples,
                    seed=plan.seed,
                )
                metrics.update(candidate_metrics)
            _validate_required_metrics(
                metrics, operational, required_metrics, operational_prefix
            )
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
                    **{f"{operational_prefix}{key}": value for key, value in operational.items()},
                }
            )
            tracking.parameters({"candidate_status": "success"})
            candidate_path = _candidate_path(
                directory, candidate, candidate_artifact_name
            )
            artifact_names.append(candidate_path.name)
            atomic_write_json(
                candidate_path,
                {
                    **_payload(result, include_samples=False),
                    "artifacts": artifact_names,
                },
            )
            tracking.artifact(candidate_path)
            return result
        except Exception as exc:
            if "temporary_directory" in locals():
                shutil.rmtree(temporary_directory, ignore_errors=True)
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
                **{f"{operational_prefix}{key}": value for key, value in operational.items()},
            }
            tracking.metrics(
                {
                    key: float(value)
                    for key, value in partial_metrics.items()
                    if _finite_number(value)
                }
            )
            if samples:
                if sample_artifact_path is None:
                    sample_artifact_path = _write_samples(directory, candidate, samples)
                    artifact_names.append(sample_artifact_path.name)
                tracking.artifact(sample_artifact_path)
            candidate_path = _candidate_path(
                directory, candidate, candidate_artifact_name
            )
            if candidate_path.name not in artifact_names:
                artifact_names.append(candidate_path.name)
            atomic_write_json(
                candidate_path,
                {
                    **_payload(result, include_samples=False),
                    "artifacts": artifact_names,
                },
            )
            tracking.artifact(candidate_path)
            tracking.mark_failed()
            return result


def _sample_rows(samples: list[SampleResult]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample.sample_id,
            "latency_seconds": sample.latency_seconds,
            **{f"metric.{key}": value for key, value in sample.metrics.items()},
            "metadata": json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True),
        }
        for sample in samples
    ]


def _write_samples(directory: Path, candidate: str, samples: list[SampleResult]) -> Path:
    path = directory / "samples" / f"{_safe(candidate)}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_sample_rows(samples)).to_parquet(path, index=False)
    return path


def _write_table(
    directory: Path,
    candidate: str,
    name: str,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name):
        raise ValueError(f"Invalid candidate artifact table name: {name}")
    path = directory / "candidates" / _safe(candidate) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)
    return path


def _candidate_path(directory: Path, candidate: str, name: str | None) -> Path:
    if name is None:
        return directory / "candidates" / f"{_safe(candidate)}.json"
    if name != "candidate.json":
        raise ValueError("The supported fixed candidate artifact name is candidate.json")
    return directory / "candidates" / _safe(candidate) / name


def _payload(result: CandidateResult, *, include_samples: bool) -> dict[str, object]:
    payload = asdict(result)
    payload["samples"] = [asdict(sample) for sample in result.samples] if include_samples else []
    return payload


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def _validate_metric_contract(
    directions: Mapping[str, str], primary_metrics: Sequence[str]
) -> None:
    if not directions:
        raise ValueError("A benchmark must declare at least one required metric")
    invalid = sorted(name for name, direction in directions.items() if direction not in {"min", "max"})
    if invalid:
        raise ValueError(f"Metrics have invalid directions: {', '.join(invalid)}")
    if not primary_metrics:
        raise ValueError("A benchmark must declare at least one primary metric")
    missing = [name for name in primary_metrics if name not in directions]
    if missing:
        raise ValueError("Primary metrics are not required metrics: " + ", ".join(missing))


@contextmanager
def _temporary_environment(directory: Path):
    names = ("TMP", "TEMP", "TMPDIR")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = str(directory.resolve())
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_sample_ids(samples: list[SampleResult]) -> None:
    identifiers = [sample.sample_id for sample in samples]
    if any(not identifier for identifier in identifiers):
        raise ValueError("Every sample result must have a non-empty sample_id")
    duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
    if duplicates:
        raise ValueError(f"Candidate produced duplicate sample IDs: {', '.join(duplicates[:10])}")


def _validate_required_metrics(
    metrics, operational, required_metrics, operational_prefix="operational."
) -> None:
    values = {
        **metrics,
        **{f"{operational_prefix}{name}": value for name, value in operational.items()},
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


def _completion_problems(results: list[CandidateResult]) -> list[str]:
    problems: list[str] = []
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
