"""Shared execution shell for direct benchmark scripts."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
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

from .contracts import (
    BenchmarkPlan,
    BenchmarkResult,
    CandidateExecutionError,
    CandidateResult,
    SampleResult,
)
from .provenance import git_provenance, hardware_summary
from .protocol import ProtocolMetadata
from .resources import ResourceMonitor
from .metrics import paired_bootstrap_interval
from .statistics import aggregate_samples
from .tracking import tracker

Evaluator = Callable[
    ...,
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
        Mapping[str, object],
    ],
]
ResourceMonitorOptions = (
    Mapping[str, object] | Callable[[str], Mapping[str, object]]
)


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
    nullable_metrics: Sequence[str] = (),
    sample_artifact_name: str = "samples",
    resource_artifact_name: str | None = None,
    resource_monitor_options: ResourceMonitorOptions | None = None,
    monitor_temporary_disk: bool = True,
    paired_group_key: str | None = None,
    run_name_prefix: str | None = None,
    shuffle_candidates: bool = True,
    evaluator_receives_context: bool = False,
    parent_artifact_builder: Callable[
        [Path, Sequence[CandidateResult], BenchmarkPlan], Sequence[Path]
    ]
    | None = None,
    operational_maximums: Mapping[str, float] | None = None,
    protocols: Mapping[str, ProtocolMetadata] | None = None,
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
    unknown_nullable = sorted(set(nullable_metrics) - set(required))
    if unknown_nullable:
        raise ValueError(
            "Nullable metrics are not required metrics: " + ", ".join(unknown_nullable)
        )
    run_name = (
        f"{run_name_prefix or f'{plan.suite}-{plan.stage}'}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    directory = artifact_root / plan.suite / plan.stage / run_id
    protocol_values = dict(protocols or {})
    for name, protocol in protocol_values.items():
        if name != protocol.name:
            raise ValueError(
                f"Protocol mapping key {name!r} does not match {protocol.name!r}"
            )
        if not protocol.source_path.is_file():
            raise FileNotFoundError(f"Protocol source is missing: {protocol.source_path}")
    confidence_level = _protocol_confidence_level(protocol_values)
    protocol_provenance = {
        name: {
            "version": protocol.version,
            "checksum": protocol.checksum,
            "source_path": str(protocol.source_path),
            "source_sha256": sha256_file(protocol.source_path),
        }
        for name, protocol in protocol_values.items()
    }
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
        "protocols": protocol_provenance,
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
    metric_contract_path = directory / "metric_contract.json"
    atomic_write_json(metric_contract_path, metric_contract)
    protocol_artifacts: dict[str, Path] = {}
    for name, protocol in protocol_values.items():
        path = directory / f"{name}_protocol.json"
        atomic_write_json(path, protocol.artifact_payload())
        protocol_artifacts[name] = path
    tracking = tracker(disabled=no_mlflow, experiment=f"EduMind / {plan.suite}")
    run_fingerprint = stable_hash({"plan": plan_payload, "provenance": provenance})
    results: list[CandidateResult] = []
    order = list(plan.candidates)
    if shuffle_candidates:
        random.Random(plan.seed).shuffle(order)
    with tracking.run(run_name) as mlflow_run_id:
        tracking.parameters(
            {
                "profile": plan.profile,
                "stage": plan.stage,
                "dataset": plan.dataset,
                "dataset_checksum": dataset_checksum,
                "seed": plan.seed,
                "warmups": plan.warmups,
                "repetitions": plan.repetitions,
                "bootstrap_resamples": plan.bootstrap_resamples,
                "candidates": json.dumps(list(plan.candidates)),
                "settings": json.dumps(plan.settings, sort_keys=True),
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
                **{
                    f"protocol.{name}.version": protocol.version
                    for name, protocol in protocol_values.items()
                },
                **{
                    f"protocol.{name}.checksum": protocol.checksum
                    for name, protocol in protocol_values.items()
                },
            }
        )
        tracking.artifact(plan_path)
        tracking.artifact(provenance_path)
        tracking.artifact(metric_contract_path)
        for name, path in protocol_artifacts.items():
            tracking.artifact(path, "protocols")
            tracking.artifact(protocol_values[name].source_path, f"inputs/protocols/{name}")
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
                    nullable_metrics,
                    sample_artifact_name,
                    resource_artifact_name,
                    resource_monitor_options,
                    monitor_temporary_disk,
                    evaluator_receives_context,
                    operational_maximums,
                    protocol_values,
                    confidence_level,
                )
            )

        successful = [result for result in results if result.status == "success"]
        failed = [result for result in results if result.status == "failed"]
        problems = _completion_problems(results)
        complete = not problems
        paired_comparisons_payload = (
            []
            if plan.profile == "smoke" or not paired_comparisons
            else _paired_comparisons(
                successful,
                {name: directions[name] for name in paired},
                resamples=plan.bootstrap_resamples,
                seed=plan.seed,
                group_key=paired_group_key,
                confidence=confidence_level,
            )
        )
        custom_parent_artifacts: list[Path] = []
        if parent_artifact_builder is not None:
            try:
                custom_parent_artifacts = list(
                    parent_artifact_builder(directory, results, plan)
                )
            except Exception as exc:
                problems.append(
                    "parent artifact validation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                complete = False
        for path in custom_parent_artifacts:
            tracking.artifact(path)
        summary = {
            "run_id": run_id,
            "fingerprint": run_fingerprint,
            "mlflow_run_id": mlflow_run_id,
            "plan": asdict(plan),
            "metric_contract": metric_contract,
            "provenance": provenance,
            "candidates": [_payload(result, include_samples=False) for result in results],
            "paired_comparisons": paired_comparisons_payload,
            "parent_artifacts": [
                {"name": path.name, "sha256": sha256_file(path)}
                for path in custom_parent_artifacts
            ],
            "protocols": {
                name: protocol.artifact_payload()
                for name, protocol in protocol_values.items()
            },
            "complete": complete,
            "completion": {
                "planned_candidates": len(plan.candidates),
                "successful_candidates": len(successful),
                "failed_candidates": len(failed),
                "sample_count": len(successful[0].samples) if successful else 0,
                "problems": problems,
            },
            "selection": {
                "made_by_runner": False,
                "instruction": (
                    "Review the MLflow child runs and artifacts. After a complete non-smoke "
                    "run, record any advancement in a separate engineer-decision JSON file."
                ),
            },
        }
        summary_path = directory / "summary.json"
        leaderboard_path = directory / "leaderboard.parquet"
        pd.DataFrame(
            [
                {
                    "candidate": candidate.candidate,
                    **candidate.metrics,
                    **{
                        f"{operational_prefix}{name}": value
                        for name, value in candidate.operational.items()
                    },
                }
                for candidate in successful
            ]
        ).to_parquet(leaderboard_path, index=False)
        atomic_write_json(summary_path, summary)
        parent_paths = [leaderboard_path, summary_path]
        if paired_comparisons:
            paired_path = directory / "paired_comparisons.json"
            atomic_write_json(paired_path, paired_comparisons_payload)
            parent_paths.insert(1, paired_path)
        for path in parent_paths:
            tracking.artifact(path)
        tracking.metrics(
            {
                "benchmark_complete": float(complete),
                "planned_candidates": float(len(plan.candidates)),
                "successful_candidates": float(len(successful)),
                "failed_candidates": float(len(failed)),
            }
        )
        tracking.tags(
            {
                "benchmark.valid": str(complete).lower(),
                "validation.status": "passed" if complete else "failed",
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
    nullable_metrics,
    sample_artifact_name,
    resource_artifact_name,
    resource_monitor_options,
    monitor_temporary_disk,
    evaluator_receives_context,
    operational_maximums,
    protocols,
    confidence_level,
) -> CandidateResult:
    samples: list[SampleResult] = []
    metrics: dict[str, float | None] = {}
    intervals: dict[str, dict[str, float]] = {}
    operational: dict[str, float] = {}
    candidate_parameters: dict[str, object] = {}
    resource_parameters: dict[str, object] = {}
    artifact_names: list[str] = []
    sample_artifact_path: Path | None = None
    resource_rows: list[dict[str, object]] = []
    fingerprint = stable_hash({"run": run_fingerprint, "candidate": candidate})
    protocol_parameters = (
        {"protocols": {
            name: {
                "version": protocol.version,
                "checksum": protocol.checksum,
                "resolved": protocol.resolved,
            }
            for name, protocol in protocols.items()
        }}
        if protocols
        else {}
    )
    candidate_parameters.update(protocol_parameters)
    with tracking.run(candidate, nested=True) as child_run_id:
        try:
            tracking.parameters(
                {
                    "candidate": candidate,
                    "profile": plan.profile,
                    **{
                        f"protocol.{name}.version": protocol.version
                        for name, protocol in protocols.items()
                    },
                    **{
                        f"protocol.{name}.checksum": protocol.checksum
                        for name, protocol in protocols.items()
                    },
                }
            )
            temporary_directory = directory / "temporary" / _safe(candidate)
            temporary_directory.mkdir(parents=True, exist_ok=True)
            if monitor_resources:
                monitor_options = (
                    resource_monitor_options(candidate)
                    if callable(resource_monitor_options)
                    else resource_monitor_options
                )
                resources = ResourceMonitor(
                    temporary_directory=(
                        temporary_directory if monitor_temporary_disk else None
                    ),
                    **dict(monitor_options or {}),
                )
                evaluation_failed = False
                try:
                    with _temporary_environment(temporary_directory), resources:
                        evaluated = (
                            evaluator(candidate, {"mlflow_run_id": child_run_id})
                            if evaluator_receives_context
                            else evaluator(candidate)
                        )
                except BaseException:
                    evaluation_failed = True
                    raise
                finally:
                    resource_rows = resources.samples()
                    resource_parameters["vram_measurement_method"] = (
                        resources.vram_measurement_method
                    )
                    try:
                        operational.update(resources.metrics())
                    except RuntimeError:
                        if not evaluation_failed:
                            raise
            else:
                with _temporary_environment(temporary_directory):
                    evaluated = (
                        evaluator(candidate, {"mlflow_run_id": child_run_id})
                        if evaluator_receives_context
                        else evaluator(candidate)
                    )
            samples = list(evaluated[0])
            operational = {**dict(evaluated[1]), **operational}
            candidate_metrics = dict(evaluated[2]) if len(evaluated) >= 3 else {}
            if len(evaluated) >= 4:
                candidate_parameters = {
                    **dict(evaluated[3]),
                    **protocol_parameters,
                }
                candidate_parameters.update(resource_parameters)
                tracking.parameters(candidate_parameters)
            candidate_intervals = dict(evaluated[4]) if len(evaluated) >= 5 else {}
            artifact_tables = dict(evaluated[5]) if len(evaluated) >= 6 else {}
            if resource_artifact_name and resource_rows:
                artifact_tables.setdefault(resource_artifact_name, resource_rows)
            if not samples:
                raise RuntimeError("Candidate produced no samples")
            _validate_sample_ids(samples)
            if artifact_tables:
                if sample_artifact_name not in artifact_tables:
                    artifact_tables[sample_artifact_name] = _sample_rows(samples)
                written = _persist_candidate_artifacts(
                    directory, candidate, artifact_tables, tracking
                )
                artifact_names.extend(path.name for path in written.values())
                sample_artifact_path = written.get(sample_artifact_name)
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
                    confidence=confidence_level,
                )
                metrics.update(candidate_metrics)
            _validate_required_metrics(
                metrics,
                operational,
                required_metrics,
                operational_prefix,
                nullable_metrics,
            )
            _validate_operational_maximums(
                operational, operational_maximums or {}
            )
            result = CandidateResult(
                candidate,
                "success",
                fingerprint,
                metrics,
                intervals,
                tuple(samples),
                operational,
                mlflow_run_id=child_run_id,
                parameters=candidate_parameters,
            )
            tracking.metrics(
                {
                    key: value
                    for key, value in {
                    **metrics,
                    **{
                        f"{name}.ci_{bound}": values[bound]
                        for name, values in intervals.items()
                        for bound in ("lower", "upper")
                    },
                    **{f"{operational_prefix}{key}": value for key, value in operational.items()},
                    }.items()
                    if _finite_number(value)
                }
            )
            tracking.parameters({"candidate_status": "success"})
            tracking.tags(
                {"benchmark.valid": "true", "validation.status": "passed"}
            )
            candidate_path = _write_candidate_result(
                directory,
                candidate,
                candidate_artifact_name,
                result,
                artifact_names,
            )
            tracking.artifact(candidate_path)
            return result
        except CandidateExecutionError as exc:
            if "temporary_directory" in locals():
                shutil.rmtree(temporary_directory, ignore_errors=True)
            samples = list(exc.samples)
            metrics = dict(exc.metrics)
            intervals = dict(exc.intervals)
            operational = {**dict(exc.operational), **operational}
            candidate_parameters = {
                **exc.parameters,
                **resource_parameters,
                **protocol_parameters,
            }
            tracking.parameters(
                {
                    **candidate_parameters,
                    "candidate_status": "failed",
                    "candidate_error": str(exc),
                }
            )
            artifact_payloads = dict(exc.artifacts)
            if resource_artifact_name and resource_rows:
                artifact_payloads.setdefault(resource_artifact_name, resource_rows)
            if samples and sample_artifact_name not in artifact_payloads:
                artifact_payloads[sample_artifact_name] = _sample_rows(samples)
            written = _persist_candidate_artifacts(
                directory, candidate, artifact_payloads, tracking
            )
            artifact_names.extend(path.name for path in written.values())
            result = CandidateResult(
                candidate,
                "failed",
                fingerprint,
                metrics,
                intervals,
                tuple(samples),
                operational,
                str(exc),
                child_run_id,
                candidate_parameters,
            )
            partial_metrics = {
                **metrics,
                **{
                    f"{operational_prefix}{key}": value
                    for key, value in operational.items()
                },
            }
            tracking.metrics(
                {
                    key: float(value)
                    for key, value in partial_metrics.items()
                    if _finite_number(value)
                }
            )
            candidate_path = _write_candidate_result(
                directory,
                candidate,
                candidate_artifact_name,
                result,
                artifact_names,
            )
            tracking.artifact(candidate_path)
            tracking.tags(
                {"benchmark.valid": "false", "validation.status": "failed"},
            )
            tracking.mark_failed()
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
                child_run_id,
                candidate_parameters,
            )
            tracking.parameters({"candidate_status": "failed", "candidate_error": error})
            tracking.tags(
                {"benchmark.valid": "false", "validation.status": "failed"}
            )
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
            candidate_path = _write_candidate_result(
                directory,
                candidate,
                candidate_artifact_name,
                result,
                artifact_names,
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


def _write_candidate_artifact(
    directory: Path,
    candidate: str,
    name: str,
    payload: object,
) -> Path:
    if isinstance(payload, Mapping):
        if not name or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in name
        ):
            raise ValueError(f"Invalid candidate artifact name: {name}")
        path = directory / "candidates" / _safe(candidate) / f"{name}.json"
        atomic_write_json(path, payload)
        return path
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise TypeError(f"Candidate artifact {name} must be an object or row sequence")
    return _write_table(directory, candidate, name, payload)


def _persist_candidate_artifacts(
    directory: Path,
    candidate: str,
    payloads: Mapping[str, object],
    tracking,
) -> dict[str, Path]:
    paths = {
        name: _write_candidate_artifact(directory, candidate, name, payload)
        for name, payload in payloads.items()
    }
    for path in paths.values():
        tracking.artifact(path)
    return paths


def _candidate_path(directory: Path, candidate: str, name: str | None) -> Path:
    if name is None:
        return directory / "candidates" / f"{_safe(candidate)}.json"
    if name != "candidate.json":
        raise ValueError("The supported fixed candidate artifact name is candidate.json")
    return directory / "candidates" / _safe(candidate) / name


def _write_candidate_result(
    directory: Path,
    candidate: str,
    name: str | None,
    result: CandidateResult,
    artifact_names: list[str],
) -> Path:
    path = _candidate_path(directory, candidate, name)
    if path.name not in artifact_names:
        artifact_names.append(path.name)
    payload = {
        **_payload(result, include_samples=False),
        "artifacts": artifact_names,
    }
    atomic_write_json(path, payload)
    return path


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
    invalid = sorted(
        name
        for name, direction in directions.items()
        if direction not in {"min", "max", "descriptive", "gate"}
    )
    if invalid:
        raise ValueError(f"Metrics have invalid directions: {', '.join(invalid)}")
    if not primary_metrics:
        raise ValueError("A benchmark must declare at least one primary metric")
    missing = [name for name in primary_metrics if name not in directions]
    if missing:
        raise ValueError("Primary metrics are not required metrics: " + ", ".join(missing))
    invalid_primary = [
        name for name in primary_metrics if directions.get(name) not in {"min", "max"}
    ]
    if invalid_primary:
        raise ValueError(
            "Primary metrics require min/max directions: "
            + ", ".join(invalid_primary)
        )


def _validate_operational_maximums(
    operational: Mapping[str, float], maximums: Mapping[str, float]
) -> None:
    exceeded = [
        f"{name}={operational[name]:.6g}>{maximum:.6g}"
        for name, maximum in maximums.items()
        if name in operational and operational[name] > maximum
    ]
    if exceeded:
        raise ValueError("Operational limit exceeded: " + ", ".join(exceeded))


@contextmanager
def _temporary_environment(directory: Path):
    names = ("TMP", "TEMP", "TMPDIR")
    previous = {name: os.environ.get(name) for name in names}
    previous_tempdir = tempfile.tempdir
    try:
        resolved = str(directory.resolve())
        for name in names:
            os.environ[name] = resolved
        tempfile.tempdir = resolved
        yield
    finally:
        tempfile.tempdir = previous_tempdir
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
    metrics,
    operational,
    required_metrics,
    operational_prefix="operational.",
    nullable_metrics=(),
) -> None:
    values = {
        **metrics,
        **{f"{operational_prefix}{name}": value for name, value in operational.items()},
    }
    missing = [name for name in required_metrics if name not in values]
    nullable = set(nullable_metrics)
    invalid = [
        name
        for name in required_metrics
        if name in values
        and values[name] is not None
        and not _finite_number(values[name])
    ]
    invalid.extend(
        name for name in required_metrics if name in values and values[name] is None and name not in nullable
    )
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
    failed = [result for result in results if result.status == "failed"]
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


def _paired_comparisons(
    results,
    directions,
    *,
    resamples: int,
    seed: int,
    group_key: str | None = None,
    confidence: float,
):
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
                left_values = [
                    left_samples[sample_id].metrics[metric] for sample_id in paired_ids
                ]
                right_values = [
                    right_samples[sample_id].metrics[metric] for sample_id in paired_ids
                ]
                paired_units = len(paired_ids)
                if group_key is not None:
                    grouped_left: dict[str, list[float]] = {}
                    grouped_right: dict[str, list[float]] = {}
                    for sample_id, left_value, right_value in zip(
                        paired_ids, left_values, right_values, strict=True
                    ):
                        left_group = str(
                            left_samples[sample_id].metadata.get(group_key, "")
                        )
                        right_group = str(
                            right_samples[sample_id].metadata.get(group_key, "")
                        )
                        if not left_group or left_group != right_group:
                            raise ValueError(
                                f"Paired metric {metric} has inconsistent {group_key} "
                                f"for sample {sample_id}"
                            )
                        grouped_left.setdefault(left_group, []).append(float(left_value))
                        grouped_right.setdefault(right_group, []).append(float(right_value))
                    groups = sorted(grouped_left)
                    left_values = [
                        sum(grouped_left[group]) / len(grouped_left[group])
                        for group in groups
                    ]
                    right_values = [
                        sum(grouped_right[group]) / len(grouped_right[group])
                        for group in groups
                    ]
                    paired_units = len(groups)
                interval = paired_bootstrap_interval(
                    left_values,
                    right_values,
                    resamples=resamples,
                    seed=seed,
                    confidence=confidence,
                )
                metrics[metric] = {
                    "left_minus_right": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "confidence": interval.confidence,
                    "direction": directions[metric],
                    "paired_samples": len(paired_ids),
                    "paired_resampling_units": paired_units,
                }
            comparisons.append({"left": left.candidate, "right": right.candidate, "metrics": metrics})
    return comparisons


def _protocol_confidence_level(
    protocols: Mapping[str, ProtocolMetadata],
) -> float:
    values = set()
    for protocol in protocols.values():
        statistics = protocol.resolved.get("statistics")
        if not isinstance(statistics, Mapping):
            continue
        value = statistics.get("confidence_level")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{protocol.name} protocol has an invalid statistics.confidence_level"
            )
        confidence = float(value)
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError(
                f"{protocol.name} protocol confidence level must be between zero and one"
            )
        values.add(confidence)
    if len(values) > 1:
        raise ValueError("Composed benchmark protocols disagree on confidence level")
    return next(iter(values), 0.95)
