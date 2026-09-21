"""Typed parent-level comparison artifacts for retrieval/reranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateResult,
)

from .metrics import paired_document_interval
from .profiles import RETRIEVERS, parse_candidate
from .protocol import RetrievalProtocol, protocol_from_settings


RERANKER_OPERATIONAL = frozenset(
    {
        "operational.full_stack_latency_ms_p50",
        "operational.full_stack_latency_ms_p95",
        "operational.cold_initialization_seconds",
        "operational.peak_process_tree_ram_mb",
        "operational.peak_vram_mb",
    }
)
RETRIEVER_OPERATIONAL = RERANKER_OPERATIONAL | {
    "operational.first_stage_latency_ms_p50",
    "operational.first_stage_latency_ms_p95",
    "operational.index_build_seconds",
}
INTEGER_COLUMNS = frozenset(
    {
        "cutoff",
        "eligible_questions",
        "eligible_documents",
        "bootstrap_resamples",
        "seed",
    }
)
FLOAT_COLUMNS = frozenset(
    {
        "baseline_value",
        "candidate_value",
        "candidate_minus_baseline",
        "ci_lower",
        "ci_upper",
        "confidence_level",
    }
)
BOOLEAN_COLUMNS = frozenset({"favors_candidate"})

COMMON_COLUMNS = (
    "comparison_id",
    "baseline_candidate",
    "candidate",
    "baseline_run_id",
    "candidate_run_id",
    "metric_name",
    "evidence_slice",
    "cutoff",
    "direction",
    "baseline_value",
    "candidate_value",
    "candidate_minus_baseline",
    "favors_candidate",
    "ci_lower",
    "ci_upper",
    "confidence_level",
    "ci_status",
    "eligible_questions",
    "eligible_documents",
    "bootstrap_resamples",
    "seed",
)


def parent_artifact_builder(
    pools: Mapping[str, Mapping[str, object]],
    directions: Mapping[str, str],
    *,
    compare_finalists: bool,
):
    """Create a runner callback over the owner pool registry."""

    def build(
        directory: Path,
        results: Sequence[CandidateResult],
        plan: BenchmarkPlan,
    ) -> Sequence[Path]:
        protocol = protocol_from_settings(plan.settings)
        successful = {
            result.candidate: result for result in results if result.status == "success"
        }
        reranker_rows = _reranker_rows(successful, directions, plan, protocol)
        retriever_rows = _retriever_rows(successful, directions, plan, protocol)
        written: list[Path] = []
        identities: dict[str, object] = {}
        for stem, rows, extra_columns in (
            (
                "reranker_comparisons",
                reranker_rows,
                ("retriever", "shared_pool_checksum"),
            ),
            (
                "retriever_comparisons",
                retriever_rows,
                (
                    "baseline_retriever",
                    "candidate_retriever",
                    "baseline_pool_checksum",
                    "candidate_pool_checksum",
                ),
            ),
        ):
            paths, identity = _write_mirrors(
                directory, stem, rows, (*extra_columns, *COMMON_COLUMNS)
            )
            written.extend(paths)
            identities[stem] = identity
        if compare_finalists:
            finalist_rows = _finalist_rows(successful, directions, plan, protocol)
            paths, identity = _write_mirrors(
                directory,
                "finalist_comparisons",
                finalist_rows,
                (
                    "baseline_pool_checksum",
                    "candidate_pool_checksum",
                    *COMMON_COLUMNS,
                ),
            )
            written.extend(paths)
            identities["finalist_comparisons"] = identity

        pool_index_path = directory / "pool_index.json"
        atomic_write_json(
            pool_index_path,
            {
                "schema_version": 1,
                "pools": [
                    {
                        "owner_candidate": owner,
                        "owner_run_id": record.get("owner_run_id"),
                        "pool_checksum": record.get("pool_checksum"),
                        "index_checksum": record.get("index_checksum"),
                        "row_count": len(record.get("rows", [])),
                    }
                    for owner, record in sorted(pools.items())
                ],
                "comparison_artifacts": identities,
            },
        )
        written.append(pool_index_path)
        return written

    return build


def _reranker_rows(
    results: Mapping[str, CandidateResult],
    directions: Mapping[str, str],
    plan: BenchmarkPlan,
    protocol: RetrievalProtocol,
) -> list[dict[str, object]]:
    pairs = []
    for candidate_name, candidate in results.items():
        parsed = parse_candidate(candidate_name)
        if parsed.reranker == "none":
            continue
        baseline = results.get(parsed.owner_identifier)
        if baseline is None:
            continue
        baseline_checksum = str(baseline.parameters.get("pool_checksum", ""))
        candidate_checksum = str(candidate.parameters.get("pool_checksum", ""))
        if not baseline_checksum or baseline_checksum != candidate_checksum:
            raise ValueError(
                f"Cannot compare {candidate_name}: its frozen pool differs from the owner"
            )
        pairs.append(
            (
                baseline,
                candidate,
                {
                    "retriever": parsed.retriever,
                    "shared_pool_checksum": baseline_checksum,
                },
            )
        )
    return _rows_for_pairs(
        pairs,
        _allowed_metrics(directions, RERANKER_OPERATIONAL),
        directions,
        plan,
        protocol,
    )


def _retriever_rows(
    results: Mapping[str, CandidateResult],
    directions: Mapping[str, str],
    plan: BenchmarkPlan,
    protocol: RetrievalProtocol,
) -> list[dict[str, object]]:
    owners = [results.get(f"{retriever}|none") for retriever in RETRIEVERS]
    owners = [result for result in owners if result is not None]
    pairs = []
    for left_index, baseline in enumerate(owners):
        for candidate in owners[left_index + 1 :]:
            pairs.append(
                (
                    baseline,
                    candidate,
                    {
                        "baseline_retriever": parse_candidate(
                            baseline.candidate
                        ).retriever,
                        "candidate_retriever": parse_candidate(
                            candidate.candidate
                        ).retriever,
                        "baseline_pool_checksum": baseline.parameters.get(
                            "pool_checksum"
                        ),
                        "candidate_pool_checksum": candidate.parameters.get(
                            "pool_checksum"
                        ),
                    },
                )
            )
    return _rows_for_pairs(
        pairs,
        _allowed_metrics(
            directions,
            RETRIEVER_OPERATIONAL
            | {
                protocol.pool_recall_metric,
                "storage.required_index_bytes",
            },
        ),
        directions,
        plan,
        protocol,
    )


def _finalist_rows(
    results: Mapping[str, CandidateResult],
    directions: Mapping[str, str],
    plan: BenchmarkPlan,
    protocol: RetrievalProtocol,
) -> list[dict[str, object]]:
    requested = plan.settings.get("requested_finalists", ())
    if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
        raise ValueError("requested_finalists must be a candidate sequence")
    candidates = [results[str(name)] for name in requested if str(name) in results]
    pairs = []
    for left_index, baseline in enumerate(candidates):
        for candidate in candidates[left_index + 1 :]:
            pairs.append(
                (
                    baseline,
                    candidate,
                    {
                        "baseline_pool_checksum": baseline.parameters.get(
                            "pool_checksum"
                        ),
                        "candidate_pool_checksum": candidate.parameters.get(
                            "pool_checksum"
                        ),
                    },
                )
            )
    return _rows_for_pairs(
        pairs,
        _allowed_metrics(directions, RERANKER_OPERATIONAL),
        directions,
        plan,
        protocol,
    )


def _allowed_metrics(
    directions: Mapping[str, str], extras: set[str] | frozenset[str]
) -> set[str]:
    return {
        name
        for name in directions
        if _is_quality_score(name) or _is_retrieved_workload(name) or name in extras
    }


def _rows_for_pairs(
    pairs: Sequence[
        tuple[CandidateResult, CandidateResult, Mapping[str, object]]
    ],
    allowed: set[str],
    directions: Mapping[str, str],
    plan: BenchmarkPlan,
    protocol: RetrievalProtocol,
) -> list[dict[str, object]]:
    rows = []
    for baseline, candidate, identity in pairs:
        baseline_values, candidate_values = _values(baseline), _values(candidate)
        for metric in sorted(allowed & baseline_values.keys() & candidate_values.keys()):
            rows.append(
                {
                    **identity,
                    **_comparison_row(
                        baseline,
                        candidate,
                        metric,
                        directions[metric],
                        plan,
                        protocol,
                        baseline_values[metric],
                        candidate_values[metric],
                    ),
                }
            )
    return rows


def _comparison_row(
    baseline: CandidateResult,
    candidate: CandidateResult,
    metric: str,
    direction: str,
    plan: BenchmarkPlan,
    protocol: RetrievalProtocol,
    baseline_value: float,
    candidate_value: float,
) -> dict[str, object]:
    difference = candidate_value - baseline_value
    evidence_slice, cutoff = _metric_dimensions(metric)
    interval = None
    eligible_questions = None
    eligible_documents = None
    ci_status = "not_supported"
    if _is_question_level_quality(metric):
        interval, eligible_questions, eligible_documents = paired_document_interval(
            baseline.samples,
            candidate.samples,
            metric,
            resamples=plan.bootstrap_resamples,
            seed=plan.seed,
            confidence=protocol.confidence_level,
        )
        if plan.profile == "smoke":
            ci_status = "smoke"
        elif interval is None:
            ci_status = "insufficient_documents"
        else:
            ci_status = "available"
    favors = None
    if direction == "max":
        favors = difference > 0
    elif direction == "min":
        favors = difference < 0
    return {
        "comparison_id": stable_hash(
            {
                "baseline": baseline.candidate,
                "candidate": candidate.candidate,
                "metric": metric,
            }
        )[:16],
        "baseline_candidate": baseline.candidate,
        "candidate": candidate.candidate,
        "baseline_run_id": baseline.mlflow_run_id,
        "candidate_run_id": candidate.mlflow_run_id,
        "metric_name": metric,
        "evidence_slice": evidence_slice,
        "cutoff": cutoff,
        "direction": direction,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "candidate_minus_baseline": difference,
        "favors_candidate": favors,
        "ci_lower": interval["lower"] if interval else None,
        "ci_upper": interval["upper"] if interval else None,
        "confidence_level": interval["confidence"] if interval else None,
        "ci_status": ci_status,
        "eligible_questions": eligible_questions,
        "eligible_documents": eligible_documents,
        "bootstrap_resamples": (
            plan.bootstrap_resamples if interval is not None else None
        ),
        "seed": plan.seed if interval is not None else None,
    }


def _values(result: CandidateResult) -> dict[str, float]:
    values = {
        **{
            name: float(value)
            for name, value in result.metrics.items()
            if value is not None
        },
        **{f"operational.{name}": float(value) for name, value in result.operational.items()},
    }
    if (
        "storage.required_index_bytes" not in values
        and "storage.index_bytes" in values
    ):
        values["storage.required_index_bytes"] = values["storage.index_bytes"]
    return values


def _metric_dimensions(metric: str) -> tuple[str | None, int | None]:
    pool_prefix = "diagnostic.pool_evidence_unit_recall_at_"
    if metric.startswith(pool_prefix):
        return "overall", int(metric.removeprefix(pool_prefix))
    if not metric.startswith("quality."):
        return None, None
    _, evidence_slice, name = metric.split(".", 2)
    cutoff = int(name.rsplit("_at_", 1)[1]) if "_at_" in name else None
    return evidence_slice, cutoff


def _is_quality_score(metric: str) -> bool:
    return metric.startswith("quality.") and not metric.endswith(".sample_count")


def _is_question_level_quality(metric: str) -> bool:
    return _is_quality_score(metric) or metric.startswith(
        "diagnostic.pool_evidence_unit_recall_at_"
    )


def _is_retrieved_workload(metric: str) -> bool:
    return metric.startswith("workload.retrieved_tokens_at_")


def _write_mirrors(
    directory: Path,
    stem: str,
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
) -> tuple[tuple[Path, Path], dict[str, object]]:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    frame = _coerce_schema(frame)
    sort_columns = [
        name
        for name in ("comparison_id", "metric_name", "evidence_slice", "cutoff")
        if name in frame.columns
    ]
    if sort_columns and not frame.empty:
        frame = frame.sort_values(sort_columns, kind="stable", na_position="last")
    frame = frame.reset_index(drop=True)
    parquet_path = directory / f"{stem}.parquet"
    csv_path = directory / f"{stem}.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    parquet = pd.read_parquet(parquet_path)
    csv = pd.read_csv(
        csv_path,
        keep_default_na=True,
        dtype={column: _column_dtype(column) for column in frame.columns},
    )
    try:
        pd.testing.assert_frame_equal(
            parquet, csv, check_exact=False, rtol=1e-12, atol=1e-12
        )
    except AssertionError as exc:
        raise RuntimeError(f"{stem} Parquet/CSV content differs: {exc}") from exc
    logical_rows = [
        {
            name: _json_value(value)
            for name, value in row.items()
        }
        for row in parquet.to_dict(orient="records")
    ]
    return (parquet_path, csv_path), {
        "parquet_sha256": sha256_file(parquet_path),
        "csv_sha256": sha256_file(csv_path),
        "logical_table_sha256": stable_hash(logical_rows),
        "row_count": len(frame),
    }


def _json_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _coerce_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].astype(_column_dtype(column))
    return result


def _column_dtype(column: str) -> str:
    if column in INTEGER_COLUMNS:
        return "Int64"
    if column in FLOAT_COLUMNS:
        return "Float64"
    if column in BOOLEAN_COLUMNS:
        return "boolean"
    return "string"
