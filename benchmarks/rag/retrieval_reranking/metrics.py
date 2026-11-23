"""RR reporting, complete-order repeatability and source-resampled statistics."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from benchmarks.common.contracts import DatasetManifest, SampleResult
from benchmarks.common.datasets import (
    answerable_questions,
    evidence_units,
    validate_evidence,
)
from benchmarks.rag.chunking_embedding.metrics import covered_unit_ids
from benchmarks.rag.statistics import (
    QUALITY_SCOPES as SCOPES,
)
from benchmarks.rag.statistics import (
    in_quality_scope,
    question_documents,
    question_macro,
    require_aligned_questions,
    source_groups,
    source_ids,
    source_interval,
    source_scope,
)
from benchmarks.rag.statistics import (
    question_mean as _mean,
)

from .protocol import RetrievalProtocol

REPEATABILITY = "diagnostic.ranking_repeatability"
OPERATIONAL_DIRECTIONS = {
    **{
        f"operational.{family}_latency_ms_p{p}": "min"
        for family in ("full_stack", "retrieval", "reranker")
        for p in (50, 95)
    },
    **{
        f"operational.{name}": "min"
        for name in (
            "cold_initialization_seconds",
            "peak_process_tree_ram_mb",
            "peak_vram_mb",
            "index_build_seconds",
        )
    },
}
STORAGE_DIRECTIONS = {
    f"storage.{name}": "descriptive"
    for name in ("index_size_bytes", "dense_index_size_bytes", "bm25_index_size_bytes")
}
VALIDITY_DIRECTIONS = {
    **{
        f"validity.{name}": "descriptive"
        for name in (
            "expected_query_count",
            "processed_query_count",
            "failed_query_count",
        )
    },
    **{
        f"validity.{name}": "gate"
        for name in (
            "truncated_input_count",
            "nonfinite_score_count",
            "exact_pool_permutation",
        )
    },
}


def directions_for(manifest: DatasetManifest, protocol: RetrievalProtocol):
    quality = {
        f"quality.{scope}.{name}": "max"
        for scope in SCOPES
        for name in protocol.quality_metrics
    }
    eligibility = {
        f"validity.quality.{scope}.{suffix}": "descriptive"
        for scope in SCOPES
        for suffix in (
            "eligible_question_count",
            "eligible_document_count",
        )
    }
    sampled = (
        *quality,
        protocol.pool_recall_metric,
        REPEATABILITY,
        *(name for name in OPERATIONAL_DIRECTIONS if "_latency_ms_" in name),
    )
    workload = {
        **{
            f"workload.retrieved_context_tokens_at_{k}_{suffix}": "descriptive"
            for k in protocol.quality_cutoffs
            for suffix in ("mean", "p95")
        },
        **{
            f"workload.reranker_input_tokens_per_query_{suffix}": "descriptive"
            for suffix in ("mean", "p95")
        },
    }
    directions = {
        **quality,
        protocol.pool_recall_metric: "max",
        REPEATABILITY: "max",
        **{name + ".sample_count": "descriptive" for name in sampled},
        **eligibility,
        **VALIDITY_DIRECTIONS,
        **OPERATIONAL_DIRECTIONS,
        **STORAGE_DIRECTIONS,
        **workload,
    }
    sources = validate_evidence(manifest)
    questions = answerable_questions(manifest)
    applicable = {
        "overall",
        *(q["evidence_type"] for q in questions),
        *(source_scope(sources[q["id"]].metadata()) for q in questions),
    }
    required = (
        *(name for name in quality if name.split(".")[1] in applicable),
        *eligibility,
        *VALIDITY_DIRECTIONS,
        *(
            name
            for name in OPERATIONAL_DIRECTIONS
            if "reranker" not in name and name != "operational.index_build_seconds"
        ),
        *(name for name in workload if "retrieved_context" in name),
    )
    return directions, required


def primary_metrics(protocol):
    return tuple(f"quality.overall.{name}" for name in protocol.primary_quality_metrics)


def reference_counts(questions):
    counts = {}
    for scope in SCOPES:
        rows = [q for q in questions if in_quality_scope(q, scope)]
        counts[f"validity.quality.{scope}.eligible_question_count"] = float(len(rows))
        counts[f"validity.quality.{scope}.eligible_document_count"] = float(
            len(question_documents(rows))
        )
    return counts


def pool_evidence_unit_recall(question, chunks):
    units = evidence_units(question)
    if not units:
        raise ValueError("Pool recall requires reviewed gold units")
    return len(
        set().union(*(covered_unit_ids(chunk, units) for chunk in chunks))
    ) / len(units)


def retrieved_tokens(chunks, cutoff, tokenizer=None):
    counts = [
        len(tokenizer.spans(chunk.text)) if tokenizer is not None else int(chunk.tokens)
        for chunk in chunks[:cutoff]
    ]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("A valid retrieval must contain source tokens")
    return sum(counts)


def repeatability(attempts, repetitions):
    """Missing/unexecuted is not a disagreement; executed failures are."""
    scheduled = repetitions * (repetitions - 1) // 2
    by_id = {int(row["repetition"]): row for row in attempts}
    complete = (
        len(attempts) == len(by_id) == repetitions
        and set(by_id) == set(range(1, repetitions + 1))
        and all(row.get("status") in {"success", "failed"} for row in attempts)
    )
    pairs = []
    for left, right in combinations(range(1, repetitions + 1), 2):
        a, b = by_id.get(left), by_id.get(right)
        known = all(
            row is not None and row.get("status") in {"success", "failed"}
            for row in (a, b)
        )
        match = (
            bool(
                a.get("valid")
                and b.get("valid")
                and a.get("ordered_chunk_ids") == b.get("ordered_chunk_ids")
            )
            if known
            else None
        )
        pairs.append({"left_attempt": left, "right_attempt": right, "matches": match})
    matches = sum(row["matches"] is True for row in pairs)
    return {
        "value": matches / scheduled if scheduled and complete else None,
        "status": "inapplicable"
        if not scheduled
        else "scored"
        if complete
        else "incomplete",
        "reason": "repeatability_not_measured"
        if not scheduled
        else None
        if complete
        else "missing_or_unexecuted_attempts",
        "scheduled_pair_count": scheduled,
        "matching_pair_count": matches,
        "complete": complete,
        "pairs": pairs,
    }


def _document_count(rows):
    return len(question_documents(row.metadata for row in rows))


def aggregate_quality(samples, *, names, resamples, seed, confidence, minimum_sources):
    values, intervals, support = {}, {}, {}
    for name in names:
        scope = name.split(".")[1] if name.startswith("quality.") else "overall"
        eligible = [row for row in samples if in_quality_scope(row.metadata, scope)]
        groups = source_groups(eligible)
        scored = [row for row in eligible if row.metrics.get(name) is not None]

        def statistic(draw, name=name):
            return question_macro(draw, name)

        estimate = statistic(groups)
        complete = len(scored) == len(eligible) and bool(eligible)
        status = (
            "inapplicable"
            if not eligible
            else "unavailable"
            if not scored
            else "scored"
            if complete
            else "incomplete"
        )
        details = {
            "status": status,
            "complete": complete,
            "estimate": estimate,
            "eligible_question_count": len(eligible),
            "scored_question_count": len(scored),
            "unavailable_question_count": len(eligible) - len(scored),
            "eligible_document_count": _document_count(eligible),
            "contributing_document_count": _document_count(scored),
            "independent_source_count": len(
                source_ids(row.metadata for row in eligible)
            ),
            "contributing_source_count": len(
                source_ids(row.metadata for row in scored)
            ),
            "failed_question_count": sum(
                row.metadata.get("first_attempt_status") == "failed" for row in eligible
            ),
        }
        values[name] = estimate if complete else None
        values[name + ".sample_count"] = float(len(scored))
        bounds = source_interval(
            groups,
            statistic,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            minimum_sources=minimum_sources,
        )
        if not complete and eligible:
            bounds.update(lower=None, upper=None, reason="incomplete_required_evidence")
        details["interval"] = bounds
        intervals[name] = bounds
        support[name] = details
    return values, intervals, support


def latency_summary(samples):
    result = {}
    for family in ("full_stack", "retrieval", "reranker"):
        values = [_latency(row, family) for row in samples]
        values = [value for value in values if value is not None]
        if values:
            for p in (50, 95):
                result[f"{family}_latency_ms_p{p}"] = float(
                    np.quantile(values, p / 100)
                )
    return result


def _latency(row, family):
    return (
        (row.latency_seconds * 1000 if row.latency_seconds is not None else None)
        if family == "full_stack"
        else row.metadata.get(f"{family}_latency_ms")
    )


def latency_intervals(samples, *, resamples, seed, minimum_sources, confidence):
    groups = source_groups(samples)
    result = {}
    for family in ("full_stack", "retrieval", "reranker"):
        if family == "reranker" and not any(
            row.metadata.get("reranker_applicable") for row in samples
        ):
            continue
        for p in (50, 95):

            def statistic(draw, family=family, p=p):
                values = [_latency(row, family) for group in draw for row in group]
                values = [v for v in values if v is not None]
                return float(np.quantile(values, p / 100)) if values else None

            result[f"operational.{family}_latency_ms_p{p}"] = source_interval(
                groups,
                statistic,
                resamples=resamples,
                seed=seed,
                confidence=confidence,
                minimum_sources=minimum_sources,
            )
    return result


def descriptor(values, expected):
    known = [value for value in values if value is not None]
    return {
        "status": "scored"
        if len(known) == expected and expected
        else "inapplicable"
        if not expected
        else "unavailable"
        if not known
        else "incomplete",
        "complete": bool(expected and len(known) == expected),
        "eligible_query_count": expected,
        "contributing_query_count": len(known),
        "unavailable_query_count": expected - len(known),
        "mean": _mean(known),
        "p95": float(np.quantile(known, 0.95)) if known else None,
    }


def paired_question_interval(
    baseline, candidate, metric, *, resamples, seed, confidence, minimum_sources=None
):
    left = {row.sample_id: row for row in baseline}
    right = {row.sample_id: row for row in candidate}
    if left.keys() != right.keys():
        raise ValueError("Paired candidates have different question inventories")
    scope = metric.split(".")[1] if metric.startswith("quality.") else "overall"
    latency = metric.startswith("operational.") and "_latency_ms_p" in metric
    family = metric.removeprefix("operational.").split("_latency_ms_p")[0]
    percentile = 0.95 if metric.endswith("p95") else 0.5
    pairs = []
    for identifier, a in left.items():
        b = right[identifier]
        require_aligned_questions(a, b)
        if not in_quality_scope(a.metadata, scope):
            continue
        pairs.append(
            SampleResult(
                identifier,
                {"baseline": _latency(a, family), "candidate": _latency(b, family)}
                if latency
                else {
                    metric: b.metrics[metric] - a.metrics[metric]
                    if a.metrics.get(metric) is not None
                    and b.metrics.get(metric) is not None
                    else None
                },
                None,
                a.metadata,
            )
        )
    groups = source_groups(pairs)

    def statistic(draw):
        if not latency:
            if any(row.metrics[metric] is None for source in draw for row in source):
                return None
            return question_macro(draw, metric)
        values = {
            side: [
                row.metrics[side]
                for source in draw
                for row in source
                if row.metrics[side] is not None
            ]
            for side in ("baseline", "candidate")
        }
        return (
            float(
                np.quantile(values["candidate"], percentile)
                - np.quantile(values["baseline"], percentile)
            )
            if all(values.values())
            else None
        )

    bounds = source_interval(
        groups,
        statistic,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        minimum_sources=minimum_sources,
    )
    return (
        bounds,
        len(pairs),
        _document_count(pairs),
        bounds["independent_source_count"],
    )
