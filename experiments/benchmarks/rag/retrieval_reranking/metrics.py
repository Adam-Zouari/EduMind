"""RR reporting, complete-order repeatability and source-resampled statistics."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from experiments.benchmarks.common.contracts import DatasetManifest, SampleResult
from experiments.benchmarks.common.datasets import answerable_questions, evidence_units
from experiments.benchmarks.rag.chunking_embedding.metrics import covered_unit_ids

from .protocol import RetrievalProtocol

SCOPES = ("overall", "text", "table", "formula", "mixed")
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
            "alpha_ndcg_eligible_question_count",
            "alpha_ndcg_eligible_document_count",
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
    applicable = {
        "overall",
        *(str(q["evidence_type"]) for q in answerable_questions(manifest)),
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
        rows = [
            q for q in questions if scope == "overall" or q["evidence_type"] == scope
        ]
        for family in ("", "alpha_ndcg_"):
            counts[f"validity.quality.{scope}.{family}eligible_question_count"] = float(
                len(rows)
            )
            counts[f"validity.quality.{scope}.{family}eligible_document_count"] = float(
                len({str(q["document_id"]) for q in rows})
            )
    return counts


def source_id(row):
    # Reviewed independent grouping wins over document identity when supplied.
    return str(
        row.get("independent_source_id")
        or row.get("source_group_id")
        or row["document_id"]
    )


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


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _groups(samples):
    groups = {}
    for row in samples:
        groups.setdefault(source_id(row.metadata), []).append(row)
    return list(groups.values())


def _documents(rows):
    documents = {}
    for row in rows:
        documents.setdefault(str(row.metadata["document_id"]), []).append(row)
    return list(documents.values())


def _document_macro(groups, metric):
    # Preserve each drawn source copy, then give its documents equal weight.
    return _mean(
        [
            _mean([row.metrics.get(metric) for row in document])
            for source in groups
            for document in _documents(source)
        ]
    )


def source_interval(
    groups, statistic, *, resamples, seed, confidence, minimum_documents
):
    estimate = statistic(groups)
    support = sum(statistic([group]) is not None for group in groups)
    reason = (
        "no_contributing_documents"
        if estimate is None
        else "smoke"
        if not resamples
        else "ci_support_not_frozen"
        if minimum_documents is None
        else "insufficient_documents"
        if len(groups) < max(2, minimum_documents)
        or support < max(2, minimum_documents)
        else None
    )
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "independent_source_count": len(groups),
        "contributing_source_count": support,
        "eligible_document_count": sum(len(_documents(source)) for source in groups),
        "requested_resamples": resamples,
        "defined_resamples": 0,
        "undefined_resamples": 0,
        "resampling_performed": False,
        "degenerate": False,
        "reason": reason,
    }
    if reason:
        return result
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        value = statistic(
            [groups[int(i)] for i in rng.integers(0, len(groups), len(groups))]
        )
        if value is not None and np.isfinite(value):
            draws.append(value)
    result.update(
        defined_resamples=len(draws),
        undefined_resamples=resamples - len(draws),
        resampling_performed=True,
    )
    if draws:
        lower, upper = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2])
        result.update(
            lower=float(lower), upper=float(upper), degenerate=bool(lower == upper)
        )
    else:
        result["reason"] = "no_defined_resamples"
    return result


def aggregate_quality(
    samples, *, names, resamples, seed, confidence, minimum_documents
):
    values, intervals, support = {}, {}, {}
    for name in names:
        scope = name.split(".")[1] if name.startswith("quality.") else "overall"
        eligible = [
            row
            for row in samples
            if scope == "overall" or row.metadata["evidence_type"] == scope
        ]
        groups = _groups(eligible)
        scored = [row for row in eligible if row.metrics.get(name) is not None]
        statistic = lambda draw, name=name: _document_macro(draw, name)
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
            "eligible_document_count": len(_documents(eligible)),
            "contributing_document_count": len(_documents(scored)),
            "independent_source_count": len(groups),
            "contributing_source_count": len(_groups(scored)),
            "failed_question_count": sum(
                row.metadata.get("first_attempt_status") == "failed" for row in eligible
            ),
        }
        values[name] = estimate if complete else None
        values[name + ".sample_count"] = float(len(_documents(scored)))
        bounds = source_interval(
            groups,
            statistic,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            minimum_documents=minimum_documents,
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


def latency_intervals(samples, *, resamples, seed, minimum_documents, confidence):
    groups = _groups(samples)
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
                minimum_documents=minimum_documents,
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


def paired_document_interval(
    baseline, candidate, metric, *, resamples, seed, confidence, minimum_documents=None
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
        if (
            source_id(a.metadata) != source_id(b.metadata)
            or a.metadata["document_id"] != b.metadata["document_id"]
        ):
            raise ValueError(f"Document mismatch for paired question {identifier}")
        if scope != "overall" and a.metadata["evidence_type"] != scope:
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
    groups = _groups(pairs)

    def statistic(draw):
        if not latency:
            if any(row.metrics[metric] is None for source in draw for row in source):
                return None
            return _document_macro(draw, metric)
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
        minimum_documents=minimum_documents,
    )
    return bounds, len(pairs), len(_documents(pairs)), len(groups)
