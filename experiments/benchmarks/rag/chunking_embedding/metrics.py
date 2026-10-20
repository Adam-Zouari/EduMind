"""Evidence scoring and document-resampled statistics, outside model execution."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np

from experiments.benchmarks.common.contracts import SampleResult
from experiments.benchmarks.common.datasets import EvidenceUnit, evidence_units


class RankedChunk(Protocol):
    identifier: str
    document_id: str
    text: str
    start: int
    end: int


class EvaluationTokenizer(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]: ...


@dataclass(frozen=True)
class QuestionScore:
    metrics: Mapping[str, float]
    matches: tuple[Mapping[str, object], ...]
    diagnostics: Mapping[str, object]


def score_question(
    question,
    selected: Sequence[RankedChunk],
    all_chunks: Sequence[RankedChunk],
    tokenizer: EvaluationTokenizer,
    *,
    cutoffs,
    alpha,
):
    units = evidence_units(question)
    if not units or not cutoffs or any(k <= 0 for k in cutoffs):
        raise ValueError("Quality scoring requires evidence and positive cutoffs")
    coverage = [covered_unit_ids(chunk, units) for chunk in selected]
    corpus = [covered_unit_ids(chunk, units) for chunk in all_chunks]
    metrics, diagnostics = {}, {}
    for k in cutoffs:
        ideal = sorted((len(covered) for covered in corpus), reverse=True)[:k]
        denominator = _dcg(ideal)
        metrics[f"ndcg_at_{k}"] = (
            _dcg([len(row) for row in coverage[:k]]) / denominator
            if denominator
            else 0.0
        )
        metrics[f"evidence_unit_recall_at_{k}"] = len(set().union(*coverage[:k])) / len(
            units
        )
        metrics[f"evidence_token_precision_at_{k}"] = evidence_token_precision_at_k(
            selected, units, tokenizer, k
        )
        raw = _alpha_ratio(coverage, corpus, k, alpha=alpha)
        metrics[f"alpha_ndcg_at_{k}"] = min(1.0, raw)
        diagnostics[f"alpha_ndcg_at_{k}.raw_ratio"] = raw
        diagnostics[f"alpha_ndcg_at_{k}.greedy_baseline_exceeded"] = raw > 1.0
    for name, value in metrics.items():
        if not math.isfinite(value) or not -1e-12 <= value <= 1 + 1e-12:
            raise ValueError(f"Scoring integrity failure: {name}={value}")
        metrics[name] = min(1.0, max(0.0, value))
    matches = tuple(
        {
            "evidence_unit_id": unit.identifier,
            "evidence_type": unit.evidence_type,
            **{
                f"recovered_at_{k}": any(unit.identifier in row for row in coverage[:k])
                for k in cutoffs
            },
            "first_rank": next(
                (
                    rank
                    for rank, row in enumerate(coverage, 1)
                    if unit.identifier in row
                ),
                None,
            ),
            "matching_chunk_ids": [
                chunk.identifier
                for chunk, row in zip(selected, coverage, strict=True)
                if unit.identifier in row
            ],
        }
        for unit in units
    )
    return QuestionScore(metrics, matches, diagnostics)


def _dcg(gains):
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def _alpha_ratio(ranking, corpus, k, *, alpha):
    if k <= 0 or not 0 <= alpha < 1:
        raise ValueError("alpha-nDCG requires k > 0 and 0 <= alpha < 1")

    def gain(covered, seen):
        return sum((1 - alpha) ** seen[unit] for unit in sorted(covered))

    def discounted(rows):
        seen = Counter()
        result = 0.0
        for rank, covered in enumerate(rows[:k], 1):
            result += gain(covered, seen) / math.log2(rank + 1)
            seen.update(covered)
        return result

    remaining = list(enumerate(corpus))
    seen, ideal = Counter(), []
    for _ in range(min(k, len(remaining))):
        position = max(
            range(len(remaining)),
            key=lambda i: (gain(remaining[i][1], seen), -remaining[i][0]),
        )
        _, covered = remaining.pop(position)
        ideal.append(covered)
        seen.update(covered)
    denominator = discounted(ideal)
    return discounted(ranking) / denominator if denominator else 0.0


def evidence_token_precision_at_k(selected, units, tokenizer, k):
    intervals = [interval for unit in units for interval in unit.intervals]
    total = relevant = 0
    for chunk in selected[:k]:
        spans = tokenizer.spans(chunk.text)
        if not spans:
            raise ValueError("Retrieval contains a token-free chunk")
        for start, end in spans:
            if not 0 <= start < end <= len(chunk.text):
                raise ValueError("Invalid evaluation-token source span")
            total += 1
            relevant += any(
                interval.document_id == chunk.document_id
                and chunk.start + start < interval.end
                and chunk.start + end > interval.start
                for interval in intervals
            )
    if not total:
        raise ValueError("A valid retrieval must contain source tokens")
    return relevant / total


def covered_unit_ids(chunk: RankedChunk, units: Sequence[EvidenceUnit]) -> set[str]:
    return {
        unit.identifier
        for unit in units
        if all(
            interval.document_id == chunk.document_id
            and chunk.start <= interval.start
            and chunk.end >= interval.end
            for interval in unit.intervals
        )
    }


def prefixed_quality(values, evidence_type):
    return {
        f"quality.{scope}.{name}": value
        for scope in ("overall", evidence_type)
        for name, value in values.items()
    }


def eligible_counts(samples: Sequence[SampleResult]):
    # Eligibility is identical for all four metric families.
    result = {}
    for scope in ("overall", "text", "table", "formula", "mixed"):
        rows = _scope(samples, scope)
        prefix = f"validity.quality.{scope}"
        result[f"{prefix}.eligible_question_count"] = float(len(rows))
        result[f"{prefix}.eligible_document_count"] = float(len(_groups(rows)))
    return result


def _scope(samples, scope):
    return [
        row
        for row in samples
        if scope == "overall" or row.metadata["evidence_type"] == scope
    ]


def _groups(samples):
    groups = {}
    for row in samples:
        groups.setdefault(str(row.metadata["document_id"]), []).append(row)
    return list(groups.values())


def source_groups(samples):
    """Reviewed independent groups containing whole documents, never group-macro weights."""
    sources = {}
    for document in _groups(samples):
        ids = {
            str(
                row.metadata.get("independent_source_id")
                or row.metadata.get("source_group_id")
                or row.metadata["document_id"]
            )
            for row in document
        }
        if len(ids) != 1:
            raise ValueError("A document has contradictory independent source groups")
        sources.setdefault(ids.pop(), []).append(document)
    return list(sources.values())


def _document_macro(groups, metric):
    return _mean(
        [
            _mean([row.metrics.get(metric) for row in document])
            for source in groups
            for document in source
        ]
    )


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _interval(groups, statistic, *, resamples, seed, minimum_documents, confidence):
    estimate = statistic(groups)
    support = sum(statistic([group]) is not None for group in groups)
    reason = (
        "no_contributing_documents"
        if estimate is None
        else "ci_support_not_frozen"
        if minimum_documents is None
        else "insufficient_independent_sources"
        if support < max(2, minimum_documents)
        else None
    )
    bounds = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "independent_unit": "reviewed_source_group",
        "independent_source_count": len(groups),
        "contributing_source_count": support,
        "eligible_document_count": sum(len(source) for source in groups),
        "contributing_document_count": sum(
            statistic([[document]]) is not None
            for source in groups
            for document in source
        ),
        "requested_resamples": resamples,
        "defined_resamples": 0,
        "undefined_resamples": 0,
        "resampling_performed": False,
        "degenerate": False,
        "reason": reason,
    }
    if reason:
        return bounds
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        # Copies remain distinct; never regroup a draw by original document ID.
        value = statistic(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        if value is not None:
            draws.append(value)
    bounds.update(
        defined_resamples=len(draws),
        undefined_resamples=resamples - len(draws),
        resampling_performed=True,
    )
    if draws:
        lower, upper = np.quantile(draws, [(1 - confidence) / 2, (1 + confidence) / 2])
        bounds.update(
            lower=float(lower), upper=float(upper), degenerate=bool(lower == upper)
        )
    else:
        bounds["reason"] = "no_defined_resamples"
    return bounds


def aggregate_quality(
    samples: Sequence[SampleResult],
    *,
    resamples,
    seed,
    confidence,
    minimum_documents=2,
    diagnostics=None,
    metric_names=None,
):
    aggregates, intervals = {}, {}
    names = sorted(
        metric_names
        if metric_names is not None
        else {name for row in samples for name in row.metrics}
    )
    for name in names:
        eligible = _scope(samples, name.split(".", 2)[1])
        groups = _groups(eligible)
        sources = source_groups(eligible)
        scored = [row for row in eligible if row.metrics.get(name) is not None]
        failed = sum(
            row.metadata.get("first_attempt_status") == "failed" for row in eligible
        )
        unexecuted = [
            row
            for row in eligible
            if row.metadata.get("first_attempt_status")
            in {"unexecuted", "not_executed"}
        ]
        status = (
            "inapplicable"
            if not eligible
            else "incomplete"
            if unexecuted
            else "unavailable"
            if not scored
            else "partial"
            if len(scored) < len(eligible)
            else "complete"
        )
        document_values = [
            _mean([row.metrics.get(name) for row in group]) for group in groups
        ]
        estimate = _mean(document_values)
        details = {
            "status": status,
            "estimate": estimate,
            "sample_unit": "source_document_macro_value",
            "independent_source_count": len(sources),
            "contributing_source_count": len(source_groups(scored)),
            "eligible_question_count": len(eligible),
            "scheduled_question_count": len(eligible),
            "scored_question_count": len(scored),
            "failed_question_count": failed,
            "unavailable_question_count": len(eligible) - len(scored),
            "unexecuted_question_count": len(unexecuted),
            "eligible_document_count": len(groups),
            "scored_document_count": len(_groups(scored)),
            "failed_document_count": len(
                _groups(
                    [
                        row
                        for row in eligible
                        if row.metadata.get("first_attempt_status") == "failed"
                    ]
                )
            ),
            "unavailable_document_count": sum(
                any(row.metrics.get(name) is None for row in group) for group in groups
            ),
            "unexecuted_document_count": len(_groups(unexecuted)),
        }
        aggregates[name] = estimate if status == "complete" else None
        aggregates[name + ".sample_count"] = float(len(_groups(scored)))
        if resamples:
            bounds = _interval(
                sources,
                lambda draw, name=name: _document_macro(draw, name),
                resamples=resamples,
                seed=seed,
                minimum_documents=minimum_documents,
                confidence=confidence,
            )
            details["interval"] = bounds
            if status == "complete":
                intervals[name] = bounds
        if diagnostics is not None:
            diagnostics[name] = details
    return aggregates, intervals


def ranking_repeatability(attempts, *, depth):
    """All scheduled pairs; failures disagree and missing execution is unknown."""
    pairs = []
    for left, right in combinations(attempts, 2):
        executed = all(
            row["status"] == "failed"
            or row["status"] == "success"
            and row.get("ranking_ids") is not None
            for row in (left, right)
        )
        valid = all(
            row["status"] == "success" and row.get("ranking_ids") is not None
            for row in (left, right)
        )
        pairs.append(
            {
                "left_attempt": left["repetition"],
                "right_attempt": right["repetition"],
                "agreement": int(
                    valid
                    and left["ranking_ids"][:depth] == right["ranking_ids"][:depth]
                )
                if executed
                else None,
            }
        )
    complete = all(pair["agreement"] is not None for pair in pairs)
    successes = sum(pair["agreement"] or 0 for pair in pairs)
    return {
        "value": successes / len(pairs) if pairs and complete else None,
        "status": "inapplicable"
        if not pairs
        else "scored"
        if complete
        else "incomplete",
        "reason": "repeatability_not_measured"
        if not pairs
        else None
        if complete
        else "missing_required_execution",
        "scheduled_pair_count": len(pairs),
        "successful_pair_count": successes,
        "pairs": pairs,
    }


def aggregate_repeatability(samples, *, resamples, seed, minimum_documents, confidence):
    name = "diagnostic.ranking_repeatability"
    groups = _groups(samples)
    sources = source_groups(samples)
    scored = [row for row in samples if row.metrics.get(name) is not None]
    inapplicable = all(
        row.metadata["repeatability_status"] == "inapplicable" for row in samples
    )
    status = (
        "inapplicable"
        if inapplicable
        else "complete"
        if len(scored) == len(samples)
        else "incomplete"
    )
    values = [_mean([row.metrics.get(name) for row in group]) for group in groups]
    details = {
        "status": status,
        "estimate": _mean(values),
        "sample_unit": "source_document_macro_value",
        "eligible_question_count": 0 if inapplicable else len(samples),
        "scored_question_count": len(scored),
        "unavailable_question_count": 0 if inapplicable else len(samples) - len(scored),
        "eligible_document_count": len(groups),
        "independent_source_count": len(sources),
        "contributing_source_count": len(source_groups(scored)),
        "contributing_document_count": len(_groups(scored)),
        "scheduled_pair_count": sum(
            row.metadata.get("scheduled_pair_count", 0) for row in samples
        ),
        "successful_pair_count": sum(
            row.metadata.get("successful_pair_count", 0) for row in samples
        ),
    }
    intervals = {}
    if resamples and status == "complete":
        intervals[name] = _interval(
            sources,
            lambda draw: _document_macro(draw, name),
            resamples=resamples,
            seed=seed,
            minimum_documents=minimum_documents,
            confidence=confidence,
        )
        details["interval"] = intervals[name]
    return (
        {
            name: details["estimate"] if status == "complete" else None,
            name + ".sample_count": float(len(_groups(scored))),
        },
        intervals,
        details,
    )


def latency_intervals(
    samples,
    *,
    resamples,
    seed,
    minimum_documents,
    confidence,
    metric="query_latency_ms",
):
    if not resamples:
        return {}
    groups = source_groups(samples)
    result = {}
    for percentile, label in ((50, "p50"), (95, "p95")):

        def statistic(groups, percentile=percentile):
            values = [
                row.latency_seconds * 1000
                if metric == "query_latency_ms"
                else row.metadata[metric]
                for group in groups
                for document in group
                for row in document
                if (
                    row.latency_seconds
                    if metric == "query_latency_ms"
                    else row.metadata.get(metric)
                )
                is not None
            ]
            return float(np.percentile(values, percentile)) if values else None

        result[f"operational.{metric}_{label}"] = _interval(
            groups,
            statistic,
            resamples=resamples,
            seed=seed,
            minimum_documents=minimum_documents,
            confidence=confidence,
        )
    return result


def paired_quality(results, metrics, *, resamples, seed, minimum_documents, confidence):
    """Aligned source draws; document-macro differences, not group-macro differences."""
    comparisons = []
    for left, right in combinations(
        [row for row in results if row.status == "success"], 2
    ):
        left_rows = {row.sample_id: row for row in left.samples}
        right_rows = {row.sample_id: row for row in right.samples}
        if left_rows.keys() != right_rows.keys():
            raise ValueError("Paired candidates have different question inventories")
        differences = []
        for identifier in sorted(left_rows):
            a, b = left_rows[identifier], right_rows[identifier]
            for field in ("document_id", "independent_source_id"):
                if a.metadata.get(field) != b.metadata.get(field):
                    raise ValueError(f"Paired candidates have inconsistent {field}")
            if any(
                a.metrics.get(name) is None or b.metrics.get(name) is None
                for name in metrics
            ):
                raise ValueError(
                    "Paired selection requires complete quality observations"
                )
            differences.append(
                SampleResult(
                    identifier,
                    {name: a.metrics[name] - b.metrics[name] for name in metrics},
                    None,
                    a.metadata,
                )
            )
        sources = source_groups(differences)
        values = {}
        for name in metrics:
            interval = _interval(
                sources,
                lambda draw, name=name: _document_macro(draw, name),
                resamples=resamples,
                seed=seed,
                minimum_documents=minimum_documents,
                confidence=confidence,
            )
            values[name] = {
                **interval,
                "left_minus_right": interval["estimate"],
                "direction": "max",
                "paired_samples": len(differences),
                "paired_documents": sum(len(source) for source in sources),
                "paired_resampling_units": len(sources),
            }
        comparisons.append(
            {"left": left.candidate, "right": right.candidate, "metrics": values}
        )
    return comparisons
