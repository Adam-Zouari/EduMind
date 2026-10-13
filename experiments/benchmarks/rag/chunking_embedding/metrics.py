"""Evidence scoring and document-resampled statistics, outside model execution."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        else "insufficient_independent_documents"
        if support < max(2, minimum_documents)
        else None
    )
    bounds = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "independent_document_count": len(groups),
        "contributing_document_count": support,
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
        scored = [row for row in eligible if row.metrics.get(name) is not None]
        failed = sum(
            row.metadata.get("first_attempt_status") == "failed" for row in eligible
        )
        status = (
            "inapplicable"
            if not eligible
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
            "eligible_question_count": len(eligible),
            "scored_question_count": len(scored),
            "failed_question_count": failed,
            "unavailable_question_count": len(eligible) - len(scored),
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
        }
        aggregates[name] = estimate if status == "complete" else None
        aggregates[name + ".sample_count"] = float(len(_groups(scored)))
        if resamples:
            bounds = _interval(
                document_values,
                _mean,
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


def latency_intervals(samples, *, resamples, seed, minimum_documents, confidence):
    if not resamples:
        return {}
    groups = _groups(samples)
    result = {}
    for percentile, label in ((50, "p50"), (95, "p95")):

        def statistic(groups, percentile=percentile):
            values = [
                row.latency_seconds * 1000
                for group in groups
                for row in group
                if row.latency_seconds is not None
            ]
            return float(np.percentile(values, percentile)) if values else None

        result[f"operational.query_latency_ms_{label}"] = _interval(
            groups,
            statistic,
            resamples=resamples,
            seed=seed,
            minimum_documents=minimum_documents,
            confidence=confidence,
        )
    return result
