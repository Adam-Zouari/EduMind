"""Evidence-unit retrieval metrics for chunker/embedding selection."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from experiments.benchmarks.common.datasets import EvidenceUnit, evidence_units
from experiments.benchmarks.common.metrics import ndcg_at_k, paired_bootstrap_interval
from experiments.benchmarks.common.contracts import SampleResult


TOP_K = 5
AUDIT_K = 20
ALPHA = 0.5
MIN_LATENCY_CI_DOCUMENTS = 20
QUALITY_METRICS = (
    "alpha_ndcg_at_5",
    "evidence_unit_recall_at_5",
    "evidence_token_precision_at_5",
    "ndcg_at_5",
)


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


def score_question(
    question: Mapping[str, object],
    selected: Sequence[RankedChunk],
    all_chunks: Sequence[RankedChunk],
    tokenizer: EvaluationTokenizer,
) -> QuestionScore:
    """Score one answerable question with the frozen @5/alpha=0.5 contract."""

    units = evidence_units(question)
    if not units:
        raise ValueError("Retrieval quality requires at least one evidence unit")
    selected_coverage = [_covered_unit_ids(chunk, units) for chunk in selected]
    corpus_coverage = [_covered_unit_ids(chunk, units) for chunk in all_chunks]
    first = selected_coverage[:TOP_K]
    recovered = set().union(*first) if first else set()
    binary_grades = [float(bool(covered)) for covered in selected_coverage]
    all_binary_grades = [float(bool(covered)) for covered in corpus_coverage]
    metrics = {
        "alpha_ndcg_at_5": alpha_ndcg_at_k(
            selected_coverage, corpus_coverage, TOP_K, alpha=ALPHA
        ),
        "evidence_unit_recall_at_5": len(recovered) / len(units),
        "evidence_token_precision_at_5": evidence_token_precision_at_k(
            selected, units, tokenizer, TOP_K
        ),
        "ndcg_at_5": ndcg_at_k(binary_grades, TOP_K, all_binary_grades),
    }
    matches = tuple(
        {
            "question_id": str(question["id"]),
            "evidence_unit_id": unit.identifier,
            "evidence_type": unit.evidence_type,
            "recovered_at_5": unit.identifier in recovered,
            "first_rank": next(
                (
                    rank
                    for rank, covered in enumerate(selected_coverage, start=1)
                    if unit.identifier in covered
                ),
                None,
            ),
            "matching_chunk_ids": [
                chunk.identifier
                for chunk, covered in zip(selected, selected_coverage, strict=True)
                if unit.identifier in covered
            ],
        }
        for unit in units
    )
    return QuestionScore(metrics, matches)


def alpha_ndcg_at_k(
    ranking: Sequence[set[str]],
    corpus: Sequence[set[str]],
    k: int,
    *,
    alpha: float = ALPHA,
) -> float:
    """Standard binary-subtopic alpha-nDCG with a deterministic greedy ideal."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")

    def discounted_gain(rows: Sequence[set[str]]) -> float:
        seen: Counter[str] = Counter()
        total = 0.0
        for rank, covered in enumerate(rows[:k], start=1):
            novelty = sum((1.0 - alpha) ** seen[identifier] for identifier in covered)
            total += novelty / math.log2(rank + 1)
            seen.update(covered)
        return total

    remaining = list(enumerate(corpus))
    ideal: list[set[str]] = []
    seen: Counter[str] = Counter()
    for _ in range(min(k, len(remaining))):
        position, (_, covered) = max(
            enumerate(remaining),
            key=lambda item: (
                sum((1.0 - alpha) ** seen[value] for value in item[1][1]),
                -item[1][0],
            ),
        )
        if not covered:
            break
        ideal.append(covered)
        seen.update(covered)
        remaining.pop(position)
    denominator = discounted_gain(ideal)
    return min(1.0, discounted_gain(ranking) / denominator) if denominator else 0.0


def evidence_token_precision_at_k(
    ranking: Sequence[RankedChunk],
    units: Sequence[EvidenceUnit],
    tokenizer: EvaluationTokenizer,
    k: int,
) -> float:
    """Count relevant source-token occurrences among all retrieved token occurrences."""

    evidence = [interval for unit in units for interval in unit.intervals]
    relevant = 0
    total = 0
    for chunk in ranking[:k]:
        for local_start, local_end in tokenizer.spans(chunk.text):
            if local_end <= local_start:
                continue
            total += 1
            start, end = chunk.start + local_start, chunk.start + local_end
            if any(
                interval.document_id == chunk.document_id
                and max(start, interval.start) < min(end, interval.end)
                for interval in evidence
            ):
                relevant += 1
    return relevant / total if total else 0.0


def aggregate_quality(
    samples: Sequence[SampleResult],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Macro-average questions within documents, then documents within the corpus."""

    names = sorted(set().union(*(sample.metrics for sample in samples))) if samples else []
    aggregates: dict[str, float] = {}
    intervals: dict[str, dict[str, float]] = {}
    for name in names:
        by_document: dict[str, list[float]] = {}
        for sample in samples:
            if name not in sample.metrics:
                continue
            document_id = str(sample.metadata["document_id"])
            by_document.setdefault(document_id, []).append(float(sample.metrics[name]))
        document_values = [
            sum(values) / len(values) for _, values in sorted(by_document.items())
        ]
        if not document_values:
            continue
        aggregates[name] = sum(document_values) / len(document_values)
        aggregates[f"{name}.sample_count"] = float(len(document_values))
        if resamples and len(document_values) >= 2:
            interval = paired_bootstrap_interval(
                document_values, resamples=resamples, seed=seed
            )
            intervals[name] = {
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence": interval.confidence,
            }
    return aggregates, intervals


def latency_intervals(
    samples: Sequence[SampleResult],
    *,
    resamples: int,
    seed: int,
    minimum_documents: int = MIN_LATENCY_CI_DOCUMENTS,
) -> dict[str, dict[str, float]]:
    """Cluster-bootstrap warm-query percentiles over source documents."""

    if not resamples:
        return {}
    by_document: dict[str, list[float]] = {}
    for sample in samples:
        by_document.setdefault(str(sample.metadata["document_id"]), []).append(
            sample.latency_seconds * 1000.0
        )
    documents = sorted(by_document)
    if len(documents) < minimum_documents:
        return {}
    rng = np.random.default_rng(seed)
    draws = {0.50: [], 0.95: []}
    for _ in range(resamples):
        selected = rng.integers(0, len(documents), size=len(documents))
        values = [
            latency
            for position in selected
            for latency in by_document[documents[int(position)]]
        ]
        for quantile in draws:
            draws[quantile].append(float(np.quantile(values, quantile)))
    all_values = [sample.latency_seconds * 1000.0 for sample in samples]
    return {
        f"operational.query_latency_ms_p{int(quantile * 100)}": {
            "estimate": float(np.quantile(all_values, quantile)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "confidence": 0.95,
        }
        for quantile, values in draws.items()
    }


def prefixed_quality(
    values: Mapping[str, float], evidence_type: str
) -> dict[str, float]:
    """Attach the overall and mutually exclusive evidence-slice namespaces."""

    return {
        **{f"quality.overall.{name}": value for name, value in values.items()},
        **{f"quality.{evidence_type}.{name}": value for name, value in values.items()},
    }


def eligible_counts(samples: Sequence[SampleResult]) -> dict[str, float]:
    """Expose question/document denominators for overall quality and every slice."""

    result: dict[str, float] = {}
    slices = {"overall", *(str(sample.metadata["evidence_type"]) for sample in samples)}
    for evidence_type in sorted(slices):
        selected = (
            list(samples)
            if evidence_type == "overall"
            else [
                sample
                for sample in samples
                if sample.metadata["evidence_type"] == evidence_type
            ]
        )
        prefix = f"validity.quality.{evidence_type}"
        result[f"{prefix}.eligible_question_count"] = float(len(selected))
        result[f"{prefix}.eligible_document_count"] = float(
            len({str(sample.metadata["document_id"]) for sample in selected})
        )
    return result


def _covered_unit_ids(chunk: RankedChunk, units: Sequence[EvidenceUnit]) -> set[str]:
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
