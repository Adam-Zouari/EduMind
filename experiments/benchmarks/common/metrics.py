"""Audited metric formulas shared by direct experiment runners."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

Interval = tuple[int, int]


def interval_overlap(left: Interval, right: Interval) -> int:
    """Length of intersection for half-open intervals; always non-negative."""
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    merged: list[Interval] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered_length(targets: Sequence[Interval], retrieved: Sequence[Interval]) -> int:
    intersections = [
        (max(target[0], candidate[0]), min(target[1], candidate[1]))
        for target in targets
        for candidate in retrieved
        if interval_overlap(target, candidate) > 0
    ]
    return sum(end - start for start, end in merge_intervals(intersections))


def context_recall(gold: Sequence[Interval], retrieved: Sequence[Interval]) -> float:
    denominator = sum(end - start for start, end in merge_intervals(gold))
    return (
        covered_length(merge_intervals(gold), merge_intervals(retrieved)) / denominator
        if denominator
        else 0.0
    )


def relevance_grades(
    gold: Sequence[Interval], retrieved: Sequence[Interval]
) -> list[float]:
    merged_gold = merge_intervals(gold)
    return [
        min(
            1.0,
            covered_length(merged_gold, [candidate])
            / max(1, candidate[1] - candidate[0]),
        )
        for candidate in retrieved
    ]


def precision_at_k(relevance: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(score > 0 for score in relevance[:k]) / k


def recall_at_k(relevance: Sequence[float], relevant_total: int, k: int) -> float:
    return (
        sum(score > 0 for score in relevance[:k]) / relevant_total
        if relevant_total
        else 0.0
    )


def hit_rate_at_k(relevance: Sequence[float], k: int) -> float:
    return float(any(score > 0 for score in relevance[:k]))


def average_precision_at_k(
    relevance: Sequence[float], relevant_total: int, k: int
) -> float:
    if not relevant_total:
        return 0.0
    hits = 0
    accumulated = 0.0
    for rank, score in enumerate(relevance[:k], start=1):
        if score > 0:
            hits += 1
            accumulated += hits / rank
    return accumulated / min(relevant_total, k)


def reciprocal_rank(relevance: Sequence[float]) -> float:
    return next(
        (1.0 / rank for rank, score in enumerate(relevance, start=1) if score > 0), 0.0
    )


def ndcg_at_k(
    grades: Sequence[float], k: int, ideal_grades: Sequence[float] | None = None
) -> float:
    def dcg(values: Sequence[float]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(values, start=1)
        )

    observed = dcg(grades[:k])
    ideal = dcg(
        sorted(ideal_grades if ideal_grades is not None else grades, reverse=True)[:k]
    )
    return observed / ideal if ideal else 0.0


def context_precision_at_k(relevance: Sequence[float], k: int) -> float:
    """Rank-aware context precision (average precision over retrieved relevant contexts)."""
    binary = [score > 0 for score in relevance[:k]]
    relevant_retrieved = sum(binary)
    if not relevant_retrieved:
        return 0.0
    hits = 0
    total = 0.0
    for rank, relevant in enumerate(binary, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    return total / relevant_retrieved


def levenshtein(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    from rapidfuzz.distance import Levenshtein

    return int(Levenshtein.distance(reference, hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> float:
    return levenshtein(list(reference), list(hypothesis)) / max(1, len(reference))


def word_error_rate(reference: str, hypothesis: str) -> float:
    words = reference.split()
    return levenshtein(words, hypothesis.split()) / max(1, len(words))


def normalize_prose(text: str) -> str:
    """Project prose into a symmetric, evaluation-only comparison form."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def normalized_tokens(text: str) -> list[str]:
    return normalize_prose(text).split()


def precision_recall_f1(
    true_positive: float,
    false_positive: float,
    false_negative: float,
) -> tuple[float, float, float]:
    """Return the shared count-based PR/F1 arithmetic without defining units."""

    if min(true_positive, false_positive, false_negative) < 0:
        raise ValueError("Precision/recall counts must be non-negative")
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95


def paired_bootstrap_interval(
    left: Sequence[float],
    right: Sequence[float] | None = None,
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if not left or (right is not None and len(left) != len(right)):
        raise ValueError("Bootstrap inputs must be non-empty paired sequences")
    differences = np.asarray(left, dtype=float)
    if right is not None:
        differences = differences - np.asarray(right, dtype=float)
    random_state = np.random.default_rng(seed)
    indices = random_state.integers(
        0, len(differences), size=(resamples, len(differences))
    )
    samples = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        float(differences.mean()),
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
        confidence,
    )
