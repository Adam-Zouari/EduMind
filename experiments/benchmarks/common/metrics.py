"""Audited metric formulas shared by direct experiment runners."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / f1_denominator if f1_denominator else 0.0
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
    group_sizes: Sequence[int] | None = None,
) -> ConfidenceInterval:
    """Resample paired means, weighting each drawn group by its member count."""
    if not left or (right is not None and len(left) != len(right)):
        raise ValueError("Bootstrap inputs must be non-empty paired sequences")
    differences = np.asarray(left, dtype=float)
    if right is not None:
        differences = differences - np.asarray(right, dtype=float)
    random_state = np.random.default_rng(seed)
    indices = random_state.integers(
        0, len(differences), size=(resamples, len(differences))
    )
    if group_sizes is None:
        estimate = differences.mean()
        samples = differences[indices].mean(axis=1)
    else:
        sizes = np.asarray(group_sizes)
        if sizes.shape != differences.shape or np.any(sizes <= 0):
            raise ValueError("Positive group sizes must align with the paired means")
        estimate = np.average(differences, weights=sizes)
        samples = (differences * sizes)[indices].sum(axis=1) / sizes[indices].sum(
            axis=1
        )
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        float(estimate),
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
        confidence,
    )
