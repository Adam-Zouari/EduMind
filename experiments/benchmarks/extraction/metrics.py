"""Small text and timestamp helpers shared by audio and video extraction."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence

from experiments.benchmarks.common.metrics import (
    character_error_rate,
    normalized_tokens,
    word_error_rate,
)


def transcript_scores(reference: str, hypothesis: str) -> dict[str, float]:
    """Score speech transcription without document-layout metrics."""

    reference = canonicalize_for_evaluation(reference)
    hypothesis = canonicalize_for_evaluation(hypothesis)
    content = content_scores(reference, hypothesis)
    return {
        "character_error_rate": character_error_rate(reference, hypothesis),
        "word_error_rate": word_error_rate(reference, hypothesis),
        "missing_speech_rate": content["missing_text_rate"],
        "hallucinated_speech_rate": content["hallucinated_text_rate"],
    }


def canonicalize_for_evaluation(text: str) -> str:
    """Standardize encoding and line endings without repairing extracted content."""

    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def content_scores(reference: str, hypothesis: str) -> dict[str, float]:
    reference_tokens = Counter(normalized_tokens(reference))
    hypothesis_tokens = Counter(normalized_tokens(hypothesis))
    overlap = sum((reference_tokens & hypothesis_tokens).values())
    reference_total = sum(reference_tokens.values())
    hypothesis_total = sum(hypothesis_tokens.values())
    precision = overlap / hypothesis_total if hypothesis_total else 0.0
    recall = overlap / reference_total if reference_total else float(not hypothesis_total)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missing = sum((reference_tokens - hypothesis_tokens).values()) / max(1, reference_total)
    hallucinated = sum((hypothesis_tokens - reference_tokens).values()) / max(1, hypothesis_total)
    return {
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
        "missing_text_rate": missing,
        "hallucinated_text_rate": hallucinated,
    }


def timestamp_mae(reference: Sequence[float], hypothesis: Sequence[float]) -> float:
    if len(reference) != len(hypothesis) or not reference:
        raise ValueError("Timestamp arrays must be non-empty and have equal length")
    return sum(abs(left - right) for left, right in zip(reference, hypothesis)) / len(reference)


def duplicate_line_rate(text: str) -> float:
    lines = _blocks(text)
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0


def _blocks(text: str) -> list[str]:
    return [" ".join(line.casefold().split()) for line in text.splitlines() if line.strip()]
