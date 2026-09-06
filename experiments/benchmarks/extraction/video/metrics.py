"""Text helpers used only by the video keyframe experiment."""

from __future__ import annotations

from collections import Counter
from experiments.benchmarks.common.metrics import normalize_prose, normalized_tokens


def content_scores(reference: str, hypothesis: str) -> dict[str, float]:
    reference_tokens = Counter(normalized_tokens(reference))
    hypothesis_tokens = Counter(normalized_tokens(hypothesis))
    overlap = sum((reference_tokens & hypothesis_tokens).values())
    reference_total = sum(reference_tokens.values())
    hypothesis_total = sum(hypothesis_tokens.values())
    precision = overlap / hypothesis_total if hypothesis_total else 0.0
    recall = overlap / reference_total if reference_total else float(not hypothesis_total)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"content_precision": precision, "content_recall": recall, "content_f1": f1}


def duplicate_line_rate(text: str) -> float:
    lines = [normalize_prose(line) for line in text.splitlines() if normalize_prose(line)]
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0
