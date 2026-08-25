"""Quality contract for text normalization."""

from __future__ import annotations

from collections.abc import Mapping

from experiments.benchmarks.common.metrics import levenshtein
from experiments.benchmarks.extraction.metrics import base_scores


def metrics(reference: str, hypothesis: str, item: Mapping[str, object], document):
    del document
    scores = base_scores(reference, hypothesis)
    observed = str(item["observed"])
    before_distance = levenshtein(list(reference), list(observed))
    after_distance = levenshtein(list(reference), list(hypothesis))
    changes = levenshtein(list(observed), list(hypothesis))
    corrected = max(0, before_distance - after_distance)
    scores["content_preservation_recall"] = scores["content_recall"]
    scores["corruption_removal_recall"] = (
        corrected / before_distance if before_distance else 1.0
    )
    scores["corruption_removal_precision"] = (
        corrected / changes if changes else float(not before_distance)
    )
    precision = scores["corruption_removal_precision"]
    recall = scores["corruption_removal_recall"]
    scores["corruption_removal_f1"] = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return scores


def directions() -> dict[str, str]:
    return {
        "character_error_rate": "min",
        "word_error_rate": "min",
        "content_f1": "max",
        "reading_order_accuracy": "max",
        "content_preservation_recall": "max",
        "corruption_removal_f1": "max",
        "operational.p95_latency_seconds": "min",
        "operational.peak_ram_mb": "min",
    }

