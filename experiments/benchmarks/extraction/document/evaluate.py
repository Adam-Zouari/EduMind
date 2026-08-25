"""Quality contract for complete document extraction."""

from __future__ import annotations

from collections.abc import Mapping

from experiments.benchmarks.extraction.metrics import (
    base_scores,
    page_scores,
    structured_document_scores,
)


def metrics(reference: str, hypothesis: str, item: Mapping[str, object], document):
    scores = base_scores(reference, hypothesis)
    if item.get("kind") in {"image", "pdf"}:
        scores.update(page_scores(item, document))
    scores.update(structured_document_scores(item.get("reference_elements"), document))
    return scores


def directions() -> dict[str, str]:
    return {
        "character_error_rate": "min",
        "word_error_rate": "min",
        "content_f1": "max",
        "reading_order_accuracy": "max",
        "page_coverage": "max",
        "page_content_f1": "max",
        "page_attribution_accuracy": "max",
        "missing_page_rate": "min",
        "duplicate_page_rate": "min",
        "block_structure_f1": "max",
        "table_detection_f1": "max",
        "table_content_f1": "max",
        "table_structure_f1": "max",
        "formula_detection_f1": "max",
        "formula_latex_similarity": "max",
        "operational.p95_latency_seconds": "min",
        "operational.peak_ram_mb": "min",
    }

