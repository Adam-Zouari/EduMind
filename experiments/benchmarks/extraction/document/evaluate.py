"""Metric directions and preflight checks for document extraction."""

from __future__ import annotations

from .metrics import validate_official_evaluators, validate_reference


def directions() -> dict[str, str]:
    return {
        "text.content_precision": "max",
        "text.content_recall": "max",
        "text.content_f1": "max",
        "text.character_error_rate": "min",
        "text.word_error_rate": "min",
        "text.reading_order_accuracy": "max",
        "pages.page_coverage": "max",
        "pages.page_content_f1": "max",
        "pages.page_attribution_accuracy": "max",
        "pages.duplicate_page_rate": "min",
        "layout.element_precision": "max",
        "layout.element_recall": "max",
        "layout.element_f1": "max",
        "layout.element_type_accuracy": "max",
        "layout.hierarchy_accuracy": "max",
        "layout.mean_bounding_box_iou": "max",
        "tables.detection_precision": "max",
        "tables.detection_recall": "max",
        "tables.detection_f1": "max",
        "tables.content_precision": "max",
        "tables.content_recall": "max",
        "tables.content_f1": "max",
        "tables.teds": "max",
        "tables.teds_s": "max",
        "formulas.detection_precision": "max",
        "formulas.detection_recall": "max",
        "formulas.detection_f1": "max",
        "formulas.recognition_similarity": "max",
        "formulas.exact_match": "max",
        "reliability.empty_output_rate": "min",
        "reliability.duplicate_content_rate": "min",
        "reliability.structured_output_determinism": "max",
        "reliability.candidate_failure_rate": "min",
        "operational.first_item_latency_seconds": "min",
        "operational.p50_warm_latency_per_page_seconds": "min",
        "operational.p95_warm_latency_per_page_seconds": "min",
        "operational.p50_complete_document_latency_seconds": "min",
        "operational.p95_complete_document_latency_seconds": "min",
        "operational.batch_pages_per_minute": "max",
        "operational.peak_process_tree_ram_mb": "min",
        "operational.peak_vram_mb": "min",
        "operational.peak_temporary_disk_mb": "min",
    }
