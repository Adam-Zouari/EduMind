"""Experiment-only extractor registration over production contracts."""

from __future__ import annotations

from edumind.extraction import SourceKind
from edumind.extraction.extractors.document import DoclingExtractor
from edumind.extraction.registry import ExtractorRegistration, ExtractorRegistry
from experiments.benchmarks.extraction.document.adapters import (
    ExperimentalDocumentExtractor,
)


def build_experiment_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registrations = [
        ExtractorRegistration(
            "docling-standard",
            frozenset({SourceKind.IMAGE, SourceKind.PDF, SourceKind.DOCX}),
            lambda: DoclingExtractor("from-lock"),
        ),
        ExtractorRegistration(
            "docling-vlm-granite-258m",
            frozenset({SourceKind.IMAGE, SourceKind.PDF}),
            lambda: ExperimentalDocumentExtractor(
                "docling-vlm-granite-258m", "from-lock"
            ),
        ),
        ExtractorRegistration(
            "paddleocr-vl-1.6",
            frozenset({SourceKind.IMAGE, SourceKind.PDF}),
            lambda: ExperimentalDocumentExtractor("paddleocr-vl-1.6", "from-lock"),
        ),
    ]
    for registration in registrations:
        registry.register(registration)
    return registry
