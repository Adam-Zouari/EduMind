"""OCR package with lazy exports."""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["DataIngestionPipeline", "ExtractionResult"]


def __getattr__(name: str):
    if name == "DataIngestionPipeline":
        from .core.pipeline import DataIngestionPipeline

        return DataIngestionPipeline
    if name == "ExtractionResult":
        from .core.base_extractor import ExtractionResult

        return ExtractionResult
    raise AttributeError(name)
