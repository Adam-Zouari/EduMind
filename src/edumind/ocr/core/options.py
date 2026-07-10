"""Typed option models for OCR pipeline requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..config import PRESERVE_LATEX

ValidPdfMode = Literal["off", "auto", "force"]
ValidBatchStrategy = Literal["auto", "threads", "sequential"]

_VALID_PDF_MODES = {"off", "auto", "force"}
_VALID_BATCH_STRATEGIES = {"auto", "threads", "sequential"}


@dataclass(frozen=True)
class ProcessFileOptions:
    """Normalized options for `DataIngestionPipeline.process_file()`."""

    clean_text: bool = True
    preserve_latex: bool = PRESERVE_LATEX
    pdf_ocr_mode: ValidPdfMode = "auto"
    include_layout: bool = False
    include_form_fields: bool = False
    profile: bool = False
    include_file_hash: bool = True
    languages: list[str] | None = None
    strict_format_detection: bool = False
    extra_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pdf_ocr_mode not in _VALID_PDF_MODES:
            raise ValueError(f"Unsupported pdf_ocr_mode: {self.pdf_ocr_mode}")

    def build_extract_kwargs(self, format_type: str) -> dict[str, object]:
        """Build extractor kwargs while preserving current public behavior."""
        extract_kwargs = dict(self.extra_kwargs)
        if self.languages is not None:
            extract_kwargs["languages"] = list(self.languages)
        if format_type == "pdf":
            extract_kwargs["pdf_ocr_mode"] = self.pdf_ocr_mode
            extract_kwargs["include_layout"] = self.include_layout
        if format_type == "image" and self.include_layout:
            extract_kwargs["return_ocr_data"] = True
        return extract_kwargs


@dataclass(frozen=True)
class BatchProcessingOptions:
    """Normalized options for `DataIngestionPipeline.process_batch()`."""

    parallel: bool = True
    max_workers: int = 4
    batch_strategy: ValidBatchStrategy = "auto"
    process_options: ProcessFileOptions = field(default_factory=ProcessFileOptions)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.batch_strategy not in _VALID_BATCH_STRATEGIES:
            raise ValueError(f"Unsupported batch_strategy: {self.batch_strategy}")

    @property
    def effective_strategy(self) -> ValidBatchStrategy:
        """Return the resolved strategy after applying `parallel=False`."""
        if not self.parallel:
            return "sequential"
        return self.batch_strategy
