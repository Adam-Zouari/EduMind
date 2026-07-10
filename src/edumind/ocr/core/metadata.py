"""Shared metadata and post-processing helpers for the OCR pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from ..processors.form_recognizer import FormRecognizer
from ..processors.layout_analyzer import LayoutAnalyzer
from ..processors.math_extractor import MathExtractor
from ..processors.text_cleaner import TextCleaner
from ..utils.file_handler import FileHandler
from .base_extractor import ExtractionResult
from .types import FormatInfo, LayoutBlockPayload, OCRTokenPayload, PerformanceStats


def apply_text_post_processing(
    result: ExtractionResult,
    *,
    clean_text: bool,
    preserve_latex: bool,
    text_cleaner: TextCleaner,
    math_extractor: MathExtractor,
    performance_stats: PerformanceStats,
) -> None:
    """Clean extracted text while preserving the current math behavior."""
    if not (result.success and clean_text and result.text):
        return

    clean_start = time.perf_counter()
    if preserve_latex:
        preserved_text, math_dict = math_extractor.preserve_math(result.text)
        cleaned = text_cleaner.clean(preserved_text, preserve_latex=True)
        result.text = math_extractor.restore_math(cleaned, math_dict)
    else:
        result.text = text_cleaner.clean(result.text, preserve_latex=False)

    result.metadata["math_expressions"] = math_extractor.extract_latex(result.text)
    performance_stats.cleaning = time.perf_counter() - clean_start


def attach_optional_metadata(
    result: ExtractionResult,
    *,
    format_type: str,
    include_layout: bool,
    include_form_fields: bool,
    layout_analyzer: LayoutAnalyzer,
    form_recognizer: FormRecognizer,
) -> None:
    """Attach optional structured OCR metadata without rewriting text output."""
    if include_form_fields:
        fields = form_recognizer.extract_form_fields(result.text)
        result.metadata["structured_fields"] = form_recognizer.to_structured_dict(fields)

    if not include_layout:
        return

    layout_blocks: list[LayoutBlockPayload] = []
    if format_type == "image":
        layout_blocks.extend(
            _build_layout_blocks_from_metadata(
                result.metadata,
                layout_analyzer=layout_analyzer,
            )
        )
    elif format_type == "pdf":
        pages = result.metadata.get("pages")
        if isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                layout_blocks.extend(
                    _build_layout_blocks_from_metadata(
                        page,
                        layout_analyzer=layout_analyzer,
                        page_index=int(page.get("page_index", 0)),
                    )
                )

    result.metadata["layout_blocks"] = layout_blocks


def finalize_result_metadata(
    result: ExtractionResult,
    *,
    format_info: FormatInfo,
    file_path: Path,
    include_file_hash: bool,
    profile: bool,
    performance_stats: PerformanceStats,
) -> None:
    """Attach shared metadata and remove internal-only payloads."""
    result.metadata["format_info"] = format_info.to_metadata_dict()
    result.metadata["file_size"] = FileHandler.get_file_size(file_path)

    if include_file_hash:
        hash_start = time.perf_counter()
        result.metadata["file_hash"] = FileHandler.get_file_hash(file_path)
        performance_stats.hashing = time.perf_counter() - hash_start

    _cleanup_internal_metadata(result)
    if profile:
        result.metadata["performance"] = performance_stats.to_metadata_dict()


def build_error_result(
    *,
    file_path: Path,
    error: str,
    total_start: float,
    profile: bool,
) -> ExtractionResult:
    """Build a standardized pipeline error result."""
    result = ExtractionResult(
        text="",
        file_path=str(file_path),
        success=False,
        error=error,
    )
    if profile:
        result.metadata["performance"] = {
            "total_processing": time.perf_counter() - total_start,
        }
    return result


def _build_layout_blocks_from_metadata(
    metadata: dict[str, object],
    *,
    layout_analyzer: LayoutAnalyzer,
    page_index: int | None = None,
) -> list[LayoutBlockPayload]:
    """Convert token-level OCR data into serialized layout blocks."""
    ocr_data = metadata.get("ocr_data")
    image_shape = metadata.get("image_shape")
    if not isinstance(ocr_data, dict) or not isinstance(image_shape, list) or not image_shape:
        return []

    typed_ocr_data: OCRTokenPayload = {
        "text": [str(value) for value in ocr_data.get("text", [])],
        "conf": [float(value) for value in ocr_data.get("conf", [])],
        "left": [int(value) for value in ocr_data.get("left", [])],
        "top": [int(value) for value in ocr_data.get("top", [])],
        "width": [int(value) for value in ocr_data.get("width", [])],
        "height": [int(value) for value in ocr_data.get("height", [])],
    }
    image = np.zeros(tuple(int(value) for value in image_shape), dtype=np.uint8)
    blocks = layout_analyzer.analyze_layout(image, typed_ocr_data)
    serialized: list[LayoutBlockPayload] = []
    for block in blocks:
        payload = cast(LayoutBlockPayload, asdict(block))
        if page_index is not None:
            payload["page_index"] = page_index
        serialized.append(payload)
    return serialized


def _cleanup_internal_metadata(result: ExtractionResult) -> None:
    """Remove internal OCR token payloads that are only needed for post-processing."""
    result.metadata.pop("ocr_data", None)
    result.metadata.pop("image_shape", None)
    pages = result.metadata.get("pages")
    if not isinstance(pages, list):
        return

    for page in pages:
        if not isinstance(page, dict):
            continue
        page.pop("ocr_data", None)
        page.pop("image_shape", None)
