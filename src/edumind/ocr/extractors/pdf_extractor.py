"""PDF extraction using PyMuPDF with optional page-level OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import cast

import fitz
import numpy as np

from ..core.base_extractor import BaseExtractor, ExtractionResult
from ..core.types import CacheStatus, OCRTokenPayload, PdfPageMetadata
from ..utils.cache_keys import build_pdf_page_cache_key
from .ocr_extractor import OCRExtractor


@dataclass(frozen=True)
class NativePdfPage:
    """Native PDF page text plus timing information."""

    page_index: int
    text: str
    native_extraction_time: float


class PDFExtractor(BaseExtractor):
    """Extract text and metadata from PDF files with optional OCR fallback."""

    def __init__(self, preserve_layout: bool = True) -> None:
        super().__init__()
        self.preserve_layout = preserve_layout

    def extract(self, file_path: Path, **kwargs: object) -> ExtractionResult:
        """Extract text from PDF files."""
        start_time = time.time()
        self.logger.info(f"Extracting PDF: {file_path}")

        pdf_ocr_mode = str(kwargs.get("pdf_ocr_mode", "auto")).lower()
        if pdf_ocr_mode not in {"off", "auto", "force"}:
            return self._create_error_result(file_path, f"Unsupported pdf_ocr_mode: {pdf_ocr_mode}")

        languages = kwargs.get("languages")
        resolved_languages = languages if isinstance(languages, list) else None
        include_layout = bool(kwargs.get("include_layout", False))

        try:
            doc = fitz.open(file_path)
            try:
                native_pages, metadata = self._extract_native_pages(doc)
                total_native_chars = sum(
                    sum(1 for character in page.text if character.isalnum())
                    for page in native_pages
                )

                page_metadata: list[PdfPageMetadata] = []
                page_texts: list[str] = []
                page_cache_hits = 0
                page_cache_misses = 0

                for page_index, page in enumerate(doc):
                    native_page = native_pages[page_index]
                    fallback_reason = self._get_fallback_reason(
                        native_page.text,
                        total_native_chars,
                        pdf_ocr_mode,
                    )
                    page_info = self._build_native_page_metadata(
                        page_index=page_index,
                        native_page=native_page,
                        fallback_reason=fallback_reason,
                    )
                    page_text = native_page.text

                    if fallback_reason:
                        ocr_result = self._extract_page_with_ocr(
                            page=page,
                            file_path=file_path,
                            page_index=page_index,
                            languages=resolved_languages,
                            include_layout=include_layout,
                        )
                        (
                            page_text,
                            page_info,
                            page_cache_hits,
                            page_cache_misses,
                        ) = self._merge_ocr_page_result(
                            page_text=page_text,
                            page_info=page_info,
                            ocr_result=ocr_result,
                            include_layout=include_layout,
                            page_cache_hits=page_cache_hits,
                            page_cache_misses=page_cache_misses,
                        )

                    page_texts.append(page_text)
                    page_metadata.append(page_info)

                combined_text = "\n\n".join(page_texts)
                metadata.update(
                    {
                        "extractor": "pymupdf",
                        "pages": page_metadata,
                        "cache": {
                            "page_hits": page_cache_hits,
                            "page_misses": page_cache_misses,
                            "mode": pdf_ocr_mode,
                        },
                    }
                )
                if any(page["source"] == "ocr" for page in page_metadata):
                    metadata["extractor"] = "pymupdf+ocr"

                return ExtractionResult(
                    text=combined_text,
                    metadata=metadata,
                    format_type="pdf",
                    file_path=str(file_path),
                    extraction_time=time.time() - start_time,
                    success=True,
                )
            finally:
                doc.close()
        except Exception as exc:
            self.logger.error(f"PDF extraction failed: {exc}")
            return self._create_error_result(file_path, str(exc))

    def _extract_native_pages(
        self,
        doc: fitz.Document,
    ) -> tuple[list[NativePdfPage], dict[str, object]]:
        """Extract native text page by page using PyMuPDF."""
        pages: list[NativePdfPage] = []
        for page_index, page in enumerate(doc):
            page_start = time.time()
            text = page.get_text("text", sort=True) if self.preserve_layout else page.get_text()
            pages.append(
                NativePdfPage(
                    page_index=page_index,
                    text=text,
                    native_extraction_time=time.time() - page_start,
                )
            )

        metadata = {
            "num_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
        }
        return pages, metadata

    def _build_native_page_metadata(
        self,
        *,
        page_index: int,
        native_page: NativePdfPage,
        fallback_reason: str | None,
    ) -> PdfPageMetadata:
        """Build the starting metadata for a native PDF page."""
        return {
            "page_index": page_index,
            "source": "native",
            "confidence": None,
            "extraction_time": native_page.native_extraction_time,
            "fallback_reason": fallback_reason,
        }

    def _merge_ocr_page_result(
        self,
        *,
        page_text: str,
        page_info: PdfPageMetadata,
        ocr_result: ExtractionResult,
        include_layout: bool,
        page_cache_hits: int,
        page_cache_misses: int,
    ) -> tuple[str, PdfPageMetadata, int, int]:
        """Merge OCR fallback results into a page metadata record."""
        if ocr_result.success and ocr_result.text.strip():
            page_text = ocr_result.text
            page_info["source"] = "ocr"
            page_info["confidence"] = _coerce_float(ocr_result.metadata.get("confidence", 0.0))
            page_info["extraction_time"] = float(ocr_result.extraction_time)
            raw_cache = ocr_result.metadata.get("cache")
            if isinstance(raw_cache, dict):
                page_info["cache"] = cast(CacheStatus, raw_cache)
            if include_layout:
                ocr_data = ocr_result.metadata.get("ocr_data")
                image_shape = ocr_result.metadata.get("image_shape")
                if isinstance(ocr_data, dict):
                    page_info["ocr_data"] = cast(OCRTokenPayload, ocr_data)
                if isinstance(image_shape, list):
                    page_info["image_shape"] = [int(value) for value in image_shape]
        else:
            page_info["ocr_error"] = ocr_result.error or "OCR fallback failed"
            page_info["extraction_time"] = (
                float(page_info["extraction_time"]) + float(ocr_result.extraction_time)
            )

        cache_status = page_info.get("cache")
        cache_hit = bool(cache_status["hit"]) if cache_status is not None else False
        if cache_hit:
            page_cache_hits += 1
        elif page_info["source"] == "ocr":
            page_cache_misses += 1

        return page_text, page_info, page_cache_hits, page_cache_misses

    def _get_fallback_reason(
        self,
        page_text: str,
        total_native_chars: int,
        pdf_ocr_mode: str,
    ) -> str | None:
        """Return the reason an OCR fallback should run for a page, if any."""
        if pdf_ocr_mode == "off":
            return None
        if pdf_ocr_mode == "force":
            return "forced_ocr"

        page_native_chars = sum(1 for character in page_text if character.isalnum())
        if total_native_chars < 150:
            return "document_native_text_below_threshold"
        if page_native_chars < 40:
            return "page_native_text_below_threshold"
        return None

    def _extract_page_with_ocr(
        self,
        *,
        page: fitz.Page,
        file_path: Path,
        page_index: int,
        languages: list[str] | None,
        include_layout: bool,
    ) -> ExtractionResult:
        """Render a PDF page to an image and run image OCR on it."""
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )

        ocr_extractor = OCRExtractor(languages=languages)
        effective_languages = languages or ocr_extractor.languages
        return ocr_extractor.extract_image(
            image,
            source_name=f"{file_path}#page-{page_index + 1}",
            cache_key=build_pdf_page_cache_key(
                file_path=file_path,
                page_index=page_index,
                languages=effective_languages,
                engine_name=ocr_extractor.engine_name,
                confidence_threshold=ocr_extractor.confidence_threshold,
            ),
            languages=effective_languages,
            return_ocr_data=include_layout,
            use_cache=True,
            format_type="pdf_page",
        )


def _coerce_float(value: object) -> float:
    """Safely coerce PDF metadata values to floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
