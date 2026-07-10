"""PDF extraction using PyMuPDF with optional page-level OCR fallback."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import fitz
import numpy as np

from ..core.base_extractor import BaseExtractor, ExtractionResult
from .ocr_extractor import OCRExtractor


class PDFExtractor(BaseExtractor):
    """Extract text and metadata from PDF files with optional OCR fallback."""

    def __init__(self, preserve_layout: bool = True):
        super().__init__()
        self.preserve_layout = preserve_layout

    def extract(self, file_path: Path, **kwargs: Any) -> ExtractionResult:
        """Extract text from PDF files."""
        start_time = time.time()
        self.logger.info(f"Extracting PDF: {file_path}")

        pdf_ocr_mode = str(kwargs.get("pdf_ocr_mode", "auto")).lower()
        if pdf_ocr_mode not in {"off", "auto", "force"}:
            return self._create_error_result(file_path, f"Unsupported pdf_ocr_mode: {pdf_ocr_mode}")

        languages = kwargs.get("languages")
        include_layout = bool(kwargs.get("include_layout", False))

        try:
            doc = fitz.open(file_path)
            native_pages, metadata = self._extract_native_pages(doc)
            total_native_chars = sum(
                sum(1 for character in page["text"] if character.isalnum())
                for page in native_pages
            )

            page_metadata: list[dict[str, Any]] = []
            page_texts: list[str] = []
            page_cache_hits = 0
            page_cache_misses = 0

            for page_index, page in enumerate(doc):
                native_page = native_pages[page_index]
                fallback_reason = self._get_fallback_reason(
                    native_page["text"],
                    total_native_chars,
                    pdf_ocr_mode,
                )
                source = "native"
                page_text = native_page["text"]
                page_confidence: float | None = None
                page_time = float(native_page["native_extraction_time"])
                page_info: dict[str, Any] = {}

                if fallback_reason:
                    ocr_result = self._extract_page_with_ocr(
                        page=page,
                        file_path=file_path,
                        page_index=page_index,
                        languages=languages,
                        include_layout=include_layout,
                    )
                    if ocr_result.success and ocr_result.text.strip():
                        source = "ocr"
                        page_text = ocr_result.text
                        page_confidence = float(ocr_result.metadata.get("confidence", 0.0))
                        page_time = float(ocr_result.extraction_time)
                        page_info = {
                            "confidence": page_confidence,
                            "cache": ocr_result.metadata.get("cache", {}),
                        }
                        if include_layout:
                            if "ocr_data" in ocr_result.metadata:
                                page_info["ocr_data"] = ocr_result.metadata["ocr_data"]
                            if "image_shape" in ocr_result.metadata:
                                page_info["image_shape"] = ocr_result.metadata["image_shape"]
                    else:
                        page_info["ocr_error"] = ocr_result.error or "OCR fallback failed"
                        page_time += float(ocr_result.extraction_time)

                    cache_hit = bool(page_info.get("cache", {}).get("hit"))
                    if cache_hit:
                        page_cache_hits += 1
                    elif source == "ocr":
                        page_cache_misses += 1

                page_texts.append(page_text)
                page_metadata.append(
                    {
                        "page_index": page_index,
                        "source": source,
                        "confidence": page_confidence,
                        "extraction_time": page_time,
                        "fallback_reason": fallback_reason,
                        **page_info,
                    }
                )

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

            extraction_time = time.time() - start_time
            doc.close()

            return ExtractionResult(
                text=combined_text,
                metadata=metadata,
                format_type="pdf",
                file_path=str(file_path),
                extraction_time=extraction_time,
                success=True,
            )
        except Exception as exc:
            self.logger.error(f"PDF extraction failed: {exc}")
            return self._create_error_result(file_path, str(exc))

    def _extract_native_pages(self, doc: fitz.Document) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Extract native text page by page using PyMuPDF."""
        pages = []
        for page_index, page in enumerate(doc):
            page_start = time.time()
            text = page.get_text("text", sort=True) if self.preserve_layout else page.get_text()
            pages.append(
                {
                    "page_index": page_index,
                    "text": text,
                    "native_extraction_time": time.time() - page_start,
                }
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
        return ocr_extractor.extract_image(
            image,
            source_name=f"{file_path}#page-{page_index + 1}",
            cache_key=self._build_page_cache_key(
                file_path=file_path,
                page_index=page_index,
                languages=languages or ocr_extractor.languages,
                use_paddle=ocr_extractor.use_paddle,
                confidence_threshold=ocr_extractor.confidence_threshold,
            ),
            languages=languages,
            return_ocr_data=include_layout,
            use_cache=True,
            format_type="pdf_page",
        )

    def _build_page_cache_key(
        self,
        *,
        file_path: Path,
        page_index: int,
        languages: list[str],
        use_paddle: bool,
        confidence_threshold: float,
    ) -> str:
        """Build a stable OCR cache key for a rendered PDF page."""
        stat = file_path.stat()
        key_str = "|".join(
            [
                str(file_path.resolve()),
                str(stat.st_mtime),
                str(stat.st_size),
                str(page_index),
                ",".join(languages),
                "paddle" if use_paddle else "tesseract",
                str(confidence_threshold),
                "pdf_page_v1",
            ]
        )
        return hashlib.md5(key_str.encode()).hexdigest()
