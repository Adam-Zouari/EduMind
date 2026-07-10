from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from edumind.ocr.core.base_extractor import ExtractionResult
from edumind.ocr.extractors._image_backends import ImageOCRBackends, OCRRunResult
from edumind.ocr.extractors.pdf_extractor import NativePdfPage, PDFExtractor


def _create_mixed_pdf(path: Path) -> None:
    doc = fitz.open()
    page_one = doc.new_page()
    page_one.insert_text(
        (72, 72),
        (
            "This page has enough native text to avoid OCR fallback and keeps the "
            "document total comfortably above the auto OCR threshold for the first page. "
            "It includes an additional sentence about algebra, calculus, and revision "
            "strategy so the native character count is well above one hundred and fifty."
        ),
    )
    doc.new_page()
    doc.save(path)
    doc.close()


def test_pdf_auto_fallback_ocrs_only_low_text_pages(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    _create_mixed_pdf(pdf_path)
    extractor = PDFExtractor()
    monkeypatch.setattr(
        extractor,
        "_extract_native_pages",
        lambda doc: (
            [
                NativePdfPage(
                    page_index=0,
                    text=" ".join(["Long native study text"] * 20),
                    native_extraction_time=0.05,
                ),
                NativePdfPage(
                    page_index=1,
                    text="",
                    native_extraction_time=0.01,
                ),
            ],
            {
                "num_pages": 2,
                "title": "",
                "author": "",
                "subject": "",
                "creator": "",
                "producer": "",
                "creation_date": "",
            },
        ),
    )

    monkeypatch.setattr(
        extractor,
        "_extract_page_with_ocr",
        lambda **kwargs: ExtractionResult(
            text="Recovered OCR page text",
            metadata={"confidence": 88.0, "cache": {"hit": True}},
            format_type="pdf_page",
            success=True,
            extraction_time=0.2,
        ),
    )

    result = extractor.extract(pdf_path, pdf_ocr_mode="auto")

    assert result.success is True
    assert result.metadata["pages"][0]["source"] == "native"
    assert result.metadata["pages"][1]["source"] == "ocr"
    assert result.metadata["pages"][1]["fallback_reason"] == "page_native_text_below_threshold"
    assert result.metadata["cache"]["page_hits"] == 1
    assert "Recovered OCR page text" in result.text


def test_pdf_force_mode_ocrs_all_pages(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "force.pdf"
    _create_mixed_pdf(pdf_path)
    extractor = PDFExtractor()
    calls: list[int] = []

    def _fake_page_ocr(**kwargs):
        calls.append(kwargs["page_index"])
        return ExtractionResult(
            text=f"OCR page {kwargs['page_index']}",
            metadata={"confidence": 90.0, "cache": {"hit": False}},
            format_type="pdf_page",
            success=True,
            extraction_time=0.1,
        )

    monkeypatch.setattr(extractor, "_extract_page_with_ocr", _fake_page_ocr)

    result = extractor.extract(pdf_path, pdf_ocr_mode="force")

    assert calls == [0, 1]
    assert all(page["source"] == "ocr" for page in result.metadata["pages"])
    assert result.metadata["cache"]["page_misses"] == 2


def test_pdf_page_cache_is_reused_on_second_run(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "cached.pdf"
    _create_mixed_pdf(pdf_path)

    def _fake_extract(self, image: np.ndarray, **kwargs) -> OCRRunResult:
        return OCRRunResult(
            text="Cached OCR text",
            confidence=95.0,
            attempts=["standard (conf: 95.00)"],
            ocr_data=None,
        )

    monkeypatch.setattr(ImageOCRBackends, "extract", _fake_extract)

    extractor = PDFExtractor()
    first = extractor.extract(pdf_path, pdf_ocr_mode="force")
    second = extractor.extract(pdf_path, pdf_ocr_mode="force")

    assert first.metadata["cache"]["page_misses"] == 2
    assert second.metadata["cache"]["page_hits"] == 2
