from __future__ import annotations

from pathlib import Path

from edumind.ocr.utils.cache_keys import build_image_cache_key, build_pdf_page_cache_key


def test_build_image_cache_key_is_stable_for_the_same_file(tmp_path: Path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"image-bytes")

    assert build_image_cache_key(image_path) == build_image_cache_key(image_path)


def test_build_pdf_page_cache_key_changes_with_page_and_engine(tmp_path: Path) -> None:
    pdf_path = tmp_path / "study.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    page_zero = build_pdf_page_cache_key(
        file_path=pdf_path,
        page_index=0,
        languages=["eng"],
        engine_name="tesseract",
        confidence_threshold=60.0,
    )
    page_one = build_pdf_page_cache_key(
        file_path=pdf_path,
        page_index=1,
        languages=["eng"],
        engine_name="tesseract",
        confidence_threshold=60.0,
    )
    paddle = build_pdf_page_cache_key(
        file_path=pdf_path,
        page_index=0,
        languages=["eng"],
        engine_name="paddleocr",
        confidence_threshold=60.0,
    )

    assert page_zero != page_one
    assert page_zero != paddle
