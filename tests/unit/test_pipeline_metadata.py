from __future__ import annotations

from pathlib import Path

from edumind.ocr.core.base_extractor import ExtractionResult
from edumind.ocr.core.metadata import build_error_result, finalize_result_metadata
from edumind.ocr.core.types import FormatInfo, PerformanceStats
from edumind.ocr.utils.file_handler import FileHandler


def test_finalize_result_metadata_attaches_format_info_and_cleans_internal_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"content")
    monkeypatch.setattr(FileHandler, "get_file_hash", staticmethod(lambda _: "hash123"))

    result = ExtractionResult(
        text="Hello world",
        metadata={
            "ocr_data": {
                "text": ["Hello"],
                "conf": [95.0],
                "left": [0],
                "top": [0],
                "width": [10],
                "height": [10],
            },
            "image_shape": [10, 10, 3],
            "pages": [
                {
                    "page_index": 0,
                    "ocr_data": {
                        "text": ["A"],
                        "conf": [90.0],
                        "left": [0],
                        "top": [0],
                        "width": [1],
                        "height": [1],
                    },
                    "image_shape": [10, 10, 3],
                }
            ],
        },
        format_type="image",
        file_path=str(image_path),
        success=True,
    )
    performance = PerformanceStats(format_detection=0.1, extraction=0.2, total_processing=0.3)

    finalize_result_metadata(
        result,
        format_info=FormatInfo(format_type="image", mime_type=None, extension=".png"),
        file_path=image_path,
        include_file_hash=True,
        profile=True,
        performance_stats=performance,
    )

    assert result.metadata["format_info"] == {
        "format_type": "image",
        "mime_type": None,
        "extension": ".png",
    }
    assert result.metadata["file_hash"] == "hash123"
    assert "ocr_data" not in result.metadata
    assert "image_shape" not in result.metadata
    assert "ocr_data" not in result.metadata["pages"][0]
    assert "image_shape" not in result.metadata["pages"][0]
    assert result.metadata["performance"]["total_processing"] == 0.3


def test_build_error_result_adds_profile_timing_when_requested(tmp_path: Path) -> None:
    file_path = tmp_path / "failed.pdf"
    result = build_error_result(
        file_path=file_path,
        error="boom",
        total_start=0.0,
        profile=True,
    )

    assert result.success is False
    assert result.error == "boom"
    assert "total_processing" in result.metadata["performance"]
