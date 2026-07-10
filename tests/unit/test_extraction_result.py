from __future__ import annotations

from datetime import datetime

from edumind.ocr.core.base_extractor import ExtractionResult


def test_to_dict_preserves_rag_ingest_shape() -> None:
    result = ExtractionResult(
        text="Hello world",
        metadata={"title": "Sample", "page": 1},
        format_type="image",
        file_path="C:/tmp/scan.png",
    )

    assert result.to_dict() == {
        "text": "Hello world",
        "title": "Sample",
        "page": 1,
        "format_type": "image",
        "source": "scan.png",
        "success": True,
    }


def test_cache_round_trip_preserves_full_extraction_result() -> None:
    timestamp = datetime(2026, 1, 2, 3, 4, 5)
    original = ExtractionResult(
        text="Extracted text",
        metadata={"confidence": 93.2, "extractor": "ocr"},
        format_type="image",
        file_path="C:/tmp/page.png",
        extraction_time=1.25,
        success=True,
        timestamp=timestamp,
    )

    restored = ExtractionResult.from_cache_dict(original.to_cache_dict())

    assert restored == original


def test_legacy_cache_payload_is_still_readable() -> None:
    restored = ExtractionResult.from_cache_dict(
        {
            "text": "Cached text",
            "title": "Legacy cache",
            "format_type": "image",
            "source": "legacy.png",
            "success": True,
        }
    )

    assert restored.text == "Cached text"
    assert restored.metadata == {"title": "Legacy cache"}
    assert restored.file_path == "legacy.png"
    assert restored.format_type == "image"
