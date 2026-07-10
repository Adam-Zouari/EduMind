from __future__ import annotations

import json

import pytest

from edumind.rag.ocr_processor import OCRProcessor


def test_normalize_document_merges_metadata_and_filter_scalars() -> None:
    processor = OCRProcessor()

    result = processor.normalize_document(
        {
            "text": "Exam revision notes",
            "source": "lesson.pdf",
            "format_type": "pdf",
            "file_path": "/tmp/lesson.pdf",
            "metadata": {
                "page": 2,
                "topic": "algebra",
                "nested": {"ignored": True},
            },
            "course": "math",
        }
    )

    assert result.source == "lesson.pdf"
    assert result.format_type == "pdf"
    assert result.file_path == "/tmp/lesson.pdf"
    assert result.metadata["page"] == 2
    assert result.metadata["topic"] == "algebra"
    assert result.metadata["course"] == "math"
    assert result.filter_metadata["page"] == 2
    assert result.filter_metadata["course"] == "math"
    assert "nested" not in result.filter_metadata
    assert result.source_id


def test_normalize_document_rejects_blank_text() -> None:
    processor = OCRProcessor()

    with pytest.raises(ValueError, match="Document text is required"):
        processor.normalize_document({"text": "   "})


def test_load_from_json_skips_invalid_documents(tmp_path) -> None:
    processor = OCRProcessor()
    payload = [
        {"text": "First note", "source": "doc-1"},
        {"text": "   ", "source": "doc-2"},
        {"metadata": {"page": 3}},
        {"text": "Second note", "metadata": {"page": 4}},
    ]
    json_path = tmp_path / "ocr.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    documents = processor.load_from_json(json_path)

    assert len(documents) == 2
    assert documents[0].source == "doc-1"
    assert documents[1].source == "ocr.json"
    assert documents[1].metadata["page"] == 4
