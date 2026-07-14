from __future__ import annotations

from dataclasses import replace

import pytest

from edumind.extraction import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    ExtractionRequest,
    SourceKind,
)
from edumind.extraction.cache import ExtractionCache
from edumind.extraction.detection import classify_source
from edumind.extraction.errors import UnsupportedSourceError
from edumind.extraction.normalization import normalize_document, normalize_text
from edumind.extraction.registry import ExtractorRegistration, ExtractorRegistry


def _document(path, profile=None) -> ExtractedDocument:
    profile = profile or ExtractionProfile("test", "fake", "1", normalization="minimal")
    return ExtractedDocument(
        path.name,
        str(path),
        SourceKind.IMAGE,
        "abc",
        "image/png",
        "alpha beta",
        (ExtractedSegment("alpha beta", 0, 10, page_number=1),),
        profile,
    )


def test_segments_enforce_half_open_offsets(tmp_path) -> None:
    with pytest.raises(ValueError, match="half-open"):
        ExtractedSegment("abc", 0, 2)
    path = tmp_path / "a.png"
    path.write_bytes(b"data")
    roundtrip = ExtractedDocument.from_dict(_document(path).to_dict())
    assert roundtrip.segments[0].start == 0
    assert roundtrip.text[0:10] == "alpha beta"


def test_normalization_rebuilds_offsets(tmp_path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"data")
    raw_text = "chap-\nter\n\n\ntext"
    document = replace(
        _document(path),
        text=raw_text,
        segments=(ExtractedSegment(raw_text, 0, len(raw_text), page_number=1),),
    )
    normalized = normalize_document(document, "conservative")
    assert normalized.text == "chapter\n\ntext"
    assert normalized.segments[0].end == len(normalized.text)
    assert normalize_text("a\r\nb", "minimal") == "a\nb"


def test_source_classification_and_unsupported(tmp_path) -> None:
    assert classify_source(tmp_path / "study.PDF")[0] is SourceKind.PDF
    assert classify_source(tmp_path / "lecture.mp4")[0] is SourceKind.VIDEO
    with pytest.raises(UnsupportedSourceError):
        classify_source(tmp_path / "page.html")


def test_cache_key_invalidates_all_output_affecting_options(tmp_path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"same")
    first_profile = ExtractionProfile("test", "fake", "1", options={"threshold": 1})
    first = ExtractionRequest.from_path(source, profile=first_profile, options={"page": 1})
    cache = ExtractionCache(tmp_path / "cache")
    cache.put(first, _document(source, first_profile))
    assert cache.get(first).cache_hit
    changed_revision = replace(first, profile=replace(first_profile, engine_revision="2"))
    changed_options = replace(first, options={"page": 2})
    assert cache.get(changed_revision) is None
    assert cache.get(changed_options) is None


def test_registry_lazily_reuses_one_engine_instance() -> None:
    registry = ExtractorRegistry()
    created: list[object] = []

    def factory():
        instance = object()
        created.append(instance)
        return instance

    registry.register(
        ExtractorRegistration("fake", frozenset({SourceKind.IMAGE}), factory, "", "test")
    )
    assert registry.create("fake", SourceKind.IMAGE) is registry.create("fake", SourceKind.IMAGE)
    assert len(created) == 1
