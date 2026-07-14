from __future__ import annotations

import pytest

from edumind.benchmarks.contracts import DatasetManifest
from edumind.benchmarks.extraction import _routing_engine, run_extraction_stage
from edumind.extraction import ExtractedDocument, ExtractedSegment, SourceKind


class FakeProductionExtraction:
    def __init__(self, settings):
        self.settings = settings

    def extract(self, path, *, source_kind, profile, use_cache):
        assert source_kind is SourceKind.IMAGE and use_cache is False
        return ExtractedDocument(
            "fixture.png",
            str(path),
            SourceKind.IMAGE,
            "checksum",
            "image/png",
            "reference text",
            (ExtractedSegment("reference text", 0, 14, page_number=1),),
            profile,
        )


def test_standard_extraction_runner_uses_production_profile(monkeypatch, tmp_path) -> None:
    manifest = DatasetManifest(
        "fixture",
        "1",
        "extraction",
        "validation",
        "local",
        "CC0",
        "1",
        "checksum",
        "v1",
        42,
        (
            {
                "id": "image-1",
                "kind": "image",
                "reference": "reference text",
                "source_path": "fixture.png",
                "engine_revision": "pinned",
                "preprocessing": "raw",
                "normalization": "conservative",
            },
        ),
    )
    monkeypatch.setattr("edumind.benchmarks.extraction.load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        "edumind.benchmarks.extraction.load_extraction_model_lock",
        lambda path: {
            "paddleocr-v5-mobile": {"revision": "pinned"},
            "paddleocr-v5-server": {"revision": "pinned"},
            "doctr-fast-parseq": {"revision": "pinned"},
        },
    )
    monkeypatch.setattr(
        "edumind.benchmarks.extraction.ExtractionPipeline", FakeProductionExtraction
    )
    result = run_extraction_stage("image", "standard", artifact_root=tmp_path)
    assert all(candidate.status == "success" for candidate in result.candidates)
    assert all(
        candidate.samples[0].metrics["character_error_rate"] == 0 for candidate in result.candidates
    )


def test_routing_engine_policy_and_invalid_stage() -> None:
    assert _routing_engine("always-native", {}) == "pypdf"
    assert _routing_engine("always-ocr", {}) == "hybrid-pdf"
    assert _routing_engine("document-router", {"layout": "scanned"}) == "hybrid-pdf"
    assert _routing_engine("page-hybrid-router", {"layout": "digital"}) == "pypdf"
    with pytest.raises(ValueError, match="Unknown routing"):
        _routing_engine("unknown", {})
    with pytest.raises(ValueError, match="Unsupported extraction"):
        run_extraction_stage("unknown", "smoke")
