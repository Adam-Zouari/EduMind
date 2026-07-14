from __future__ import annotations

import json

from edumind.common.config import load_settings
from edumind.extraction import ExtractionPipeline, ExtractionProfile, SourceKind
from edumind.extraction.extractors.base import build_document
from edumind.extraction.registry import ExtractorRegistration, ExtractorRegistry


class FakeExtractor:
    name = "fake"
    revision = "1"
    supported_kinds = frozenset({SourceKind.IMAGE})

    def __init__(self) -> None:
        self.calls = 0
        self.options = {}

    def extract(self, request, kind):
        self.calls += 1
        self.options = dict(request.options)
        return build_document(request, kind, request.profile, ["alpha  beta"], pages=[1])


def test_pipeline_uses_registry_normalization_and_cache(tmp_path) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"fake image")
    fake = FakeExtractor()
    registry = ExtractorRegistry()
    registry.register(
        ExtractorRegistration("fake", frozenset({SourceKind.IMAGE}), lambda: fake, "", "image")
    )
    settings = load_settings(overrides={"extraction": {"cache_directory": str(tmp_path / "cache")}})
    pipeline = ExtractionPipeline(settings, registry)
    profile = ExtractionProfile("test", "fake", "1", normalization="conservative")
    first = pipeline.extract(source, profile=profile)
    second = pipeline.extract(source, profile=profile)
    assert first.text == "alpha beta"
    assert second.cache_hit
    assert fake.calls == 1


def test_pipeline_injects_prepared_model_paths_from_lock(tmp_path) -> None:
    source = tmp_path / "page.png"
    source.write_bytes(b"fake image")
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    lock = tmp_path / "extraction-models.json"
    lock.write_text(
        json.dumps(
            {
                "models": {
                    "fake": {
                        "revision": "abc123",
                        "model_path": str(model_directory),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    fake = FakeExtractor()
    registry = ExtractorRegistry()
    registry.register(
        ExtractorRegistration("fake", frozenset({SourceKind.IMAGE}), lambda: fake, "", "image")
    )
    settings = load_settings(
        overrides={
            "extraction": {
                "cache_directory": str(tmp_path / "cache"),
                "model_lock_path": str(lock),
            }
        }
    )
    ExtractionPipeline(settings, registry).extract(
        source, profile=ExtractionProfile("test", "fake", "abc123")
    )
    assert fake.options["model_path"] == str(model_directory)
