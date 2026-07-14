from __future__ import annotations

from dataclasses import replace

import pytest

from edumind.benchmarks.datasets import (
    DatasetValidationError,
    assert_no_split_leakage,
    load_manifest,
)
from edumind.common.paths import PROJECT_ROOT


def test_smoke_dataset_has_eight_documents_24_questions_and_exact_offsets() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    assert sum(item.get("kind") == "document" for item in manifest.samples) == 8
    assert sum(item.get("kind") == "question" for item in manifest.samples) == 24


def test_manifest_checksum_tampering_is_detected(tmp_path) -> None:
    source = PROJECT_ROOT / "data/benchmarks/rag/smoke.json"
    content = source.read_text(encoding="utf-8").replace("chemical energy", "changed", 1)
    path = tmp_path / "tampered.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="checksum"):
        load_manifest(path)


def test_document_level_leakage_is_rejected() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    with pytest.raises(DatasetValidationError, match="leakage"):
        assert_no_split_leakage([manifest, replace(manifest, split="other")])


def test_near_duplicate_document_leakage_is_rejected() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    document = next(item for item in manifest.samples if item.get("kind") == "document")
    altered = {
        **document,
        "id": "near-copy",
        "text": str(document["text"]) + " extra",
    }
    second = replace(manifest, name="near", split="other", samples=(altered,))
    with pytest.raises(DatasetValidationError, match="Near-duplicate"):
        assert_no_split_leakage([manifest, second])
