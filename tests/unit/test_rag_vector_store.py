from __future__ import annotations

import logging
from pathlib import Path

import pytest

from edumind.rag.errors import MetadataFilterError
from edumind.rag.types import ChunkRecord, VectorStoreSettings
from edumind.rag.vector_store import VectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        for doc_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=False,
        ):
            self.records[doc_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        del query_embeddings, include
        filtered = [
            (doc_id, payload)
            for doc_id, payload in self.records.items()
            if _matches_where(payload["metadata"], where)
        ]
        limited = filtered[:n_results]
        return {
            "ids": [[doc_id for doc_id, _ in limited]],
            "documents": [[str(payload["document"]) for _, payload in limited]],
            "metadatas": [[payload["metadata"] for _, payload in limited]],
            "distances": [[0.1 for _ in limited]],
        }

    def get(self, *, ids, include=None):
        del include
        doc_id = ids[0]
        payload = self.records[doc_id]
        return {
            "documents": [payload["document"]],
            "metadatas": [payload["metadata"]],
        }

    def count(self) -> int:
        return len(self.records)


class FakePersistentClient:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.collection = FakeCollection()

    def get_or_create_collection(self, name: str, metadata: dict[str, object]):
        del name, metadata
        return self.collection

    def delete_collection(self, name: str) -> None:
        del name
        self.collection = FakeCollection()


class FakeBM25Okapi:
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        return [
            float(len(set(tokens) & set(query_tokens)))
            for tokens in self.corpus
        ]


def _matches_where(metadata: object, where: dict[str, object] | None) -> bool:
    if where is None:
        return True
    if not isinstance(metadata, dict):
        return False
    for key, value in where.items():
        if metadata.get(key) != value:
            return False
    return True


def test_upsert_query_filter_and_reset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "edumind.rag.vector_store._load_runtime_dependencies",
        lambda: (FakePersistentClient, FakeSettings, FakeBM25Okapi),
    )

    store = VectorStore(
        settings=VectorStoreSettings(
            collection_name="test-collection",
            persist_directory=tmp_path,
            distance_metric="cosine",
        )
    )
    chunk = ChunkRecord(
        id="doc-1:0",
        source_id="doc-1",
        text="alpha beta gamma",
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "lesson.pdf", "page": 1},
        filter_metadata={"source": "lesson.pdf", "page": 1, "topic": "math"},
        embedding=[0.1, 0.2],
    )
    updated_chunk = ChunkRecord(
        id="doc-1:0",
        source_id="doc-1",
        text="alpha beta gamma updated",
        chunk_index=0,
        total_chunks=1,
        metadata={"source": "lesson.pdf", "page": 1},
        filter_metadata={"source": "lesson.pdf", "page": 1, "topic": "math"},
        embedding=[0.2, 0.3],
    )

    store.upsert_chunks([chunk])
    store.upsert_chunks([updated_chunk])
    results = store.query_hybrid(
        "alpha",
        [0.1, 0.2],
        top_k=5,
        filter_metadata={"topic": "math"},
    )

    assert store.get_collection_count() == 1
    assert len(store.lexical_entries) == 1
    assert results
    assert results[0].metadata["source"] == "lesson.pdf"
    assert 0.0 <= results[0].score <= 1.0

    store.reset_collection()
    assert store.get_collection_count() == 0
    assert store.lexical_entries == []
    assert not (tmp_path / "lexical_index.json").exists()


def test_query_rejects_non_scalar_filters(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "edumind.rag.vector_store._load_runtime_dependencies",
        lambda: (FakePersistentClient, FakeSettings, FakeBM25Okapi),
    )

    store = VectorStore(
        settings=VectorStoreSettings(
            collection_name="test-collection",
            persist_directory=tmp_path,
            distance_metric="cosine",
        )
    )

    with pytest.raises(MetadataFilterError, match="Only scalar top-level metadata filters"):
        store.query_hybrid("alpha", [0.1, 0.2], filter_metadata={"nested": {"bad": True}})


def test_malformed_lexical_manifest_is_ignored(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(
        "edumind.rag.vector_store._load_runtime_dependencies",
        lambda: (FakePersistentClient, FakeSettings, FakeBM25Okapi),
    )
    manifest_path = tmp_path / "lexical_index.json"
    manifest_path.write_text("{bad json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = VectorStore(
            settings=VectorStoreSettings(
                collection_name="test-collection",
                persist_directory=tmp_path,
                distance_metric="cosine",
            )
        )

    assert store.lexical_entries == []
    assert "Failed to load lexical manifest" in caplog.text


class FakeSettings:
    def __init__(self, anonymized_telemetry: bool) -> None:
        self.anonymized_telemetry = anonymized_telemetry
