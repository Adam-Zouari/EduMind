from __future__ import annotations

import pytest

from edumind.rag.contracts import IndexManifest
from edumind.rag.errors import IndexCompatibilityError, MetadataFilterError
from edumind.rag.types import ChunkRecord, VectorStoreSettings
from edumind.rag.vector_store import VectorStore


def test_chroma_multi_filter_uses_and() -> None:
    assert VectorStore._chroma_where({"course": "math", "page": 1}) == {
        "$and": [{"course": "math"}, {"page": 1}]
    }


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query):
        return [float(len(set(query) & set(tokens))) for tokens in self.corpus]


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        for values in zip(ids, embeddings, documents, metadatas, strict=True):
            self.rows[values[0]] = values[1:]

    def delete(self, where):
        source = where.get("__source_id")
        self.rows = {
            key: value for key, value in self.rows.items() if value[2].get("__source_id") != source
        }

    def get(self, ids=None, where=None, include=None):
        del include
        rows = self.rows.items()
        if ids:
            rows = [(key, self.rows[key]) for key in ids if key in self.rows]
        if where:
            rows = [
                (key, value)
                for key, value in rows
                if value[2].get("__source_id") == where["__source_id"]
            ]
        rows = list(rows)
        return {
            "ids": [key for key, _ in rows],
            "embeddings": [value[0] for _, value in rows],
            "documents": [value[1] for _, value in rows],
            "metadatas": [value[2] for _, value in rows],
        }

    def query(self, **kwargs):
        where = kwargs.get("where")
        rows = list(self.rows.items())
        if where and "$and" in where:
            for clause in where["$and"]:
                key, value = next(iter(clause.items()))
                rows = [(row_id, row) for row_id, row in rows if row[2].get(key) == value]
        return {
            "ids": [[key for key, _ in rows]],
            "documents": [[row[1] for _, row in rows]],
            "metadatas": [[row[2] for _, row in rows]],
            "distances": [[0.1 for _ in rows]],
        }

    def count(self):
        return len(self.rows)


class FakeClient:
    def __init__(self, **kwargs):
        self.collection = FakeCollection()

    def get_or_create_collection(self, **kwargs):
        return self.collection

    def delete_collection(self, **kwargs):
        self.collection = FakeCollection()


class FakeSettings:
    def __init__(self, **kwargs):
        pass


def _chunk(text="alpha", identifier="source:0") -> ChunkRecord:
    return ChunkRecord(
        identifier,
        "source",
        text,
        0,
        1,
        0,
        len(text),
        1,
        {"source": "x"},
        {"course": "math", "page": 1},
        [1.0, 0.0],
    )


def test_manifest_replacement_and_filter_validation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "edumind.rag.vector_store._load_runtime_dependencies",
        lambda: (FakeClient, FakeSettings, FakeBM25),
    )
    store = VectorStore(VectorStoreSettings("test", tmp_path, "cosine"))
    manifest = IndexManifest(1, "content", "embedding", "chunking", "chroma", "test")
    store.ensure_manifest(manifest)
    assert store.replace_document("source", [_chunk()]) == 0
    first_content_checksum = store.load_index_manifest().content_checksum
    assert store.replace_document("source", [_chunk("updated", "source:1")]) == 1
    assert store.load_index_manifest().content_checksum != first_content_checksum
    assert store.get_collection_count() == 1
    hits = store.query_dense([1.0, 0.0], top_k=5, filter_metadata={"course": "math", "page": 1})
    assert hits[0].document == "updated"
    with pytest.raises(MetadataFilterError, match="not stored"):
        store.query_dense([1.0, 0.0], filter_metadata={"missing": "x"})
    with pytest.raises(IndexCompatibilityError):
        store.ensure_manifest(IndexManifest(1, "content", "other", "chunking", "chroma", "test"))
