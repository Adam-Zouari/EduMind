from __future__ import annotations

from pathlib import Path

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
            if where is None or all(payload["metadata"].get(key) == value for key, value in where.items())
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
        payload = self.records[ids[0]]
        return {"documents": [payload["document"]], "metadatas": [payload["metadata"]]}

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
        return [float(len(set(tokens) & set(query_tokens))) for tokens in self.corpus]


class FakeSettings:
    def __init__(self, anonymized_telemetry: bool) -> None:
        self.anonymized_telemetry = anonymized_telemetry


def test_dense_and_lexical_queries_are_exposed(monkeypatch, tmp_path: Path) -> None:
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
    store.upsert_chunks([chunk])

    dense_hits = store.query_dense([0.1, 0.2], top_k=5)
    lexical_hits = store.query_lexical("alpha", top_k=5)

    assert dense_hits
    assert lexical_hits
    assert dense_hits[0].id == chunk.id
    assert lexical_hits[0].id == chunk.id
