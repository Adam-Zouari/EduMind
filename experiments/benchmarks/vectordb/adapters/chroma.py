"""Chroma HTTP server adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import Config, Hit, InvalidIndexState, Record, ensure_dimension


class Chroma:
    def __init__(self, config: Config) -> None:
        import chromadb
        from chromadb.config import Settings

        self.config = config
        self.client = chromadb.HttpClient(
            host="127.0.0.1",
            port=8001,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection: Any | None = None

    def health(self) -> bool:
        return bool(self.client.heartbeat())

    def reset(self) -> None:
        from chromadb.errors import NotFoundError

        self.collection = None
        try:
            self.client.delete_collection(self.config.collection)
        except NotFoundError:
            pass
        except Exception as exc:
            raise InvalidIndexState("Chroma could not delete the previous collection") from exc
        self.collection = self.client.create_collection(
            self.config.collection,
            metadata={"hnsw:space": "cosine"},
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "max_neighbors": self.config.m,
                    "ef_construction": self.config.ef_construction,
                    "ef_search": self.config.ef_search,
                }
            },
        )
        self.index_info()

    def upsert(self, records: Sequence[Record]) -> None:
        ensure_dimension(self.config, records)
        for start in range(0, len(records), 2_000):
            rows = records[start : start + 2_000]
            self._collection().upsert(
                ids=[row.identifier for row in rows],
                embeddings=[list(row.vector) for row in rows],
                documents=[row.text for row in rows],
                metadatas=[dict(row.metadata) for row in rows],
            )

    def search(self, vector, limit, filters=None) -> list[Hit]:
        clauses = [{key: value} for key, value in sorted((filters or {}).items())]
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
        result = self._collection().query(
            query_embeddings=[list(vector)],
            n_results=limit,
            where=where,
            include=["metadatas", "distances"],
        )
        ids = result["ids"][0]
        metadata = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            Hit(str(identifier), 1.0 - float(distances[index]), metadata[index] or {})
            for index, identifier in enumerate(ids)
        ]

    def delete(self, identifiers: Sequence[str]) -> None:
        if identifiers:
            self._collection().delete(ids=list(identifiers))

    def delete_document(self, source_id: str) -> int:
        rows = self._collection().get(where={"source_id": source_id}, include=[])
        self._collection().delete(where={"source_id": source_id})
        return len(rows["ids"])

    def count(self) -> int:
        return int(self._collection().count())

    def index_info(self) -> Mapping[str, object]:
        value = getattr(self._collection(), "configuration", None)
        if not isinstance(value, Mapping):
            value = getattr(self._collection(), "configuration_json", None)
        if not isinstance(value, Mapping):
            raise InvalidIndexState("Chroma did not report its collection configuration")
        hnsw = value.get("hnsw")
        if not isinstance(hnsw, Mapping):
            raise InvalidIndexState("Chroma did not report an HNSW index")
        expected = {
            "space": "cosine",
            "max_neighbors": self.config.m,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
        }
        mismatches = {
            name: {"expected": expected_value, "actual": hnsw.get(name)}
            for name, expected_value in expected.items()
            if hnsw.get(name) != expected_value
        }
        if mismatches:
            raise InvalidIndexState(f"Chroma HNSW configuration mismatch: {mismatches}")
        return {"type": "hnsw", "configuration": dict(hnsw)}

    def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if callable(closer):
            closer()

    def _collection(self):
        if self.collection is None:
            self.collection = self.client.get_collection(self.config.collection)
        return self.collection
