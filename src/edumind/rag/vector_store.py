"""Provisional production vector store: Chroma over HTTP, dense retrieval only."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import urlparse

from edumind.common.config import load_settings

from .contracts import IndexManifest
from .errors import IndexCompatibilityError, MetadataFilterError, RAGConfigurationError
from .types import ChunkRecord, MetadataScalar, RetrievalHit, VectorStoreSettings, is_metadata_scalar

RAW_METADATA = "__raw_metadata"
SOURCE_ID = "__source_id"
CHUNK_INDEX = "__chunk_index"
TOTAL_CHUNKS = "__total_chunks"
START = "__start"
END = "__end"
TOKEN_COUNT = "__token_count"
COMPATIBILITY = "edumind:compatibility"


class VectorStore:
    """Small Chroma HTTP adapter used by the provisional application."""

    def __init__(
        self,
        settings: VectorStoreSettings | None = None,
        config_path: str | None = None,
    ) -> None:
        if settings is None:
            configured = load_settings(config_path).vector
            settings = VectorStoreSettings(
                configured.collection_name,
                configured.endpoint,
                configured.distance_metric,
            )
        if settings.distance_metric not in {"cosine", "dot"}:
            raise RAGConfigurationError(f"Unsupported vector distance: {settings.distance_metric}")
        endpoint = urlparse(settings.endpoint)
        if not endpoint.hostname or not endpoint.port:
            raise RAGConfigurationError("Chroma endpoint requires a host and port")
        self.collection_name = settings.collection_name
        self.endpoint = settings.endpoint
        self.distance_metric = settings.distance_metric
        chromadb = _load_chromadb()
        try:
            self.client = chromadb.HttpClient(
                host=endpoint.hostname,
                port=endpoint.port,
                ssl=endpoint.scheme == "https",
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
        except Exception as exc:
            raise RAGConfigurationError(
                f"Cannot connect to Chroma at {self.endpoint}. Start it with "
                "`docker compose -f infrastructure/chroma.yml up -d`."
            ) from exc
        self._manifest: IndexManifest | None = None

    def ensure_manifest(self, expected: IndexManifest) -> IndexManifest:
        metadata = dict(self.collection.metadata or {})
        existing = metadata.get(COMPATIBILITY)
        if existing is not None and existing != expected.compatibility_key:
            raise IndexCompatibilityError(
                "The Chroma collection uses incompatible embedding or chunking settings. "
                "Reset the collection before using the new configuration."
            )
        if existing is None:
            metadata[COMPATIBILITY] = expected.compatibility_key
            metadata.setdefault("hnsw:space", self.distance_metric)
            self.collection.modify(metadata=metadata)
        self._manifest = expected
        return expected

    def load_index_manifest(self) -> IndexManifest | None:
        return self._manifest

    def replace_document(self, source_id: str, chunks: Sequence[ChunkRecord]) -> int:
        if any(chunk.source_id != source_id for chunk in chunks):
            raise ValueError("Every chunk must belong to the document being replaced")
        previous: Any = self.collection.get(
            where={SOURCE_ID: source_id}, include=["documents", "metadatas", "embeddings"]
        )
        previous_ids = list(previous.get("ids") or [])
        self.collection.delete(where={SOURCE_ID: source_id})
        try:
            self._upsert(chunks)
        except Exception:
            self.collection.delete(where={SOURCE_ID: source_id})
            if previous_ids:
                self.collection.upsert(
                    ids=previous_ids,
                    documents=previous.get("documents") or [],
                    metadatas=previous.get("metadatas") or [],
                    embeddings=previous.get("embeddings") or [],
                )
            raise
        return len(previous_ids)

    def query_dense(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        filters = self._filters(filter_metadata)
        response: Any = self.collection.query(
            query_embeddings=cast(Any, [list(query_embedding)]),
            n_results=max(1, top_k),
            where=cast(Any, self._chroma_where(filters)),
            include=["documents", "metadatas", "distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        hits: list[RetrievalHit] = []
        for index, doc_id in enumerate(ids):
            metadata = self._deserialize(metadatas[index])
            distance = float(distances[index])
            score = 1.0 - distance if self.distance_metric == "cosine" else -distance
            hits.append(
                RetrievalHit(
                    str(doc_id),
                    str(documents[index]),
                    metadata,
                    score,
                    index + 1,
                    "dense",
                    _as_int(metadata.get("token_count")),
                )
            )
        return hits

    def get_collection_count(self) -> int:
        return int(self.collection.count())

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        self._manifest = None

    def _upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        for offset in range(0, len(chunks), 5000):
            batch = chunks[offset : offset + 5000]
            if any(chunk.embedding is None for chunk in batch):
                raise ValueError("Every indexed chunk requires an embedding")
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                embeddings=cast(Any, [chunk.embedding for chunk in batch]),
                documents=[chunk.text for chunk in batch],
                metadatas=cast(Any, [self._serialize(chunk) for chunk in batch]),
            )

    @staticmethod
    def _serialize(chunk: ChunkRecord) -> dict[str, MetadataScalar]:
        metadata: dict[str, MetadataScalar] = dict(chunk.filter_metadata)
        metadata.update(
            {
                RAW_METADATA: json.dumps(chunk.metadata, ensure_ascii=False, default=str),
                SOURCE_ID: chunk.source_id,
                CHUNK_INDEX: chunk.chunk_index,
                TOTAL_CHUNKS: chunk.total_chunks,
                START: chunk.start,
                END: chunk.end,
                TOKEN_COUNT: chunk.token_count,
            }
        )
        return metadata

    @staticmethod
    def _deserialize(metadata: Mapping[str, object] | None) -> dict[str, object]:
        if not metadata:
            return {}
        raw = metadata.get(RAW_METADATA)
        try:
            result = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            result = {}
        if not isinstance(result, dict):
            result = {}
        reserved = {RAW_METADATA, SOURCE_ID, CHUNK_INDEX, TOTAL_CHUNKS, START, END, TOKEN_COUNT}
        result.update({key: value for key, value in metadata.items() if key not in reserved})
        result.update(
            {
                "source_id": metadata.get(SOURCE_ID),
                "chunk_index": metadata.get(CHUNK_INDEX),
                "total_chunks": metadata.get(TOTAL_CHUNKS),
                "start": metadata.get(START),
                "end": metadata.get(END),
                "token_count": metadata.get(TOKEN_COUNT, 0),
            }
        )
        return result

    @staticmethod
    def _filters(filters: Mapping[str, object] | None) -> dict[str, MetadataScalar]:
        validated: dict[str, MetadataScalar] = {}
        for key, value in (filters or {}).items():
            if not is_metadata_scalar(value):
                raise MetadataFilterError(f"Filter '{key}' must have a scalar value")
            validated[key] = value
        return validated

    @staticmethod
    def _chroma_where(filters: Mapping[str, MetadataScalar]) -> dict[str, object] | None:
        clauses = [{key: value} for key, value in sorted(filters.items())]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _load_chromadb():
    try:
        import chromadb
    except ModuleNotFoundError as exc:
        raise RAGConfigurationError("Install the application dependencies to use Chroma") from exc
    return chromadb


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, (str, int, float)) else 0
