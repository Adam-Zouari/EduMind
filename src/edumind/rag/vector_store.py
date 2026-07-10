"""Vector store operations for the RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from edumind.common.config import load_yaml_config

from .embedder import Embedder
from .errors import MetadataFilterError, RAGConfigurationError
from .types import (
    ChunkRecord,
    FilterMetadata,
    MetadataScalar,
    RAGConfig,
    RetrievalHit,
    VectorStoreSettings,
    is_metadata_scalar,
)

logger = logging.getLogger(__name__)

LEXICAL_MANIFEST_FILENAME = "lexical_index.json"
LEGACY_BM25_FILENAME = "bm25_index.pkl"
RESERVED_RAW_METADATA_KEY = "__raw_metadata"
RESERVED_SOURCE_ID_KEY = "__source_id"
RESERVED_CHUNK_INDEX_KEY = "__chunk_index"
RESERVED_TOTAL_CHUNKS_KEY = "__total_chunks"


@dataclass(frozen=True)
class LexicalEntry:
    """Persisted lexical retrieval record."""

    id: str
    document: str
    filter_metadata: FilterMetadata


class VectorStore:
    """Manage Chroma persistence plus a JSON-backed BM25 lexical manifest."""

    def __init__(
        self,
        settings: VectorStoreSettings | None = None,
        config_path: str | None = None,
    ) -> None:
        if settings is None:
            raw_config = load_yaml_config(config_path)
            settings = RAGConfig.from_mapping(raw_config).vector_store

        if settings.distance_metric != "cosine":
            raise RAGConfigurationError(
                "Unsupported distance metric "
                f"'{settings.distance_metric}'. Only 'cosine' is supported."
            )

        self.collection_name = settings.collection_name
        self.distance_metric = settings.distance_metric
        self.persist_directory = settings.persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.bm25_alpha = settings.bm25_alpha

        persistent_client_cls, settings_cls, bm25_cls = _load_runtime_dependencies()
        self._bm25_cls = bm25_cls
        self.client = persistent_client_cls(
            path=str(self.persist_directory),
            settings=settings_cls(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

        self.lexical_manifest_path = self.persist_directory / LEXICAL_MANIFEST_FILENAME
        self.legacy_bm25_path = self.persist_directory / LEGACY_BM25_FILENAME
        self.lexical_entries: list[LexicalEntry] = self._load_lexical_entries()
        self.bm25: Any | None = None
        self.doc_map: list[str] = []
        self._rebuild_bm25()

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        """Persist chunk records into dense and lexical retrieval stores."""
        if not chunks:
            return 0

        batch_size = 5000
        for index in range(0, len(chunks), batch_size):
            self._upsert_batch(chunks[index : index + batch_size])

        lexical_map = {entry.id: entry for entry in self.lexical_entries}
        for chunk in chunks:
            lexical_map[chunk.id] = LexicalEntry(
                id=chunk.id,
                document=chunk.text,
                filter_metadata=dict(chunk.filter_metadata),
            )
        self.lexical_entries = [lexical_map[key] for key in sorted(lexical_map)]
        self._save_lexical_entries()
        self._rebuild_bm25()
        return len(chunks)

    def add_documents(self, chunks: Sequence[ChunkRecord]) -> int:
        """Backward-compatible alias for the new upsert behavior."""
        return self.upsert_chunks(chunks)

    def query_by_text(
        self,
        query_text: str,
        embedder: Embedder,
        *,
        top_k: int = 5,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        """Embed a query and run hybrid retrieval."""
        query_embedding = embedder.embed_text(query_text)
        return self.query_hybrid(
            query_text,
            query_embedding.tolist(),
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

    def query_hybrid(
        self,
        query_text: str,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        """Run dense plus lexical retrieval and merge scores."""
        validated_filter = self._validate_filter_metadata(filter_metadata)
        dense_hits = self._query_dense(
            query_embedding,
            top_k=top_k,
            filter_metadata=validated_filter,
        )
        lexical_scores = self._query_lexical(
            query_text,
            top_k=top_k,
            filter_metadata=validated_filter,
        )

        all_ids = set(dense_hits) | set(lexical_scores)
        combined_hits: list[RetrievalHit] = []
        for doc_id in all_ids:
            dense_hit = dense_hits.get(doc_id)
            dense_score = dense_hit.score if dense_hit is not None else 0.0
            lexical_score = lexical_scores.get(doc_id, 0.0)
            final_score = (
                (1.0 - self.bm25_alpha) * dense_score
            ) + (self.bm25_alpha * lexical_score)

            if dense_hit is not None:
                combined_hits.append(
                    RetrievalHit(
                        id=dense_hit.id,
                        document=dense_hit.document,
                        metadata=dense_hit.metadata,
                        score=final_score,
                    )
                )
                continue

            fetched_hit = self._fetch_hit_by_id(doc_id)
            if fetched_hit is None:
                continue
            combined_hits.append(
                RetrievalHit(
                    id=fetched_hit.id,
                    document=fetched_hit.document,
                    metadata=fetched_hit.metadata,
                    score=final_score,
                )
            )

        combined_hits.sort(key=lambda hit: hit.score, reverse=True)
        return combined_hits[:top_k]

    def get_collection_count(self) -> int:
        """Return the number of indexed chunks."""
        return int(self.collection.count())

    def delete_collection(self) -> None:
        """Delete both dense and lexical retrieval state."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as exc:
            logger.warning("Could not delete Chroma collection %s: %s", self.collection_name, exc)

        for path in (self.lexical_manifest_path, self.legacy_bm25_path):
            if path.exists():
                os.remove(path)

        self.lexical_entries = []
        self.bm25 = None
        self.doc_map = []

    def reset_collection(self) -> None:
        """Recreate an empty collection and lexical manifest."""
        self.delete_collection()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def _upsert_batch(self, chunks: Sequence[ChunkRecord]) -> None:
        """Persist a batch of dense vectors into Chroma."""
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} is missing an embedding")

            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(self._serialize_metadata(chunk))

        self.collection.upsert(
            ids=ids,
            embeddings=cast(Any, embeddings),
            documents=documents,
            metadatas=cast(Any, metadatas),
        )

    def _serialize_metadata(self, chunk: ChunkRecord) -> dict[str, str | int | float | bool]:
        """Convert chunk metadata into Chroma-safe flat metadata."""
        serialized: dict[str, str | int | float | bool] = dict(chunk.filter_metadata)
        serialized[RESERVED_SOURCE_ID_KEY] = chunk.source_id
        serialized[RESERVED_CHUNK_INDEX_KEY] = chunk.chunk_index
        serialized[RESERVED_TOTAL_CHUNKS_KEY] = chunk.total_chunks
        serialized[RESERVED_RAW_METADATA_KEY] = json.dumps(
            chunk.metadata,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return serialized

    def _deserialize_metadata(self, metadata: Mapping[str, object] | None) -> dict[str, object]:
        """Rebuild the public metadata payload from stored flat metadata."""
        if metadata is None:
            return {}

        rebuilt: dict[str, object] = {}
        raw_metadata = metadata.get(RESERVED_RAW_METADATA_KEY)
        if isinstance(raw_metadata, str) and raw_metadata:
            try:
                payload = json.loads(raw_metadata)
                if isinstance(payload, dict):
                    rebuilt.update(payload)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed stored metadata payload")

        for key, value in metadata.items():
            if key in {
                RESERVED_RAW_METADATA_KEY,
                RESERVED_SOURCE_ID_KEY,
                RESERVED_CHUNK_INDEX_KEY,
                RESERVED_TOTAL_CHUNKS_KEY,
            }:
                continue
            rebuilt[key] = value

        if RESERVED_SOURCE_ID_KEY in metadata:
            rebuilt["source_id"] = metadata[RESERVED_SOURCE_ID_KEY]
        if RESERVED_CHUNK_INDEX_KEY in metadata:
            rebuilt["chunk_index"] = metadata[RESERVED_CHUNK_INDEX_KEY]
        if RESERVED_TOTAL_CHUNKS_KEY in metadata:
            rebuilt["total_chunks"] = metadata[RESERVED_TOTAL_CHUNKS_KEY]
        return rebuilt

    def _load_lexical_entries(self) -> list[LexicalEntry]:
        """Load the lexical manifest from disk."""
        if self.legacy_bm25_path.exists() and not self.lexical_manifest_path.exists():
            logger.info("Ignoring legacy BM25 pickle in favor of the new lexical manifest format")

        if not self.lexical_manifest_path.exists():
            return []

        try:
            with self.lexical_manifest_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to load lexical manifest: %s", exc)
            return []

        if not isinstance(payload, list):
            return []

        entries: list[LexicalEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            filter_metadata = item.get("filter_metadata", {})
            if not isinstance(filter_metadata, dict):
                filter_metadata = {}
            entries.append(
                LexicalEntry(
                    id=str(item.get("id", "")),
                    document=str(item.get("document", "")),
                    filter_metadata={
                        key: value
                        for key, value in filter_metadata.items()
                        if is_metadata_scalar(value)
                    },
                )
            )
        return entries

    def _save_lexical_entries(self) -> None:
        """Persist the lexical manifest to disk."""
        payload = [
            {
                "id": entry.id,
                "document": entry.document,
                "filter_metadata": entry.filter_metadata,
            }
            for entry in self.lexical_entries
        ]
        with self.lexical_manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _rebuild_bm25(self) -> None:
        """Rebuild the in-memory BM25 index from lexical entries."""
        if not self.lexical_entries:
            self.bm25 = None
            self.doc_map = []
            return

        self.doc_map = [entry.id for entry in self.lexical_entries]
        corpus = [self._tokenize(entry.document) for entry in self.lexical_entries]
        self.bm25 = self._bm25_cls(corpus)

    def _query_dense(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        filter_metadata: FilterMetadata,
    ) -> dict[str, RetrievalHit]:
        """Query Chroma and return normalized dense hits."""
        dense_results: Any = self.collection.query(
            query_embeddings=cast(Any, [query_embedding]),
            n_results=max(top_k * 2, top_k),
            where=cast(Any, filter_metadata or None),
            include=["documents", "metadatas", "distances"],
        )

        dense_hits: dict[str, RetrievalHit] = {}
        ids = dense_results.get("ids") or []
        if not ids or not ids[0]:
            return dense_hits

        distances = dense_results.get("distances") or [[]]
        documents = dense_results.get("documents") or [[]]
        metadatas = dense_results.get("metadatas") or [[]]
        for index, doc_id in enumerate(ids[0]):
            distance = float(distances[0][index]) if distances and distances[0] else 1.0
            dense_hits[str(doc_id)] = RetrievalHit(
                id=str(doc_id),
                document=str(documents[0][index]),
                metadata=self._deserialize_metadata(metadatas[0][index]),
                score=max(0.0, min(1.0, 1.0 - distance)),
            )
        return dense_hits

    def _query_lexical(
        self,
        query_text: str,
        *,
        top_k: int,
        filter_metadata: FilterMetadata,
    ) -> dict[str, float]:
        """Query the in-memory BM25 index with optional scalar metadata filtering."""
        if self.bm25 is None or not self.lexical_entries:
            return {}

        scores = self.bm25.get_scores(self._tokenize(query_text))
        scored_entries = [
            (entry.id, float(score))
            for entry, score in zip(self.lexical_entries, scores, strict=False)
            if self._matches_filter(entry.filter_metadata, filter_metadata)
        ]
        if not scored_entries:
            return {}

        scored_entries.sort(key=lambda item: item[1], reverse=True)
        top_scored_entries = scored_entries[: max(top_k * 2, top_k)]
        max_score = max(score for _, score in top_scored_entries)
        if max_score <= 0:
            return {doc_id: 0.0 for doc_id, _ in top_scored_entries}
        return {doc_id: score / max_score for doc_id, score in top_scored_entries}

    def _fetch_hit_by_id(self, doc_id: str) -> RetrievalHit | None:
        """Fetch one stored chunk by id from Chroma."""
        try:
            fetched: Any = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        except Exception as exc:
            logger.warning("Could not fetch document %s from Chroma: %s", doc_id, exc)
            return None

        documents = fetched.get("documents") or []
        metadatas = fetched.get("metadatas") or []
        if not documents:
            return None
        return RetrievalHit(
            id=doc_id,
            document=str(documents[0]),
            metadata=self._deserialize_metadata(metadatas[0] if metadatas else None),
            score=0.0,
        )

    def _validate_filter_metadata(
        self,
        filter_metadata: Mapping[str, object] | None,
    ) -> FilterMetadata:
        """Validate that only scalar top-level filters are used."""
        if filter_metadata is None:
            return {}

        validated: FilterMetadata = {}
        for key, value in filter_metadata.items():
            if not is_metadata_scalar(value):
                raise MetadataFilterError(
                    "Only scalar top-level metadata filters are supported "
                    f"(got {key}={type(value).__name__})"
                )
            validated[key] = value
        return validated

    def _matches_filter(
        self,
        stored_filter_metadata: Mapping[str, MetadataScalar],
        requested_filter_metadata: Mapping[str, MetadataScalar],
    ) -> bool:
        """Return whether one lexical entry matches the requested filter."""
        for key, value in requested_filter_metadata.items():
            if stored_filter_metadata.get(key) != value:
                return False
        return True

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for lexical retrieval."""
        return text.lower().split()


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    """Load vector-store dependencies only when the store is constructed."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ModuleNotFoundError as exc:
        raise RAGConfigurationError(
            "ChromaDB is required for the RAG vector store. Install the `rag` extra."
        ) from exc

    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise RAGConfigurationError(
            "rank_bm25 is required for lexical retrieval. Install the `rag` extra."
        ) from exc

    return chromadb.PersistentClient, Settings, BM25Okapi
