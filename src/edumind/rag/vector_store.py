"""Manifest-protected Chroma persistence and independent lexical retrieval."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from edumind.common.artifacts import atomic_write_json, local_file_lock, stable_hash
from edumind.common.config import load_settings

from .contracts import IndexManifest
from .errors import IndexCompatibilityError, MetadataFilterError, RAGConfigurationError
from .types import (
    ChunkRecord,
    FilterMetadata,
    MetadataScalar,
    RetrievalHit,
    VectorStoreSettings,
    is_metadata_scalar,
)

logger = logging.getLogger(__name__)

LEXICAL_MANIFEST_FILENAME = "lexical_index.json"
INDEX_MANIFEST_FILENAME = "index_manifest.json"
RESERVED_RAW_METADATA_KEY = "__raw_metadata"
RESERVED_SOURCE_ID_KEY = "__source_id"
RESERVED_CHUNK_INDEX_KEY = "__chunk_index"
RESERVED_TOTAL_CHUNKS_KEY = "__total_chunks"
RESERVED_START_KEY = "__start"
RESERVED_END_KEY = "__end"
RESERVED_TOKEN_COUNT_KEY = "__token_count"


@dataclass(frozen=True)
class LexicalEntry:
    id: str
    source_id: str
    document: str
    filter_metadata: FilterMetadata


class VectorStore:
    """Runtime Chroma backend with atomic logical-document replacement."""

    def __init__(
        self,
        settings: VectorStoreSettings | None = None,
        config_path: str | None = None,
    ) -> None:
        if settings is None:
            configured = load_settings(config_path).vector
            settings = VectorStoreSettings(
                configured.collection_name, configured.persist_directory, configured.distance_metric
            )
        if settings.distance_metric not in {"cosine", "dot"}:
            raise RAGConfigurationError(f"Unsupported vector distance: {settings.distance_metric}")
        self.collection_name = settings.collection_name
        self.distance_metric = settings.distance_metric
        self.persist_directory = settings.persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        client_class, settings_class, bm25_class = _load_runtime_dependencies()
        self._bm25_class = bm25_class
        self.client = client_class(
            path=str(self.persist_directory), settings=settings_class(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": self.distance_metric}
        )
        self.lexical_manifest_path = self.persist_directory / LEXICAL_MANIFEST_FILENAME
        self.index_manifest_path = self.persist_directory / INDEX_MANIFEST_FILENAME
        self._lock_path = self.persist_directory / ".index.lock"
        self.lexical_entries = self._load_lexical_entries()
        self.bm25: Any | None = None
        self.doc_map: list[str] = []
        self._rebuild_bm25()

    def ensure_manifest(self, expected: IndexManifest) -> IndexManifest:
        existing = self.load_index_manifest()
        if existing is not None and existing.compatibility_key != expected.compatibility_key:
            raise IndexCompatibilityError(
                "The persisted index has incompatible embedding, chunking, or backend contracts. "
                f"Rebuild {self.persist_directory} before querying it."
            )
        if existing is None:
            atomic_write_json(self.index_manifest_path, asdict(expected))
            return expected
        return existing

    def load_index_manifest(self) -> IndexManifest | None:
        if not self.index_manifest_path.is_file():
            return None
        try:
            payload = json.loads(self.index_manifest_path.read_text(encoding="utf-8"))
            payload["filter_fields"] = tuple(payload.get("filter_fields", []))
            return IndexManifest(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IndexCompatibilityError(
                f"Index manifest is unreadable; rebuild {self.persist_directory}: {exc}"
            ) from exc

    def replace_document(self, source_id: str, chunks: Sequence[ChunkRecord]) -> int:
        """Replace all chunks for one logical document while retaining rollback data."""
        if any(chunk.source_id != source_id for chunk in chunks):
            raise ValueError("Every replacement chunk must belong to the requested source_id")
        with local_file_lock(self._lock_path):
            old_entries = [entry for entry in self.lexical_entries if entry.source_id == source_id]
            old_dense = self._snapshot_source(source_id)
            try:
                self.collection.delete(where={RESERVED_SOURCE_ID_KEY: source_id})
                self._upsert_batches(chunks)
            except Exception:
                try:
                    self.collection.delete(where={RESERVED_SOURCE_ID_KEY: source_id})
                    if old_dense:
                        self.collection.upsert(**old_dense)
                except Exception as rollback_error:  # pragma: no cover
                    logger.exception("Dense rollback failed: %s", rollback_error)
                raise
            self.lexical_entries = [
                entry for entry in self.lexical_entries if entry.source_id != source_id
            ]
            self.lexical_entries.extend(
                LexicalEntry(chunk.id, source_id, chunk.text, dict(chunk.filter_metadata))
                for chunk in chunks
            )
            self.lexical_entries.sort(key=lambda item: item.id)
            try:
                self._save_lexical_entries()
            except Exception:
                self.lexical_entries = [
                    entry for entry in self.lexical_entries if entry.source_id != source_id
                ] + old_entries
                self.lexical_entries.sort(key=lambda item: item.id)
                raise
            self._rebuild_bm25()
            self._update_manifest(chunks)
        return len(old_entries)

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        grouped: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.source_id, []).append(chunk)
        for source_id, source_chunks in grouped.items():
            self.replace_document(source_id, source_chunks)
        return len(chunks)

    def add_documents(self, chunks: Sequence[ChunkRecord]) -> int:
        return self.upsert_chunks(chunks)

    def query_dense(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        filters = self._validate_filter_metadata(filter_metadata)
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
        hits = []
        for rank, doc_id in enumerate(ids, start=1):
            distance = float(distances[rank - 1])
            score = 1.0 - distance if self.distance_metric == "cosine" else -distance
            metadata = self._deserialize_metadata(metadatas[rank - 1])
            hits.append(
                RetrievalHit(
                    str(doc_id),
                    str(documents[rank - 1]),
                    metadata,
                    score,
                    rank,
                    "dense",
                    _metadata_int(metadata.get("token_count")),
                )
            )
        return hits

    def query_lexical(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        filters = self._validate_filter_metadata(filter_metadata)
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query_text))
        ranked = sorted(
            (
                (entry, float(score))
                for entry, score in zip(self.lexical_entries, scores, strict=True)
                if _matches(entry.filter_metadata, filters)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        hits = []
        for rank, (entry, score) in enumerate(ranked, start=1):
            hydrated = self._fetch_hit_by_id(entry.id)
            if hydrated is not None:
                hits.append(replace(hydrated, score=score, rank=rank, retrieval_method="bm25"))
        return hits

    def query_by_text(self, query_text: str, embedder, *, top_k: int = 5, filter_metadata=None):
        """Dense query convenience retained for direct callers."""
        return self.query_dense(
            embedder.embed_query(query_text), top_k=top_k, filter_metadata=filter_metadata
        )

    def get_collection_count(self) -> int:
        return int(self.collection.count())

    def reset_collection(self) -> None:
        with local_file_lock(self._lock_path):
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception as exc:
                logger.warning("Could not delete collection %s: %s", self.collection_name, exc)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": self.distance_metric}
            )
            self.lexical_entries = []
            self.bm25 = None
            self.doc_map = []
            self.lexical_manifest_path.unlink(missing_ok=True)
            self.index_manifest_path.unlink(missing_ok=True)

    def _snapshot_source(self, source_id: str) -> dict[str, object] | None:
        try:
            payload: Any = self.collection.get(
                where={RESERVED_SOURCE_ID_KEY: source_id},
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            return None
        ids = payload.get("ids") or []
        if not ids:
            return None
        return {
            "ids": ids,
            "documents": payload.get("documents") or [],
            "metadatas": payload.get("metadatas") or [],
            "embeddings": payload.get("embeddings") or [],
        }

    def _upsert_batches(self, chunks: Sequence[ChunkRecord]) -> None:
        for index in range(0, len(chunks), 5000):
            batch = chunks[index : index + 5000]
            if any(chunk.embedding is None for chunk in batch):
                raise ValueError("Every indexed chunk must have an embedding")
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                embeddings=cast(Any, [chunk.embedding for chunk in batch]),
                documents=[chunk.text for chunk in batch],
                metadatas=cast(Any, [self._serialize_metadata(chunk) for chunk in batch]),
            )

    def _serialize_metadata(self, chunk: ChunkRecord) -> dict[str, MetadataScalar]:
        payload: dict[str, MetadataScalar] = dict(chunk.filter_metadata)
        payload.update(
            {
                RESERVED_SOURCE_ID_KEY: chunk.source_id,
                RESERVED_CHUNK_INDEX_KEY: chunk.chunk_index,
                RESERVED_TOTAL_CHUNKS_KEY: chunk.total_chunks,
                RESERVED_START_KEY: chunk.start,
                RESERVED_END_KEY: chunk.end,
                RESERVED_TOKEN_COUNT_KEY: chunk.token_count,
                RESERVED_RAW_METADATA_KEY: json.dumps(
                    chunk.metadata, ensure_ascii=False, sort_keys=True, default=str
                ),
            }
        )
        return payload

    def _deserialize_metadata(self, metadata: Mapping[str, object] | None) -> dict[str, object]:
        if metadata is None:
            return {}
        rebuilt: dict[str, object] = {}
        raw = metadata.get(RESERVED_RAW_METADATA_KEY)
        if isinstance(raw, str):
            try:
                value = json.loads(raw)
                if isinstance(value, dict):
                    rebuilt.update(value)
            except json.JSONDecodeError:
                logger.warning("Stored raw metadata is malformed")
        reserved = {
            RESERVED_RAW_METADATA_KEY,
            RESERVED_SOURCE_ID_KEY,
            RESERVED_CHUNK_INDEX_KEY,
            RESERVED_TOTAL_CHUNKS_KEY,
            RESERVED_START_KEY,
            RESERVED_END_KEY,
            RESERVED_TOKEN_COUNT_KEY,
        }
        rebuilt.update({key: value for key, value in metadata.items() if key not in reserved})
        rebuilt.update(
            {
                "source_id": metadata.get(RESERVED_SOURCE_ID_KEY),
                "chunk_index": metadata.get(RESERVED_CHUNK_INDEX_KEY),
                "total_chunks": metadata.get(RESERVED_TOTAL_CHUNKS_KEY),
                "start": metadata.get(RESERVED_START_KEY),
                "end": metadata.get(RESERVED_END_KEY),
                "token_count": metadata.get(RESERVED_TOKEN_COUNT_KEY, 0),
            }
        )
        return rebuilt

    def _fetch_hit_by_id(self, doc_id: str) -> RetrievalHit | None:
        payload: Any = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        documents = payload.get("documents") or []
        if not documents:
            return None
        metadata = self._deserialize_metadata((payload.get("metadatas") or [None])[0])
        return RetrievalHit(
            doc_id,
            str(documents[0]),
            metadata,
            0.0,
            token_count=_metadata_int(metadata.get("token_count")),
        )

    def _validate_filter_metadata(self, filters: Mapping[str, object] | None) -> FilterMetadata:
        if not filters:
            return {}
        manifest = self.load_index_manifest()
        allowed = set(manifest.filter_fields if manifest else ())
        validated: FilterMetadata = {}
        for key, value in filters.items():
            if not is_metadata_scalar(value):
                raise MetadataFilterError(f"Filter '{key}' must have a scalar value")
            if key not in allowed:
                raise MetadataFilterError(
                    f"Filter field '{key}' is not stored in this index. Available: "
                    f"{', '.join(sorted(allowed)) or 'none'}"
                )
            validated[key] = value
        return validated

    @staticmethod
    def _chroma_where(filters: FilterMetadata) -> dict[str, object] | None:
        if not filters:
            return None
        clauses: list[dict[str, object]] = [{key: value} for key, value in sorted(filters.items())]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _load_lexical_entries(self) -> list[LexicalEntry]:
        if not self.lexical_manifest_path.is_file():
            return []
        try:
            payload = json.loads(self.lexical_manifest_path.read_text(encoding="utf-8"))
            rows = payload.get("entries", [])
            return [
                LexicalEntry(
                    str(row["id"]),
                    str(row["source_id"]),
                    str(row["document"]),
                    {
                        key: value
                        for key, value in row.get("filter_metadata", {}).items()
                        if is_metadata_scalar(value)
                    },
                )
                for row in rows
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable lexical manifest: %s", exc)
            return []

    def _save_lexical_entries(self) -> None:
        atomic_write_json(
            self.lexical_manifest_path,
            {"schema": 2, "entries": [asdict(entry) for entry in self.lexical_entries]},
        )

    def _rebuild_bm25(self) -> None:
        self.doc_map = [entry.id for entry in self.lexical_entries]
        self.bm25 = (
            self._bm25_class([_tokenize(entry.document) for entry in self.lexical_entries])
            if self.lexical_entries
            else None
        )

    def _update_manifest(self, chunks: Sequence[ChunkRecord]) -> None:
        manifest = self.load_index_manifest()
        if manifest is None:
            return
        fields = set(manifest.filter_fields)
        for chunk in chunks:
            fields.update(chunk.filter_metadata)
        content_checksum = stable_hash(
            [
                {
                    "id": entry.id,
                    "source_id": entry.source_id,
                    "document_checksum": stable_hash(entry.document),
                }
                for entry in self.lexical_entries
            ]
        )
        atomic_write_json(
            self.index_manifest_path,
            asdict(
                replace(
                    manifest,
                    content_checksum=content_checksum,
                    filter_fields=tuple(sorted(fields)),
                )
            ),
        )


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def _matches(stored: Mapping[str, MetadataScalar], requested: Mapping[str, MetadataScalar]) -> bool:
    return all(stored.get(key) == value for key, value in requested.items())


def _metadata_int(value: object) -> int:
    return int(value) if isinstance(value, (str, int, float)) else 0


def _load_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        import chromadb
        from chromadb.config import Settings
    except ModuleNotFoundError as exc:
        raise RAGConfigurationError(
            "ChromaDB is required for the runtime backend; install .[rag]"
        ) from exc
    try:
        from rank_bm25 import BM25Okapi
    except ModuleNotFoundError as exc:
        raise RAGConfigurationError("rank_bm25 is required for retrieval; install .[rag]") from exc
    return chromadb.PersistentClient, Settings, BM25Okapi
