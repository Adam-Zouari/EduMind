"""Experimental vector-backend adapters used by the staged benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from edumind.common.config import load_yaml_config
from edumind.rag.types import ChunkRecord, RAGConfig, RetrievalHit
from edumind.rag.vector_store import VectorStore


class VectorBackendUnavailableError(RuntimeError):
    """Raised when an experimental backend cannot be used locally."""


@dataclass(frozen=True)
class VectorBackendSpec:
    """One vector-backend candidate in the experiment grid."""

    name: str
    description: str


class ExperimentVectorBackend:
    """Thin adapter over one vector backend with shared query modes."""

    def __init__(
        self,
        *,
        spec: VectorBackendSpec,
        persist_directory: Path,
        collection_name: str,
        bm25_alpha: float,
        config_path: str | None = None,
    ) -> None:
        self.spec = spec
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.bm25_alpha = bm25_alpha

        if spec.name != "chroma":
            raise VectorBackendUnavailableError(
                f"{spec.name} benchmarking is scaffolded but unavailable in this local runtime. "
                "Install and wire the backend adapter before running this candidate."
            )

        raw_config = load_yaml_config(config_path)
        rag_config = RAGConfig.from_mapping(raw_config)
        settings = replace(
            rag_config.vector_store,
            collection_name=collection_name,
            persist_directory=persist_directory,
            bm25_alpha=bm25_alpha,
        )
        self._store = VectorStore(settings=settings)
        self._store.reset_collection()

    def upsert_chunks(self, chunks: Sequence[ChunkRecord]) -> int:
        """Persist chunks in the selected vector backend."""
        return self._store.upsert_chunks(chunks)

    def query(
        self,
        *,
        mode: str,
        query_text: str,
        query_embedding: list[float],
        top_k: int,
        filter_metadata: Mapping[str, object] | None = None,
        bm25_alpha: float | None = None,
    ) -> list[RetrievalHit]:
        """Query the backend with dense, lexical, or hybrid retrieval."""
        if bm25_alpha is not None:
            self._store.bm25_alpha = bm25_alpha

        if mode == "dense_only":
            return self._store.query_dense(
                query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        if mode == "bm25_only":
            return self._store.query_lexical(
                query_text,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        if mode == "hybrid":
            return self._store.query_hybrid(
                query_text,
                query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def reset(self) -> None:
        """Clear backend state for a fresh experiment run."""
        self._store.reset_collection()

    def count(self) -> int:
        """Return the current indexed chunk count."""
        return self._store.get_collection_count()

    def storage_size_mb(self) -> float:
        """Estimate on-disk storage size for the backend persist directory."""
        total_bytes = 0
        if not self.persist_directory.exists():
            return 0.0
        for path in self.persist_directory.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size
        return total_bytes / (1024 * 1024)
