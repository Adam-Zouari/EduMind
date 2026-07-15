"""Retrieval-augmented generation public API."""

from .contracts import (
    ChunkingStrategy,
    EmbeddingSpec,
    GenerationProfile,
    IndexManifest,
)
from .types import AnswerResult, ChunkRecord, IngestDocument, IngestReport, RetrievalHit

__all__ = [
    "AnswerResult",
    "ChunkRecord",
    "ChunkingStrategy",
    "EmbeddingSpec",
    "GenerationProfile",
    "IndexManifest",
    "IngestDocument",
    "IngestReport",
    "RetrievalHit",
]
