"""Retrieval-augmented generation public API."""

from .contracts import (
    ChunkingStrategy,
    EmbeddingSpec,
    GenerationProfile,
    IndexManifest,
    RecommendationManifest,
    Reranker,
    RetrievalStrategy,
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
    "RecommendationManifest",
    "Reranker",
    "RetrievalHit",
    "RetrievalStrategy",
]
