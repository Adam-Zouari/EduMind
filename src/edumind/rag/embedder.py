"""Embedding utilities for the RAG pipeline."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from edumind.common.config import load_yaml_config

from .errors import RAGConfigurationError
from .types import ChunkRecord, EmbeddingSettings, RAGConfig

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using a lazily loaded sentence-transformers model."""

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        config_path: str | None = None,
    ) -> None:
        if settings is None:
            raw_config = load_yaml_config(config_path)
            settings = RAGConfig.from_mapping(raw_config).embedding

        self.model_name = settings.model_name
        self.embedding_dim = settings.embedding_dim
        self.batch_size = settings.batch_size
        self.device = self._resolve_device(settings.device)
        self._model: SentenceTransformer | None = None

    @property
    def model_loaded(self) -> bool:
        """Return whether the embedding model has already been instantiated."""
        return self._model is not None

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text value into a dense vector."""
        if not text or not text.strip():
            return np.zeros(self.embedding_dim, dtype=float)
        return self.embed_texts([text], show_progress=False)[0]

    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed many text values at once."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=float)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=self.batch_size,
        )
        array = np.asarray(embeddings, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        self._validate_dimension(array.shape[1])
        return array

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Return chunk records with embeddings attached."""
        if not chunks:
            return []

        embeddings = self.embed_texts([chunk.text for chunk in chunks], show_progress=False)
        return [
            replace(chunk, embedding=embedding.tolist())
            for chunk, embedding in zip(chunks, embeddings, strict=False)
        ]

    def _get_model(self) -> SentenceTransformer:
        """Load and cache the sentence-transformers model on first use."""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model {self.model_name} on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve a requested device while keeping import-time behavior light."""
        if requested_device != "cuda":
            return requested_device

        try:
            import torch
        except ImportError:  # pragma: no cover - optional runtime dependency
            logger.warning("CUDA requested but torch is unavailable. Falling back to CPU.")
            return "cpu"

        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return requested_device

    def _validate_dimension(self, actual_dimension: int) -> None:
        """Ensure the configured embedding dimension matches the model output."""
        if actual_dimension != self.embedding_dim:
            raise RAGConfigurationError(
                "Configured embedding_dim does not match model output "
                f"({self.embedding_dim} != {actual_dimension})"
            )
