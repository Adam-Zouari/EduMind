"""Embedding utilities for the RAG pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from edumind.common.config import load_yaml_config

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using sentence-transformers."""

    def __init__(self, config_path: str | None = None):
        self.config = load_yaml_config(config_path)
        embed_config = self.config["embedding"]
        self.model_name = embed_config["model_name"]
        self.device = embed_config["device"]
        self.embedding_dim = embed_config["embedding_dim"]

        if self.device == "cuda" and (torch is None or not torch.cuda.is_available()):
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            self.device = "cpu"

        logger.info(f"Loading embedding model {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        if not texts:
            return np.array([])
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32,
        )

    def embed_chunks(self, chunks: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
        if not chunks:
            return []

        texts = [chunk.get(text_key, "") for chunk in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.tolist()
        return chunks
