"""Text chunking for the RAG pipeline."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from edumind.common.config import load_yaml_config

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text using semantic similarity with a configurable target size."""

    def __init__(self, config_path: str | None = None):
        self.config = load_yaml_config(config_path)
        chunk_config = self.config["chunking"]
        self.chunk_size = chunk_config.get("chunk_size", 1000)
        self.min_chunk_size = min(500, self.chunk_size)

        embedding_config = self.config.get("embedding", {})
        model_name = embedding_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        device = embedding_config.get("device", "cpu")
        logger.info(f"Loading chunking model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def _split_into_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        if len(text) < self.chunk_size:
            chunk_obj = {"text": text, "chunk_index": 0, "total_chunks": 1}
            if metadata:
                chunk_obj.update(metadata)
            return [chunk_obj]

        embeddings = self.model.encode(sentences)
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]

        breakpoints: list[int] = []
        if similarities:
            threshold = np.percentile(similarities, 10)
            breakpoints = [index for index, value in enumerate(similarities) if value < threshold]

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for index, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_length += len(sentence)

            is_semantic_break = index in breakpoints
            is_too_big = current_length >= self.chunk_size
            is_big_enough = current_length >= self.min_chunk_size
            is_last = index == len(sentences) - 1

            if (is_semantic_break and is_big_enough) or is_too_big or is_last:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        chunk_objects: list[dict[str, Any]] = []
        for idx, chunk_text in enumerate(chunks):
            chunk_obj: dict[str, Any] = {
                "text": chunk_text,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            if metadata:
                chunk_obj.update(metadata)
            chunk_objects.append(chunk_obj)

        return chunk_objects

    def chunk_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_chunks: list[dict[str, Any]] = []
        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")
            metadata = {key: value for key, value in doc.items() if key != "text"}
            metadata["document_index"] = doc_idx
            all_chunks.extend(self.chunk_text(text, metadata))
        return all_chunks
