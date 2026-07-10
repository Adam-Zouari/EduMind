"""Text chunking for the RAG pipeline."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

import numpy as np

from edumind.common.config import load_yaml_config

from .embedder import Embedder
from .types import (
    ChunkingSettings,
    ChunkRecord,
    IngestDocument,
    RAGConfig,
    build_chunk_id,
    build_source_id,
    sanitize_filter_metadata,
)

logger = logging.getLogger(__name__)


class TextChunker:
    """Chunk text using shared embeddings plus active overlap and separators."""

    def __init__(
        self,
        settings: ChunkingSettings | None = None,
        embedder: Embedder | None = None,
        config_path: str | None = None,
    ) -> None:
        if settings is None or embedder is None:
            raw_config = load_yaml_config(config_path)
            typed_config = RAGConfig.from_mapping(raw_config)
            settings = settings or typed_config.chunking
            embedder = embedder or Embedder(settings=typed_config.embedding)

        self.settings = settings
        self.embedder = embedder
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = max(0, min(settings.chunk_overlap, settings.chunk_size))
        self.separators = tuple(settings.separators)

    def chunk_document(self, document: IngestDocument) -> list[ChunkRecord]:
        """Chunk one normalized document into deterministic chunk records."""
        text = document.text.strip()
        if not text:
            return []

        units = self._build_units(text)
        if not units:
            return []

        if len(units) == 1 and len(units[0]) <= self.chunk_size:
            chunk_texts = [units[0]]
        else:
            breakpoints = self._choose_breakpoints(units)
            chunk_texts = self._assemble_chunks(units, breakpoints)

        base_metadata = dict(document.metadata)
        base_metadata.setdefault("source", document.source)
        if document.format_type:
            base_metadata.setdefault("format_type", document.format_type)
        if document.file_path:
            base_metadata.setdefault("file_path", document.file_path)

        total_chunks = len(chunk_texts)
        chunks: list[ChunkRecord] = []
        for chunk_index, chunk_text in enumerate(chunk_texts):
            metadata = dict(base_metadata)
            metadata["source_id"] = document.source_id
            metadata["chunk_index"] = chunk_index
            metadata["total_chunks"] = total_chunks
            chunks.append(
                ChunkRecord(
                    id=build_chunk_id(document.source_id, chunk_index, chunk_text),
                    source_id=document.source_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    metadata=metadata,
                    filter_metadata=dict(document.filter_metadata),
                )
            )
        return chunks

    def chunk_text(
        self,
        text: str,
        metadata: Mapping[str, object] | None = None,
        *,
        source_id: str | None = None,
    ) -> list[ChunkRecord]:
        """Chunk freeform text by first normalizing it into an ingest document."""
        normalized_metadata = dict(metadata or {})
        source = str(normalized_metadata.get("source", "text-input"))
        format_type = _as_string(normalized_metadata.get("format_type"))
        file_path = _as_string(normalized_metadata.get("file_path"))
        resolved_source_id = source_id or build_source_id(
            text=text,
            source=source,
            file_path=file_path,
            format_type=format_type,
            metadata=normalized_metadata,
        )

        document = IngestDocument(
            text=text,
            source_id=resolved_source_id,
            source=source,
            format_type=format_type,
            file_path=file_path,
            metadata=normalized_metadata,
            filter_metadata=sanitize_filter_metadata(
                normalized_metadata,
                source=source,
                format_type=format_type,
                file_path=file_path,
            ),
        )
        return self.chunk_document(document)

    def chunk_documents(self, documents: Sequence[IngestDocument]) -> list[ChunkRecord]:
        """Chunk many normalized documents."""
        all_chunks: list[ChunkRecord] = []
        for document in documents:
            all_chunks.extend(self.chunk_document(document))
        return all_chunks

    def _build_units(self, text: str) -> list[str]:
        """Split text into chunkable units using configured separators and sentences."""
        blocks = self._split_with_separators(text)
        units: list[str] = []
        for block in blocks:
            sentences = self._split_into_sentences(block)
            if not sentences:
                continue

            for sentence in sentences:
                if len(sentence) <= self.chunk_size:
                    units.append(sentence)
                else:
                    units.extend(self._split_oversized_unit(sentence))

        return units

    def _split_with_separators(self, text: str) -> list[str]:
        """Use configured separators to break oversized text into smaller blocks."""
        blocks = [text.strip()]
        for separator in self.separators:
            if not separator:
                continue

            next_blocks: list[str] = []
            for block in blocks:
                if len(block) <= self.chunk_size or separator not in block:
                    next_blocks.append(block)
                    continue

                parts = [part.strip() for part in block.split(separator) if part.strip()]
                if len(parts) <= 1:
                    next_blocks.append(block)
                else:
                    next_blocks.extend(parts)
            blocks = next_blocks

        return [block for block in blocks if block]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split a block into sentences while handling line-oriented study notes."""
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _split_oversized_unit(self, text: str) -> list[str]:
        """Fallback split for very large units with weak punctuation."""
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_length = 0
        for word in words:
            extra_length = len(word) + (1 if current else 0)
            if current and current_length + extra_length > self.chunk_size:
                chunks.append(" ".join(current))
                current = [word]
                current_length = len(word)
            else:
                current.append(word)
                current_length += extra_length

        if current:
            chunks.append(" ".join(current))
        return chunks

    def _choose_breakpoints(self, units: Sequence[str]) -> set[int]:
        """Choose semantic breakpoints between adjacent units."""
        if len(units) < 2:
            return set()

        embeddings = self.embedder.embed_texts(list(units), show_progress=False)
        similarities = [
            _cosine_similarity(embeddings[index], embeddings[index + 1])
            for index in range(len(units) - 1)
        ]
        if not similarities:
            return set()

        threshold = float(np.percentile(similarities, 10))
        return {index for index, value in enumerate(similarities) if value <= threshold}

    def _assemble_chunks(self, units: Sequence[str], breakpoints: set[int]) -> list[str]:
        """Assemble chunks while honoring semantic breaks and configured overlap."""
        chunks: list[str] = []
        current_units: list[str] = []
        current_length = 0
        index = 0

        while index < len(units):
            unit = units[index]
            unit_length = len(unit)
            if (
                current_units
                and current_length + unit_length > self.chunk_size
                and current_length >= self.settings.min_chunk_size
            ):
                chunks.append(" ".join(current_units).strip())
                current_units = self._overlap_tail(current_units)
                current_length = _joined_length(current_units)
                continue

            current_units.append(unit)
            current_length = _joined_length(current_units)
            is_break = index in breakpoints and current_length >= self.settings.min_chunk_size
            is_last = index == len(units) - 1

            if is_break or current_length >= self.chunk_size or is_last:
                chunks.append(" ".join(current_units).strip())
                if is_last:
                    current_units = []
                    current_length = 0
                else:
                    current_units = self._overlap_tail(current_units)
                    current_length = _joined_length(current_units)
            index += 1

        return [chunk for chunk in chunks if chunk]

    def _overlap_tail(self, units: Sequence[str]) -> list[str]:
        """Carry trailing units into the next chunk according to chunk_overlap."""
        if self.chunk_overlap <= 0 or not units:
            return []

        tail: list[str] = []
        accumulated = 0
        for unit in reversed(units):
            extra_length = len(unit) + (1 if tail else 0)
            if tail and accumulated + extra_length > self.chunk_overlap:
                break
            tail.insert(0, unit)
            accumulated += extra_length
        return tail


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity for two embedding vectors."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _joined_length(units: Sequence[str]) -> int:
    """Estimate the final joined chunk length."""
    if not units:
        return 0
    return sum(len(unit) for unit in units) + max(0, len(units) - 1)


def _as_string(value: object) -> str | None:
    """Normalize optional string-like values."""
    if isinstance(value, str) and value:
        return value
    return None
