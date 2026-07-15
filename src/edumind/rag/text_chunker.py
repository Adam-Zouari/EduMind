"""Offset-preserving production chunking strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass

from edumind.common.artifacts import stable_hash

from .contracts import ChunkingStrategy
from .tokenizers import OffsetTokenizer, TiktokenOffsetTokenizer
from .types import ChunkRecord, IngestDocument, build_chunk_id


@dataclass(frozen=True)
class TokenChunkingStrategy:
    tokenizer: OffsetTokenizer
    size: int
    overlap: int
    name: str = "token"

    def __post_init__(self) -> None:
        if self.size <= 0 or self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("Token chunking requires size > overlap >= 0")

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "size": self.size,
                "overlap": self.overlap,
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        tokens = self.tokenizer.spans(text)
        if not tokens:
            return []
        step = self.size - self.overlap
        chunks: list[tuple[int, int, int]] = []
        for token_start in range(0, len(tokens), step):
            token_end = min(token_start + self.size, len(tokens))
            start = tokens[token_start][0]
            end = tokens[token_end - 1][1]
            chunks.append((start, end, token_end - token_start))
            if token_end == len(tokens):
                break
        return chunks


@dataclass(frozen=True)
class SentenceChunkingStrategy:
    tokenizer: OffsetTokenizer
    sentences: int = 8
    overlap: int = 2
    name: str = "sentence"

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "sentences": self.sentences,
                "overlap": self.overlap,
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        sentence_spans = [
            (match.start(), match.end()) for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n|$)", text)
        ]
        if not sentence_spans:
            return []
        step = max(1, self.sentences - self.overlap)
        chunks = []
        for index in range(0, len(sentence_spans), step):
            selected = sentence_spans[index : index + self.sentences]
            start, end = selected[0][0], selected[-1][1]
            chunks.append((start, end, self.tokenizer.count(text[start:end])))
            if index + self.sentences >= len(sentence_spans):
                break
        return chunks


@dataclass(frozen=True)
class RecursiveCharacterChunkingStrategy:
    tokenizer: OffsetTokenizer
    size: int = 1000
    overlap: int = 200
    name: str = "recursive-character"

    @property
    def fingerprint(self) -> str:
        return stable_hash({"name": self.name, "size": self.size, "overlap": self.overlap})

    def split(self, text: str) -> list[tuple[int, int, int]]:
        if not text.strip():
            return []
        chunks: list[tuple[int, int, int]] = []
        start = 0
        while start < len(text):
            target = min(start + self.size, len(text))
            end = target
            if target < len(text):
                boundaries = [
                    text.rfind(separator, start, target) for separator in ("\n\n", "\n", ". ", " ")
                ]
                boundary = max(boundaries)
                if boundary > start + self.size // 2:
                    end = boundary + 1
            chunks.append((start, end, self.tokenizer.count(text[start:end])))
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap)
        return chunks


class SemanticChunkingStrategy:
    """Sentence-boundary semantic chunking with a hard token ceiling."""

    name = "semantic"

    def __init__(
        self,
        tokenizer: OffsetTokenizer,
        embed_sentences,
        maximum_tokens: int = 384,
        percentile: float = 0.2,
    ) -> None:
        self.tokenizer = tokenizer
        self.embed_sentences = embed_sentences
        self.maximum_tokens = maximum_tokens
        self.percentile = percentile

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "maximum_tokens": self.maximum_tokens,
                "percentile": self.percentile,
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        import numpy as np

        spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n|$)", text)]
        if len(spans) < 2:
            return [(0, len(text), self.tokenizer.count(text))] if text else []
        sentences = [text[start:end] for start, end in spans]
        vectors = np.asarray(self.embed_sentences(sentences), dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors /= norms
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.quantile(similarities, self.percentile))
        break_after = {index for index, score in enumerate(similarities) if score <= threshold}
        chunks: list[tuple[int, int, int]] = []
        start_index = 0
        for index in range(len(spans)):
            start = spans[start_index][0]
            end = spans[index][1]
            if index == len(spans) - 1 or index in break_after:
                chunks.extend(self._bounded(text, start, end))
                start_index = index + 1
        return chunks
    def _bounded(self, text: str, start: int, end: int) -> list[tuple[int, int, int]]:
        local = self.tokenizer.spans(text[start:end])
        if not local:
            return []
        return [
            (
                start + local[index][0],
                start + local[min(index + self.maximum_tokens, len(local)) - 1][1],
                min(self.maximum_tokens, len(local) - index),
            )
            for index in range(0, len(local), self.maximum_tokens)
        ]


def build_chunking_strategy(
    name: str, *, tokenizer: OffsetTokenizer | None = None, embed_sentences=None
) -> ChunkingStrategy:
    tokenizer = tokenizer or TiktokenOffsetTokenizer()
    if name == "token-256-32":
        return TokenChunkingStrategy(tokenizer, 256, 32, name)
    if name == "token-384-64":
        return TokenChunkingStrategy(tokenizer, 384, 64, "token-384-64")
    if name == "sentence-8-2":
        return SentenceChunkingStrategy(tokenizer, 8, 2, name)
    if name == "recursive-character":
        return RecursiveCharacterChunkingStrategy(tokenizer)
    if name == "semantic":
        if embed_sentences is None:
            raise ValueError(
                "Semantic chunking requires the production sentence embedding function"
            )
        return SemanticChunkingStrategy(tokenizer, embed_sentences)
    raise ValueError(f"Unknown chunking strategy: {name}")


class TextChunker:
    """Turn exact strategy spans into deterministic index records."""

    def __init__(
        self,
        strategy: ChunkingStrategy,
    ) -> None:
        self.strategy = strategy
        self.chunk_size = getattr(strategy, "size", getattr(strategy, "maximum_tokens", 0))
        self.chunk_overlap = getattr(strategy, "overlap", 0)

    def chunk_document(self, document: IngestDocument) -> list[ChunkRecord]:
        spans = self.strategy.split(document.text)
        total = len(spans)
        chunks: list[ChunkRecord] = []
        for index, (start, end, tokens) in enumerate(spans):
            text = document.text[start:end]
            metadata = {
                **document.metadata,
                "source": document.source,
                "source_id": document.source_id,
                "chunk_index": index,
                "total_chunks": total,
                "start": start,
                "end": end,
                "token_count": tokens,
            }
            chunks.append(
                ChunkRecord(
                    id=build_chunk_id(document.source_id, start, end, text),
                    source_id=document.source_id,
                    text=text,
                    chunk_index=index,
                    total_chunks=total,
                    start=start,
                    end=end,
                    token_count=tokens,
                    metadata=metadata,
                    filter_metadata=dict(document.filter_metadata),
                )
            )
        return chunks
