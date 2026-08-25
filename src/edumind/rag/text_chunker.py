"""Production token chunking and exact-offset chunk records."""

from __future__ import annotations

from dataclasses import dataclass

from edumind.common.artifacts import stable_hash

from .contracts import ChunkingStrategy
from .tokenizers import OffsetTokenizer
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


class TextChunker:
    """Turn exact strategy spans into deterministic index records."""

    def __init__(self, strategy: ChunkingStrategy) -> None:
        self.strategy = strategy
        self.chunk_size = getattr(strategy, "size", 0)
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
