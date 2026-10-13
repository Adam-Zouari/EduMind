"""Production token chunking and exact-offset chunk records."""

from __future__ import annotations

from bisect import bisect_right
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
        ends = [end for _, end in tokens]
        chunks: list[tuple[int, int, int]] = []
        start = 0
        while start < len(text):
            token_start = bisect_right(ends, start)
            if token_start == len(tokens):
                break
            end = ends[min(token_start + self.size, len(tokens)) - 1]
            # A UTF-8 character can span multiple IDs. Retokenize the intact
            # source string instead of misreporting an ID slice as its length.
            while end > start and self.tokenizer.count(text[start:end]) > self.size:
                end -= 1
            if end == start:
                raise ValueError("Token ceiling cannot contain one source character")
            local = self.tokenizer.spans(text[start:end])
            chunks.append((start, end, len(local)))
            if end == len(text):
                break
            next_start = (
                end
                if not self.overlap
                else start + local[max(0, len(local) - self.overlap)][0]
            )
            start = max(start + 1, next_start)
        return chunks


class TextChunker:
    """Turn exact strategy spans into deterministic index records."""

    def __init__(self, strategy: ChunkingStrategy) -> None:
        self.strategy = strategy

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
