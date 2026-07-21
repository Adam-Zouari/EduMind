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


@dataclass(frozen=True)
class SectionAwareChunkingStrategy:
    """Keep Markdown sections intact where possible, then apply a token ceiling."""

    tokenizer: OffsetTokenizer
    size: int = 512
    overlap: int = 64
    name: str = "section-aware-512-64"

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "size": self.size,
                "overlap": self.overlap,
                "heading_pattern": "markdown-v1",
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        if not text.strip():
            return []
        headings = [match.start() for match in re.finditer(r"(?m)^#{1,6}[ \t]+\S", text)]
        boundaries = sorted({0, *headings, len(text)})
        if len(boundaries) == 2:
            return _token_spans(text, 0, len(text), self.tokenizer, self.size, self.overlap)
        chunks: list[tuple[int, int, int]] = []
        for start, end in zip(boundaries, boundaries[1:]):
            if text[start:end].strip():
                chunks.extend(
                    _token_spans(text, start, end, self.tokenizer, self.size, self.overlap)
                )
        return chunks


@dataclass(frozen=True)
class StructureAwareChunkingStrategy:
    """Respect Markdown sections, tables, and display formulas under a token ceiling."""

    tokenizer: OffsetTokenizer
    size: int = 512
    overlap: int = 64
    name: str = "structure-aware-512-64"

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "size": self.size,
                "overlap": self.overlap,
                "structure_parser": "markdown-table-formula-v1",
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        if not text.strip():
            return []
        protected = _structured_spans(text)
        headings = [match.start() for match in re.finditer(r"(?m)^#{1,6}[ \t]+\S", text)]
        boundaries = sorted(
            {
                0,
                len(text),
                *headings,
                *(value for span in protected for value in span),
            }
        )
        units = [
            (start, end)
            for start, end in zip(boundaries, boundaries[1:])
            if text[start:end].strip()
        ]
        chunks: list[tuple[int, int, int]] = []
        pending_start: int | None = None
        pending_end: int | None = None
        for start, end in units:
            unit_tokens = self.tokenizer.count(text[start:end])
            if unit_tokens > self.size:
                if pending_start is not None and pending_end is not None:
                    chunks.append(
                        (
                            pending_start,
                            pending_end,
                            self.tokenizer.count(text[pending_start:pending_end]),
                        )
                    )
                    pending_start = pending_end = None
                chunks.extend(
                    _split_structured_unit(
                        text, start, end, self.tokenizer, self.size, self.overlap
                    )
                )
                continue
            proposed_start = start if pending_start is None else pending_start
            proposed_tokens = self.tokenizer.count(text[proposed_start:end])
            if pending_start is not None and proposed_tokens > self.size:
                assert pending_end is not None
                chunks.append(
                    (
                        pending_start,
                        pending_end,
                        self.tokenizer.count(text[pending_start:pending_end]),
                    )
                )
                pending_start = start
            elif pending_start is None:
                pending_start = start
            pending_end = end
        if pending_start is not None and pending_end is not None:
            chunks.append(
                (
                    pending_start,
                    pending_end,
                    self.tokenizer.count(text[pending_start:pending_end]),
                )
            )
        return chunks


def _token_spans(
    text: str,
    start: int,
    end: int,
    tokenizer: OffsetTokenizer,
    size: int,
    overlap: int,
) -> list[tuple[int, int, int]]:
    local = tokenizer.spans(text[start:end])
    if not local:
        return []
    step = size - overlap
    result: list[tuple[int, int, int]] = []
    for token_start in range(0, len(local), step):
        token_end = min(token_start + size, len(local))
        result.append(
            (
                start + local[token_start][0],
                start + local[token_end - 1][1],
                token_end - token_start,
            )
        )
        if token_end == len(local):
            break
    return result


def _structured_spans(text: str) -> list[tuple[int, int]]:
    formulas = [
        match.span()
        for match in re.finditer(r"(?s)\$\$.*?\$\$|\\\[.*?\\\]", text)
    ]
    tables = []
    for match in re.finditer(r"(?m)(?:^[^\n]*\|[^\n]*(?:\n|$)){2,}", text):
        block = match.group(0)
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) >= 2 and any(
            re.fullmatch(r"\|?[ :]?-{3,}[-| :]*\|?", line) for line in lines[1:3]
        ):
            tables.append(match.span())
    html_tables = [
        match.span() for match in re.finditer(r"(?is)<table\b.*?</table>", text)
    ]
    return sorted({*formulas, *tables, *html_tables})


def _split_structured_unit(
    text: str,
    start: int,
    end: int,
    tokenizer: OffsetTokenizer,
    size: int,
    overlap: int,
) -> list[tuple[int, int, int]]:
    """Split oversized Markdown tables at row boundaries; otherwise use tokens."""
    block = text[start:end]
    lines = list(re.finditer(r"[^\n]+(?:\n|$)", block))
    is_markdown_table = len(lines) >= 2 and all("|" in line.group(0) for line in lines)
    if not is_markdown_table:
        return _token_spans(text, start, end, tokenizer, size, overlap)
    result: list[tuple[int, int, int]] = []
    group_start = 0
    group_end = 0
    for line in lines:
        proposed_end = line.end()
        if group_end > group_start and tokenizer.count(block[group_start:proposed_end]) > size:
            absolute_start, absolute_end = start + group_start, start + group_end
            result.append(
                (absolute_start, absolute_end, tokenizer.count(text[absolute_start:absolute_end]))
            )
            group_start = line.start()
        group_end = proposed_end
    if group_end > group_start:
        absolute_start, absolute_end = start + group_start, start + group_end
        if tokenizer.count(text[absolute_start:absolute_end]) <= size:
            result.append(
                (absolute_start, absolute_end, tokenizer.count(text[absolute_start:absolute_end]))
            )
        else:
            result.extend(
                _token_spans(text, absolute_start, absolute_end, tokenizer, size, overlap)
            )
    return result


def build_chunking_strategy(
    name: str, *, tokenizer: OffsetTokenizer | None = None, embed_sentences=None
) -> ChunkingStrategy:
    tokenizer = tokenizer or TiktokenOffsetTokenizer()
    if name == "token-256-32":
        return TokenChunkingStrategy(tokenizer, 256, 32, name)
    if name == "token-384-64":
        return TokenChunkingStrategy(tokenizer, 384, 64, "token-384-64")
    if name == "token-512-64":
        return TokenChunkingStrategy(tokenizer, 512, 64, "token-512-64")
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
    if name == "section-aware-512-64":
        return SectionAwareChunkingStrategy(tokenizer)
    if name == "structure-aware-512-64":
        return StructureAwareChunkingStrategy(tokenizer)
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
