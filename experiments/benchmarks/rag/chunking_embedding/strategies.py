"""Offset-preserving chunking candidates used only by the RAG experiment."""

from __future__ import annotations

import re
from dataclasses import dataclass

from edumind.common.artifacts import stable_hash
from edumind.rag.contracts import ChunkingStrategy
from edumind.rag.text_chunker import TokenChunkingStrategy
from edumind.rag.tokenizers import OffsetTokenizer, TiktokenOffsetTokenizer
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    ChunkingEmbeddingProtocol,
)


@dataclass(frozen=True)
class SentenceChunkingStrategy:
    tokenizer: OffsetTokenizer
    sentences: int
    overlap: int
    name: str

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
        if not text.strip():
            return []
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
            tokens = self.tokenizer.count(text[start:end])
            if tokens:
                chunks.append((start, end, tokens))
            if index + self.sentences >= len(sentence_spans):
                break
        return chunks


@dataclass(frozen=True)
class RecursiveCharacterChunkingStrategy:
    tokenizer: OffsetTokenizer
    size: int
    overlap: int
    name: str
    separators: tuple[str, ...]
    minimum_boundary_ratio: float

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "size": self.size,
                "overlap": self.overlap,
                "separators": self.separators,
                "minimum_boundary_ratio": self.minimum_boundary_ratio,
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        if not text.strip():
            return []
        chunks: list[tuple[int, int, int]] = []
        start = 0
        while start < len(text):
            target = min(start + self.size, len(text))
            end = target
            if target < len(text):
                for separator in self.separators:
                    boundary = text.rfind(separator, start, target)
                    if boundary > start + self.size * self.minimum_boundary_ratio:
                        end = boundary + len(separator)
                        break
            tokens = self.tokenizer.count(text[start:end])
            if tokens:
                chunks.append((start, end, tokens))
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
        boundary_embedding_fingerprint: str,
        maximum_tokens: int,
        percentile: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.embed_sentences = embed_sentences
        self.boundary_embedding_fingerprint = boundary_embedding_fingerprint
        self.maximum_tokens = maximum_tokens
        self.percentile = percentile

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "boundary_embedding_fingerprint": self.boundary_embedding_fingerprint,
                "maximum_tokens": self.maximum_tokens,
                "percentile": self.percentile,
            }
        )

    def split(self, text: str) -> list[tuple[int, int, int]]:
        import numpy as np

        if not text.strip():
            return []
        spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|\n|$)", text)]
        if len(spans) < 2:
            return self._bounded(text, 0, len(text))
        sentences = [text[start:end] for start, end in spans]
        vectors = np.asarray(self.embed_sentences(sentences), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(sentences):
            raise RuntimeError("Semantic sentence embeddings have an invalid shape")
        if not np.isfinite(vectors).all():
            raise RuntimeError("Semantic sentence embeddings contain non-finite values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("Semantic sentence embeddings contain zero-norm vectors")
        vectors /= norms
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
        if float(np.ptp(similarities)) <= np.finfo(similarities.dtype).eps:
            break_after: set[int] = set()
        else:
            threshold = float(np.quantile(similarities, self.percentile))
            break_after = {
                index for index, score in enumerate(similarities) if score <= threshold
            }
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
    size: int
    overlap: int
    name: str
    parser_identity: str

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "size": self.size,
                "overlap": self.overlap,
                "parser": self.parser_identity,
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
    size: int
    overlap: int
    name: str
    parser_identity: str

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "tokenizer": self.tokenizer.name,
                "size": self.size,
                "overlap": self.overlap,
                "structure_parser": self.parser_identity,
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
    return _merge_spans(sorted({*formulas, *tables, *html_tables}))


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


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
            result.extend(
                _bounded_span(
                    text, absolute_start, absolute_end, tokenizer, size, overlap
                )
            )
            group_start = line.start()
        group_end = proposed_end
    if group_end > group_start:
        absolute_start, absolute_end = start + group_start, start + group_end
        result.extend(
            _bounded_span(text, absolute_start, absolute_end, tokenizer, size, overlap)
        )
    return result


def _bounded_span(
    text: str,
    start: int,
    end: int,
    tokenizer: OffsetTokenizer,
    size: int,
    overlap: int,
) -> list[tuple[int, int, int]]:
    tokens = tokenizer.count(text[start:end])
    if tokens <= size:
        return [(start, end, tokens)] if tokens else []
    return _token_spans(text, start, end, tokenizer, size, overlap)


def build_chunking_strategy(
    name: str,
    *,
    protocol: ChunkingEmbeddingProtocol,
    tokenizer: OffsetTokenizer | None = None,
    embed_sentences=None,
    semantic_embedding_fingerprint: str | None = None,
) -> ChunkingStrategy:
    tokenizer_name = protocol.tokenizer.removeprefix("tiktoken:")
    tokenizer = tokenizer or TiktokenOffsetTokenizer(tokenizer_name)
    settings = protocol.strategy(name)
    kind = str(settings["kind"])
    if kind == "token":
        return TokenChunkingStrategy(
            tokenizer, int(settings["size"]), int(settings["overlap"]), name
        )
    if kind == "sentence":
        return SentenceChunkingStrategy(
            tokenizer, int(settings["sentences"]), int(settings["overlap"]), name
        )
    if kind == "recursive-character":
        return RecursiveCharacterChunkingStrategy(
            tokenizer,
            int(settings["size"]),
            int(settings["overlap"]),
            name,
            tuple(str(value) for value in settings["separators"]),
            float(settings["minimum_boundary_ratio"]),
        )
    if kind == "semantic":
        if embed_sentences is None:
            raise ValueError(
                "Semantic chunking requires the production sentence embedding function"
            )
        if not semantic_embedding_fingerprint:
            raise ValueError(
                "Semantic chunking requires the boundary-embedding fingerprint"
            )
        return SemanticChunkingStrategy(
            tokenizer,
            embed_sentences,
            semantic_embedding_fingerprint,
            int(settings["maximum_tokens"]),
            float(settings["boundary_percentile"]),
        )
    if kind == "section-aware":
        return SectionAwareChunkingStrategy(
            tokenizer,
            int(settings["size"]),
            int(settings["overlap"]),
            name,
            str(settings["parser"]),
        )
    if kind == "structure-aware":
        return StructureAwareChunkingStrategy(
            tokenizer,
            int(settings["size"]),
            int(settings["overlap"]),
            name,
            str(settings["parser"]),
        )
    raise ValueError(f"Unknown chunking strategy: {name}")


