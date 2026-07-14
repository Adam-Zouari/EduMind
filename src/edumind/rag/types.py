"""Typed payloads for indexing, retrieval, and answers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias, TypeGuard

MetadataScalar: TypeAlias = str | int | float | bool
RawMetadata: TypeAlias = dict[str, object]
FilterMetadata: TypeAlias = dict[str, MetadataScalar]


def is_metadata_scalar(value: object) -> TypeGuard[MetadataScalar]:
    return isinstance(value, (str, int, float, bool))


def sanitize_filter_metadata(
    metadata: Mapping[str, object],
    *,
    source: str | None = None,
    format_type: str | None = None,
    file_path: str | None = None,
) -> FilterMetadata:
    sanitized = {key: value for key, value in metadata.items() if is_metadata_scalar(value)}
    if source:
        sanitized.setdefault("source", source)
    if format_type:
        sanitized.setdefault("format_type", format_type)
    if file_path:
        sanitized.setdefault("file_path", file_path)
    return sanitized


def build_source_id(
    *,
    text: str,
    source: str | None,
    file_path: str | None,
    format_type: str | None,
    metadata: Mapping[str, object],
) -> str:
    file_hash = metadata.get("source_checksum") or metadata.get("file_hash")
    identity = str(file_hash or file_path or source or text)
    return sha256(identity.encode("utf-8")).hexdigest()


def build_chunk_id(source_id: str, start: int, end: int, text: str) -> str:
    digest = sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}:{start}-{end}:{digest}"


@dataclass(frozen=True)
class IngestDocument:
    text: str
    source_id: str
    source: str
    format_type: str | None = None
    file_path: str | None = None
    metadata: RawMetadata = field(default_factory=dict)
    filter_metadata: FilterMetadata = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    source_id: str
    text: str
    chunk_index: int
    total_chunks: int
    start: int
    end: int
    token_count: int
    metadata: RawMetadata = field(default_factory=dict)
    filter_metadata: FilterMetadata = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass(frozen=True)
class RetrievalHit:
    id: str
    document: str
    metadata: RawMetadata
    score: float
    rank: int = 0
    retrieval_method: str = "dense"
    token_count: int = 0

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "Unknown"))

    @property
    def page(self) -> str:
        return str(self.metadata.get("page_number", self.metadata.get("page", "N/A")))


@dataclass(frozen=True)
class IngestReport:
    source_id: str
    source: str
    chunks_created: int
    chunks_replaced: int = 0
    elapsed_seconds: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryResult:
    query: str
    hits: tuple[RetrievalHit, ...]
    retrieval_seconds: float
    context_tokens: int


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RetrievalHit]
    context: str
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    prompt_tokens: int = 0
    answer_tokens: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VectorStoreSettings:
    collection_name: str
    persist_directory: Path
    distance_metric: str = "cosine"


# Compatibility settings remain small value objects for callers constructing components directly.
@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str
    embedding_dim: int
    device: str = "cpu"
    batch_size: int = 32


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int
    chunk_overlap: int
    separators: tuple[str, ...] = ("\n\n", "\n", " ", "")
