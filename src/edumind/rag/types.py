"""Typed contracts for the RAG subsystem."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import TypeAlias, TypeGuard

from edumind.common.paths import PROJECT_ROOT

MetadataScalar: TypeAlias = str | int | float | bool
RawMetadata: TypeAlias = dict[str, object]
FilterMetadata: TypeAlias = dict[str, MetadataScalar]

DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_BM25_ALPHA = 0.3
DEFAULT_OLLAMA_TIMEOUT = 120


def is_metadata_scalar(value: object) -> TypeGuard[MetadataScalar]:
    """Return whether a metadata value is safe for filtering and Chroma storage."""
    return isinstance(value, (str, int, float, bool))


def sanitize_filter_metadata(
    metadata: Mapping[str, object],
    *,
    source: str | None = None,
    format_type: str | None = None,
    file_path: str | None = None,
) -> FilterMetadata:
    """Extract top-level scalar metadata for queryable storage."""
    sanitized: FilterMetadata = {}
    for key, value in metadata.items():
        if is_metadata_scalar(value):
            sanitized[key] = value

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
    """Build a deterministic document identity for repeated ingestion."""
    identity_parts: list[str] = []
    file_hash = metadata.get("file_hash")
    if isinstance(file_hash, str) and file_hash:
        identity_parts.append(file_hash)

    for value in (file_path, source, format_type):
        if value:
            identity_parts.append(value)

    if not identity_parts:
        identity_parts.append(text.strip())

    payload = "|".join(identity_parts)
    return sha1(payload.encode("utf-8")).hexdigest()


def build_chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    """Build a deterministic chunk identity."""
    text_hash = sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{source_id}:{chunk_index}:{text_hash}"


@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding runtime settings."""

    model_name: str
    embedding_dim: int
    device: str = "cpu"
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE


@dataclass(frozen=True)
class ChunkingSettings:
    """Chunking settings."""

    chunk_size: int
    chunk_overlap: int
    separators: tuple[str, ...]

    @property
    def min_chunk_size(self) -> int:
        """Minimum target size before allowing early semantic breaks."""
        return min(500, self.chunk_size)


@dataclass(frozen=True)
class VectorStoreSettings:
    """Vector-store settings."""

    collection_name: str
    persist_directory: Path
    distance_metric: str
    bm25_alpha: float = DEFAULT_BM25_ALPHA


@dataclass(frozen=True)
class LLMSettings:
    """Ollama generation settings."""

    model_name: str
    base_url: str
    temperature: float
    max_tokens: int
    request_timeout: int = DEFAULT_OLLAMA_TIMEOUT


@dataclass(frozen=True)
class RAGSettings:
    """Retrieval settings."""

    top_k: int
    score_threshold: float


@dataclass(frozen=True)
class RAGConfig:
    """Typed RAG configuration bundle."""

    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    vector_store: VectorStoreSettings
    rag: RAGSettings
    llm: LLMSettings

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> RAGConfig:
        """Build typed config from the shared YAML mapping."""
        embedding_raw = _as_dict(data.get("embedding"))
        chunking_raw = _as_dict(data.get("chunking"))
        vectordb_raw = _as_dict(data.get("vectordb"))
        rag_raw = _as_dict(data.get("rag"))
        llm_raw = _as_dict(data.get("llm"))

        persist_directory = Path(
            str(vectordb_raw.get("persist_directory", "artifacts/rag/vector_store"))
        )
        if not persist_directory.is_absolute():
            persist_directory = (PROJECT_ROOT / persist_directory).resolve()

        return cls(
            embedding=EmbeddingSettings(
                model_name=str(
                    embedding_raw.get(
                        "model_name",
                        "sentence-transformers/all-MiniLM-L6-v2",
                    )
                ),
                embedding_dim=_coerce_int(embedding_raw.get("embedding_dim"), 384),
                device=str(embedding_raw.get("device", "cpu")),
                batch_size=_coerce_int(
                    embedding_raw.get("batch_size"),
                    DEFAULT_EMBED_BATCH_SIZE,
                ),
            ),
            chunking=ChunkingSettings(
                chunk_size=_coerce_int(chunking_raw.get("chunk_size"), 1000),
                chunk_overlap=_coerce_int(chunking_raw.get("chunk_overlap"), 200),
                separators=tuple(
                    str(separator)
                    for separator in _coerce_sequence(
                        chunking_raw.get("separators"),
                        ("\n\n", "\n", " ", ""),
                    )
                ),
            ),
            vector_store=VectorStoreSettings(
                collection_name=str(vectordb_raw.get("collection_name", "ocr_documents")),
                persist_directory=persist_directory,
                distance_metric=str(vectordb_raw.get("distance_metric", "cosine")),
            ),
            rag=RAGSettings(
                top_k=_coerce_int(rag_raw.get("top_k"), 5),
                score_threshold=_coerce_float(rag_raw.get("score_threshold"), 0.5),
            ),
            llm=LLMSettings(
                model_name=str(llm_raw.get("model_name", "qwen3:1.7b")),
                base_url=str(llm_raw.get("base_url", "http://localhost:11434")),
                temperature=_coerce_float(llm_raw.get("temperature"), 0.7),
                max_tokens=_coerce_int(llm_raw.get("max_tokens"), 2048),
            ),
        )


@dataclass(frozen=True)
class IngestDocument:
    """Normalized document ready for chunking and indexing."""

    text: str
    source_id: str
    source: str
    format_type: str | None = None
    file_path: str | None = None
    metadata: RawMetadata = field(default_factory=dict)
    filter_metadata: FilterMetadata = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    """Chunk payload stored in the retrieval system."""

    id: str
    source_id: str
    text: str
    chunk_index: int
    total_chunks: int
    metadata: RawMetadata = field(default_factory=dict)
    filter_metadata: FilterMetadata = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass(frozen=True)
class RetrievalHit:
    """Retrieved chunk plus normalized retrieval score."""

    id: str
    document: str
    metadata: RawMetadata
    score: float

    @property
    def source(self) -> str:
        value = self.metadata.get("source")
        return str(value) if value is not None else "Unknown"

    @property
    def page(self) -> str:
        value = self.metadata.get("page")
        return str(value) if value is not None else "N/A"


@dataclass(frozen=True)
class IngestReport:
    """Result of ingesting one or more documents."""

    source_id: str
    source: str
    chunks_created: int


@dataclass(frozen=True)
class AnswerResult:
    """Final answer payload returned by the RAG pipeline."""

    answer: str
    sources: list[RetrievalHit]
    context: str


def _as_dict(value: object) -> Mapping[str, object]:
    """Normalize nested config sections into mappings."""
    if isinstance(value, Mapping):
        return value
    return {}


def _coerce_int(value: object, default: int) -> int:
    """Normalize config values into integers."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: object, default: float) -> float:
    """Normalize config values into floats."""
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_sequence(value: object, default: tuple[str, ...]) -> tuple[object, ...]:
    """Normalize config sequences used for chunk separators."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return default
