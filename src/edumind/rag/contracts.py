"""Public RAG strategy and persistence contracts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from edumind.common.artifacts import stable_hash

if TYPE_CHECKING:
    from .types import RetrievalHit


@dataclass(frozen=True)
class EmbeddingSpec:
    model_name: str
    revision: str
    tokenizer: str
    query_prefix: str
    document_prefix: str
    normalize: bool
    dimension: int
    similarity: str
    maximum_length: int
    document_device: str = "cpu"
    query_device: str = "cpu"

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


EMBEDDING_SPECS: dict[str, EmbeddingSpec] = {
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingSpec(
        "sentence-transformers/all-MiniLM-L6-v2",
        "main",
        "sentence-transformers/all-MiniLM-L6-v2",
        "",
        "",
        True,
        384,
        "cosine",
        256,
    ),
    "BAAI/bge-base-en-v1.5": EmbeddingSpec(
        "BAAI/bge-base-en-v1.5",
        "main",
        "BAAI/bge-base-en-v1.5",
        "Represent this sentence for searching relevant passages: ",
        "",
        True,
        768,
        "cosine",
        512,
    ),
    "nomic-ai/nomic-embed-text-v1.5": EmbeddingSpec(
        "nomic-ai/nomic-embed-text-v1.5",
        "main",
        "nomic-ai/nomic-embed-text-v1.5",
        "search_query: ",
        "search_document: ",
        True,
        768,
        "cosine",
        8192,
    ),
    "Qwen/Qwen3-Embedding-0.6B": EmbeddingSpec(
        "Qwen/Qwen3-Embedding-0.6B",
        "main",
        "Qwen/Qwen3-Embedding-0.6B",
        "Instruct: Retrieve relevant educational evidence\nQuery: ",
        "",
        True,
        1024,
        "cosine",
        32768,
    ),
}


def embedding_spec(
    name: str,
    *,
    document_device: str = "cpu",
    query_device: str = "cpu",
    revision: str | None = None,
) -> EmbeddingSpec:
    try:
        spec = EMBEDDING_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"No audited embedding contract for model: {name}") from exc
    return EmbeddingSpec(
        **{
            **asdict(spec),
            "revision": revision or spec.revision,
            "document_device": document_device,
            "query_device": query_device,
        }
    )


@dataclass(frozen=True)
class GenerationProfile:
    model_name: str
    digest: str
    thinking: str
    temperature: float
    seed: int
    context_tokens: int
    maximum_answer_tokens: int
    keep_alive: str | int = "5m"

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int
    content_checksum: str
    embedding_contract: str
    chunking_contract: str
    backend: str
    collection_name: str
    filter_fields: tuple[str, ...] = ()

    @property
    def compatibility_key(self) -> str:
        return stable_hash(
            {
                "schema_version": self.schema_version,
                "embedding_contract": self.embedding_contract,
                "chunking_contract": self.chunking_contract,
                "backend": self.backend,
                "collection_name": self.collection_name,
            }
        )


@dataclass(frozen=True)
class RecommendationManifest:
    schema_version: int
    status: str
    benchmark_run_ids: tuple[str, ...]
    extraction: Mapping[str, str]
    rag: Mapping[str, str]
    authoritative: bool
    reason: str


def load_recommendation_manifest() -> RecommendationManifest:
    resource = files("edumind").joinpath("recommendations/default.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    manifest = RecommendationManifest(
        schema_version=int(payload["schema_version"]),
        status=str(payload["status"]),
        benchmark_run_ids=tuple(str(value) for value in payload.get("benchmark_run_ids", [])),
        extraction=dict(payload.get("extraction", {})),
        rag=dict(payload.get("rag", {})),
        authoritative=bool(payload.get("authoritative", False)),
        reason=str(payload.get("reason", "")),
    )
    if manifest.authoritative and not manifest.benchmark_run_ids:
        raise ValueError("An authoritative recommendation requires benchmark run IDs")
    return manifest


@runtime_checkable
class ChunkingStrategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def split(self, text: str) -> list[tuple[int, int, int]]:
        """Return half-open character spans plus token counts."""


@runtime_checkable
class RetrievalStrategy(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        query_embedding: Sequence[float],
        limit: int,
        filters: Mapping[str, object] | None = None,
    ): ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, hits: Sequence[RetrievalHit], limit: int
    ) -> list[RetrievalHit]: ...


EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]
