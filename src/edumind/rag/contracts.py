"""Public RAG strategy and persistence contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from edumind.common.artifacts import stable_hash

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
    interface: str = "encode"
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    trust_remote_code: bool = False
    encode_options: tuple[tuple[str, float], ...] = ()

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
    "google/embeddinggemma-300m": EmbeddingSpec(
        "google/embeddinggemma-300m",
        "main",
        "google/embeddinggemma-300m",
        "task: search result | query: ",
        "title: none | text: ",
        True,
        768,
        "cosine",
        2048,
        interface="query-document",
    ),
    "infgrad/Jasper-Token-Compression-600M": EmbeddingSpec(
        "infgrad/Jasper-Token-Compression-600M",
        "main",
        "infgrad/Jasper-Token-Compression-600M",
        "",
        "",
        True,
        1024,
        "cosine",
        1024,
        query_prompt_name="query",
        trust_remote_code=True,
        encode_options=(("compression_ratio", 0.3333),),
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
    "nvidia/Nemotron-3-Embed-1B-BF16": EmbeddingSpec(
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "main",
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "query: ",
        "passage: ",
        True,
        2048,
        "cosine",
        32768,
        interface="query-document",
    ),
    "Qwen/Qwen3-Embedding-4B": EmbeddingSpec(
        "Qwen/Qwen3-Embedding-4B",
        "main",
        "Qwen/Qwen3-Embedding-4B",
        "Instruct: Retrieve relevant educational evidence\nQuery: ",
        "",
        True,
        2560,
        "cosine",
        32768,
    ),
    "nvidia/Nemotron-3-Embed-8B-BF16": EmbeddingSpec(
        "nvidia/Nemotron-3-Embed-8B-BF16",
        "main",
        "nvidia/Nemotron-3-Embed-8B-BF16",
        "query: ",
        "passage: ",
        True,
        4096,
        "cosine",
        32768,
        interface="query-document",
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
    embedding_contract: str
    chunking_contract: str
    backend: str
    collection_name: str

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


@runtime_checkable
class ChunkingStrategy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def split(self, text: str) -> list[tuple[int, int, int]]:
        """Return half-open character spans plus token counts."""


EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]
