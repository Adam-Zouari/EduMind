"""Shared candidate definitions for the staged benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.mlflow.vector_backends import VectorBackendSpec

DEFAULT_TOP_K = 5
DEFAULT_RANDOM_SEED = 7


@dataclass(frozen=True)
class ChunkingCandidate:
    """Chunking strategy candidate for Stage 1."""

    name: str
    description: str
    kind: str
    chunk_size: int
    chunk_overlap: int
    child_size: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingCandidate:
    """Embedding-model candidate for Stage 2."""

    model_name: str
    embedding_dim: int
    description: str

    @property
    def name(self) -> str:
        return self.model_name.split("/")[-1].replace(":", "_").replace(".", "_")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalStrategyCandidate:
    """Retrieval strategy candidate for Stage 4."""

    name: str
    mode: str
    bm25_alpha: float
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SLMCandidate:
    """Local language model candidate for Stage 5."""

    model_name: str
    description: str

    @property
    def name(self) -> str:
        return self.model_name.replace(":", "_").replace(".", "_")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalStackCandidate:
    """Full retrieval stack promoted out of Stage 4."""

    chunker_name: str
    embedding_model: str
    vector_backend: str
    retrieval_name: str
    bm25_alpha: float

    @property
    def name(self) -> str:
        return (
            f"{self.chunker_name}__{self.embedding_model.split('/')[-1]}__"
            f"{self.vector_backend}__{self.retrieval_name}"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FullStackCandidate:
    """Final promoted stack including the answer model."""

    retrieval_stack: RetrievalStackCandidate
    llm_model: str

    @property
    def name(self) -> str:
        return f"{self.retrieval_stack.name}__{self.llm_model.replace(':', '_').replace('.', '_')}"

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval_stack": self.retrieval_stack.to_dict(),
            "llm_model": self.llm_model,
        }


CHUNKING_CANDIDATES = (
    ChunkingCandidate(
        name="token_256_32",
        description="256-token windows with 32-token overlap",
        kind="token",
        chunk_size=256,
        chunk_overlap=32,
    ),
    ChunkingCandidate(
        name="token_512_64",
        description="512-token windows with 64-token overlap",
        kind="token",
        chunk_size=512,
        chunk_overlap=64,
    ),
    ChunkingCandidate(
        name="token_1024_128",
        description="1024-token windows with 128-token overlap",
        kind="token",
        chunk_size=1024,
        chunk_overlap=128,
    ),
    ChunkingCandidate(
        name="sentence_window",
        description="Sentence windows with sentence overlap",
        kind="sentence_window",
        chunk_size=10,
        chunk_overlap=2,
    ),
    ChunkingCandidate(
        name="semantic_chunker",
        description="Semantic chunker using the shared RAG implementation",
        kind="semantic",
        chunk_size=1000,
        chunk_overlap=200,
    ),
    ChunkingCandidate(
        name="hierarchical",
        description="Parent/child fixed windows",
        kind="hierarchical",
        chunk_size=2000,
        chunk_overlap=0,
        child_size=500,
    ),
)

EMBEDDING_CANDIDATES = (
    EmbeddingCandidate(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        description="Lightweight English baseline",
    ),
    EmbeddingCandidate(
        model_name="sentence-transformers/all-mpnet-base-v2",
        embedding_dim=768,
        description="Balanced English encoder",
    ),
    EmbeddingCandidate(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        embedding_dim=768,
        description="Question-answer retrieval encoder",
    ),
    EmbeddingCandidate(
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
        description="Compact BGE English encoder",
    ),
    EmbeddingCandidate(
        model_name="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
        description="Larger English BGE baseline",
    ),
)

VECTOR_BACKEND_CANDIDATES = (
    VectorBackendSpec("chroma", "Current Chroma vector store"),
    VectorBackendSpec("qdrant", "Qdrant local persistence"),
    VectorBackendSpec("lancedb", "LanceDB local persistence"),
)

RETRIEVAL_STRATEGY_CANDIDATES = (
    RetrievalStrategyCandidate(
        name="dense_only",
        mode="dense_only",
        bm25_alpha=0.0,
        description="Dense retrieval only",
    ),
    RetrievalStrategyCandidate(
        name="bm25_only",
        mode="bm25_only",
        bm25_alpha=1.0,
        description="Lexical BM25 retrieval only",
    ),
    RetrievalStrategyCandidate(
        name="hybrid_0_25",
        mode="hybrid",
        bm25_alpha=0.25,
        description="Hybrid retrieval with 25% BM25 weight",
    ),
    RetrievalStrategyCandidate(
        name="hybrid_0_50",
        mode="hybrid",
        bm25_alpha=0.50,
        description="Hybrid retrieval with 50% BM25 weight",
    ),
    RetrievalStrategyCandidate(
        name="hybrid_0_75",
        mode="hybrid",
        bm25_alpha=0.75,
        description="Hybrid retrieval with 75% BM25 weight",
    ),
)

SLM_CANDIDATES = (
    SLMCandidate("qwen3:1.7b", "Qwen 3 1.7B"),
    SLMCandidate("qwen2.5:3b", "Qwen 2.5 3B"),
    SLMCandidate("llama3.2:3b", "Llama 3.2 3B"),
    SLMCandidate("phi3:mini", "Phi 3 Mini"),
    SLMCandidate("gemma3:4b", "Gemma 3 4B"),
)
