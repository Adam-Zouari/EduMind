"""Embedding contracts that remain experimental until benchmark promotion."""

from __future__ import annotations

from dataclasses import asdict

from edumind.rag.contracts import EmbeddingSpec, embedding_spec as production_embedding_spec

EXPERIMENTAL_EMBEDDING_SPECS: dict[str, EmbeddingSpec] = {
    "Snowflake/snowflake-arctic-embed-m-v2.0": EmbeddingSpec(
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "from-lock",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        None,
        "",
        "",
        True,
        768,
        "cosine",
        8192,
        "cls",
        query_prompt_name="query",
        trust_remote_code=True,
    ),
    "codefuse-ai/F2LLM-v2-0.6B": EmbeddingSpec(
        "codefuse-ai/F2LLM-v2-0.6B",
        "from-lock",
        "codefuse-ai/F2LLM-v2-0.6B",
        None,
        "",
        "",
        True,
        1024,
        "cosine",
        32768,
        "last-token",
        interface="query-document",
    ),
    "Octen/Octen-Embedding-0.6B": EmbeddingSpec(
        "Octen/Octen-Embedding-0.6B",
        "from-lock",
        "Octen/Octen-Embedding-0.6B",
        None,
        "",
        "",
        True,
        1024,
        "cosine",
        32768,
        "last-token",
    ),
    "Qwen/Qwen3-Embedding-0.6B": EmbeddingSpec(
        "Qwen/Qwen3-Embedding-0.6B",
        "from-lock",
        "Qwen/Qwen3-Embedding-0.6B",
        None,
        "Instruct: Retrieve relevant educational evidence\nQuery:",
        "",
        True,
        1024,
        "cosine",
        32768,
        "last-token",
    ),
    "nvidia/Nemotron-3-Embed-1B-BF16": EmbeddingSpec(
        "nvidia/Nemotron-3-Embed-1B-BF16",
        "from-lock",
        "nvidia/Nemotron-3-Embed-1B-BF16",
        None,
        "",
        "",
        True,
        2048,
        "cosine",
        32768,
        "mean",
        interface="query-document",
    ),
    "Octen/Octen-Embedding-4B": EmbeddingSpec(
        "Octen/Octen-Embedding-4B",
        "from-lock",
        "Octen/Octen-Embedding-4B",
        None,
        "",
        "",
        True,
        2560,
        "cosine",
        32768,
        "last-token",
    ),
    "Qwen/Qwen3-Embedding-4B": EmbeddingSpec(
        "Qwen/Qwen3-Embedding-4B",
        "from-lock",
        "Qwen/Qwen3-Embedding-4B",
        None,
        "Instruct: Retrieve relevant educational evidence\nQuery:",
        "",
        True,
        2560,
        "cosine",
        32768,
        "last-token",
    ),
}


def embedding_spec(
    name: str,
    *,
    revision: str,
    local_path: str,
    document_device: str = "cpu",
    query_device: str = "cpu",
) -> EmbeddingSpec:
    if name == "sentence-transformers/all-MiniLM-L6-v2":
        return production_embedding_spec(
            name,
            revision=revision,
            local_path=local_path,
            document_device=document_device,
            query_device=query_device,
        )
    try:
        spec = EXPERIMENTAL_EMBEDDING_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"No experimental embedding contract for model: {name}") from exc
    return EmbeddingSpec(
        **{
            **asdict(spec),
            "revision": revision,
            "local_path": local_path,
            "document_device": document_device,
            "query_device": query_device,
        }
    )
