"""Shared exact-search evaluator for chunking and retrieval experiments."""

from __future__ import annotations

import time
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from edumind.common.artifacts import stable_hash
from edumind.rag.embedder import Embedder
from edumind.rag.contracts import EmbeddingSpec
from edumind.rag.tokenizers import OffsetTokenizer, TiktokenOffsetTokenizer

from experiments.benchmarks.common.datasets import EvidenceInterval, evidence_units
from experiments.benchmarks.common.metrics import (
    average_precision_at_k,
    context_precision_at_k,
    context_recall,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance_grades,
    reciprocal_rank,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import embedding_spec
from experiments.benchmarks.rag.chunking_embedding.strategies import build_chunking_strategy
from experiments.benchmarks.rag.methods import BM25, Reranker, reciprocal_rank_fusion
from experiments.benchmarks.rag.retrieval.protocol import (
    RetrievalProtocol,
    default_protocol,
)


@dataclass(frozen=True)
class Chunk:
    identifier: str
    document_id: str
    text: str
    start: int
    end: int
    tokens: int
    model_tokens: int | None = None


@dataclass
class ExactIndex:
    chunks: list[Chunk]
    vectors: np.ndarray | None
    embedder: Embedder | None
    tokenizer: OffsetTokenizer
    bm25: BM25 | None
    embedding_spec: EmbeddingSpec
    chunking_fingerprint: str
    chunking_contract: Mapping[str, object]
    corpus_build_seconds: float
    preflight: Mapping[str, object]


class InputCompatibilityError(RuntimeError):
    def __init__(self, message: str, report: Mapping[str, object]) -> None:
        super().__init__(message)
        self.report = dict(report)


def build_index(
    manifest,
    chunker_name,
    embedding_name,
    model_lock,
    with_bm25=True,
    with_dense=True,
    retrieval_protocol: RetrievalProtocol | None = None,
) -> ExactIndex:
    protocol = retrieval_protocol or default_protocol()
    entry = model_lock[embedding_name]
    revision = str(entry["revision"])
    local_path = str(entry["model_path"])
    device = os.environ.get("EDUMIND_BENCHMARK_EMBEDDING_DEVICE", "cpu").strip() or "cpu"
    spec = embedding_spec(
        embedding_name,
        revision=revision,
        local_path=local_path,
        document_device=device,
        query_device=device,
    )
    dtype = os.environ.get("EDUMIND_BENCHMARK_EMBEDDING_DTYPE", "float32")
    embedder = (
        Embedder(
            spec,
            dtype=dtype,
            batch_size=protocol.embedding_batch_size,
            enforce_device=True,
        )
        if with_dense or chunker_name == "semantic"
        else None
    )
    tokenizer = TiktokenOffsetTokenizer("cl100k_base")
    if embedder is not None:
        embedder.prepare()
    build_started = time.perf_counter()
    input_preflight_seconds = 0.0
    semantic_sentence_counts: list[int] = []

    def input_counts(texts: Sequence[str], *, role: str) -> list[int]:
        nonlocal input_preflight_seconds
        started = time.perf_counter()
        try:
            if embedder is None:
                raise RuntimeError("Model-input validation requires an embedding runtime")
            return _model_input_counts(embedder, texts, role=role)
        finally:
            input_preflight_seconds += time.perf_counter() - started

    def embed_sentences(sentences: Sequence[str]):
        if embedder is None:
            raise RuntimeError("Semantic chunking requires an embedding runtime")
        counts = input_counts(sentences, role="document")
        semantic_sentence_counts.extend(counts)
        offending = [count for count in counts if count > spec.maximum_length]
        if offending:
            report = {
                "status": "failed",
                "reason_code": "input_length_exceeded",
                "maximum_length": spec.maximum_length,
                "semantic_sentence_input_count": len(semantic_sentence_counts),
                "maximum_semantic_sentence_tokens": max(
                    semantic_sentence_counts, default=0
                ),
                "offending_semantic_sentence_count": len(offending),
                "truncated_inputs": 0,
                "input_preflight_seconds_excluded_from_corpus_build": (
                    input_preflight_seconds
                ),
                "chunking_contract": _chunking_contract(chunker),
                "resolved_document_input": embedder.input_configuration(
                    "document"
                ),
                "resolved_query_input": embedder.input_configuration("query"),
            }
            raise InputCompatibilityError(
                f"{chunker_name}|{embedding_name} has semantic sentence inputs beyond "
                f"the model's {spec.maximum_length}-token contract",
                report,
            )
        return embedder.embed_texts(sentences)

    chunker = build_chunking_strategy(
        chunker_name,
        tokenizer=tokenizer,
        embed_sentences=embed_sentences if embedder is not None else None,
        semantic_embedding_fingerprint=stable_hash(
            {
                "embedding_contract": {
                    name: value
                    for name, value in vars(spec).items()
                    if name != "local_path"
                },
                "model_cache_manifest_sha256": entry.get(
                    "model_cache_manifest_sha256"
                ),
                "dtype": dtype,
                "batch_size": embedder.batch_size if embedder is not None else None,
            }
        ),
    )
    chunks: list[Chunk] = []
    documents = [row for row in manifest.samples if row.get("kind") == "document"]
    for document in documents:
        text = str(document["text"])
        for start, end, tokens in chunker.split(text):
            if not (0 <= start < end <= len(text)):
                raise RuntimeError(
                    f"Chunker {chunker_name} returned invalid offsets for {document['id']}"
                )
            actual_tokens = tokenizer.count(text[start:end])
            if tokens != actual_tokens:
                raise RuntimeError(
                    f"Chunker {chunker_name} token-count mismatch for {document['id']}: "
                    f"reported {tokens}, counted {actual_tokens}"
                )
            chunks.append(
                Chunk(
                    f"{document['id']}:{start}:{end}",
                    str(document["id"]),
                    text[start:end],
                    start,
                    end,
                    tokens,
                )
            )
    if not chunks:
        raise RuntimeError("Chunking produced an empty corpus")
    chunk_ids = [chunk.identifier for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError(f"Chunker {chunker_name} produced duplicate stable IDs")
    questions = [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("answerable")
    ]
    preflight: dict[str, object] = {
        "maximum_length": spec.maximum_length,
        "document_inputs": len(chunks),
        "query_inputs": len(questions),
        "truncated_inputs": 0,
        "chunking_contract": _chunking_contract(chunker),
        "semantic_sentence_input_count": len(semantic_sentence_counts),
        "maximum_semantic_sentence_tokens": max(semantic_sentence_counts, default=0),
    }
    if embedder is not None:
        document_counts = input_counts(
            [chunk.text for chunk in chunks], role="document"
        )
        query_counts = input_counts(
            [str(question["question"]) for question in questions], role="query"
        )
        chunks = [
            replace(chunk, model_tokens=count)
            for chunk, count in zip(chunks, document_counts, strict=True)
        ]
        preflight.update(
            {
                "maximum_document_tokens": max(document_counts, default=0),
                "maximum_query_tokens": max(query_counts, default=0),
                "document_token_counts": {
                    chunk.identifier: count
                    for chunk, count in zip(chunks, document_counts, strict=True)
                },
                "query_token_counts": {
                    str(question["id"]): count
                    for question, count in zip(questions, query_counts, strict=True)
                },
                "resolved_query_input": embedder.input_configuration("query"),
                "resolved_document_input": embedder.input_configuration("document"),
                "input_preflight_seconds_excluded_from_corpus_build": (
                    input_preflight_seconds
                ),
            }
        )
        offending_documents = [
            {
                "chunk_id": chunk.identifier,
                "model_tokens": chunk.model_tokens,
            }
            for chunk in chunks
            if chunk.model_tokens is not None and chunk.model_tokens > spec.maximum_length
        ]
        offending_queries = [
            {
                "question_id": str(question["id"]),
                "model_tokens": count,
            }
            for question, count in zip(questions, query_counts, strict=True)
            if count > spec.maximum_length
        ]
        if offending_documents or offending_queries:
            preflight.update(
                {
                    "status": "failed",
                    "reason_code": "input_length_exceeded",
                    "offending_document_inputs": offending_documents[:20],
                    "offending_query_inputs": offending_queries[:20],
                    "offending_document_count": len(offending_documents),
                    "offending_query_count": len(offending_queries),
                }
            )
            raise InputCompatibilityError(
                f"{chunker_name}|{embedding_name} exceeds the model's "
                f"{spec.maximum_length}-token input contract",
                preflight,
            )
    preflight["status"] = "passed"
    vectors = (
        np.asarray(embedder.embed_texts([chunk.text for chunk in chunks]))
        if with_dense and embedder is not None
        else None
    )
    if vectors is not None:
        _validate_vectors(vectors, len(chunks), spec.dimension, "document")
        vectors = vectors.astype(np.float32, copy=False)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    bm25 = (
        BM25(
            [chunk.text for chunk in chunks],
            k1=protocol.bm25_k1,
            b=protocol.bm25_b,
            epsilon=protocol.bm25_epsilon,
        )
        if with_bm25
        else None
    )
    corpus_build_seconds = max(
        0.0, time.perf_counter() - build_started - input_preflight_seconds
    )
    return ExactIndex(
        chunks,
        vectors,
        embedder,
        tokenizer,
        bm25,
        spec,
        chunker.fingerprint,
        _chunking_contract(chunker),
        corpus_build_seconds,
        preflight,
    )


def rank(
    index: ExactIndex,
    query: str,
    method: str,
    reranker: Reranker | None = None,
    *,
    retrieval_protocol: RetrievalProtocol | None = None,
) -> list[int]:
    protocol = retrieval_protocol or default_protocol()
    retriever = method.split("|", 1)[0]
    if retriever == "bm25":
        if index.bm25 is None:
            raise RuntimeError("This index was built without BM25")
        base = [
            identifier
            for identifier, _ in index.bm25.rank(query, protocol.pool_size)
        ]
        if reranker is None:
            return base
        local_order = reranker.rank(
            query, [index.chunks[position].text for position in base]
        )
        return [base[position] for position in local_order]

    if index.vectors is None or index.embedder is None:
        raise RuntimeError("This index was built without dense vectors")
    dense = [
        position
        for position, _ in dense_rank_with_scores(index, query, protocol.pool_size)
    ]
    if retriever == "dense":
        base = dense
    else:
        if retriever != "rrf":
            raise ValueError(f"Unsupported retrieval method: {method}")
        if index.bm25 is None:
            raise RuntimeError("This index was built without BM25")
        lexical = [
            identifier
            for identifier, _ in index.bm25.rank(query, protocol.pool_size)
        ]
        base = reciprocal_rank_fusion(
            [dense, lexical],
            protocol.pool_size,
            protocol.rrf_constant,
            weights=protocol.rrf_weights,
            tie_keys={
                position: chunk.identifier
                for position, chunk in enumerate(index.chunks)
            },
        )
    if reranker is None:
        return base
    local_order = reranker.rank(
        query, [index.chunks[position].text for position in base]
    )
    return [base[position] for position in local_order]


def dense_rank_with_scores(
    index: ExactIndex, query: str, limit: int | None = None
) -> list[tuple[int, float]]:
    """Exact cosine ranking with stable corpus-order tie-breaking."""

    if index.vectors is None or index.embedder is None:
        raise RuntimeError("This index was built without dense vectors")
    limit = default_protocol().pool_size if limit is None else limit
    query_vector = np.asarray(index.embedder.embed_query(query), dtype=np.float32)
    if query_vector.ndim != 1:
        raise RuntimeError("Query embedding must be one-dimensional")
    _validate_vectors(query_vector.reshape(1, -1), 1, index.embedding_spec.dimension, "query")
    query_norm = float(np.linalg.norm(query_vector))
    scores = index.vectors @ (query_vector / query_norm)
    positions = np.arange(len(scores))
    order = np.lexsort((positions, -scores))[:limit]
    return [(int(position), float(scores[position])) for position in order]


def _model_input_counts(
    embedder: Embedder, texts: Sequence[str], *, role: str
) -> list[int]:
    counts: list[int] = []
    for start in range(0, len(texts), 256):
        counts.extend(embedder.input_token_counts(texts[start : start + 256], role=role))
    return counts


def _validate_vectors(
    vectors: np.ndarray, expected_rows: int, expected_dimension: int, label: str
) -> None:
    if vectors.shape != (expected_rows, expected_dimension):
        raise RuntimeError(
            f"{label.title()} embedding shape mismatch: expected "
            f"{(expected_rows, expected_dimension)}, received {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError(f"{label.title()} embeddings contain non-finite values")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise RuntimeError(f"{label.title()} embeddings contain zero-norm vectors")


def _chunking_contract(chunker) -> dict[str, object]:
    names = (
        "name",
        "size",
        "overlap",
        "sentences",
        "maximum_tokens",
        "percentile",
        "boundary_embedding_fingerprint",
        "separators",
        "minimum_boundary_ratio",
    )
    return {
        **{
            name: list(value) if isinstance(value, tuple) else value
            for name in names
            if isinstance(
                (value := getattr(chunker, name, None)), (str, int, float, tuple)
            )
        },
        "tokenizer": str(chunker.tokenizer.name),
        "fingerprint": str(chunker.fingerprint),
    }


def retrieval_metrics(question, selected, all_chunks, _tokenizer) -> tuple[dict[str, float], int]:
    evidence = [
        interval
        for unit in evidence_units(question)
        for interval in unit.intervals
    ]
    all_grades = [_grade(chunk, evidence) for chunk in all_chunks]
    grades = [_grade(chunk, evidence) for chunk in selected]
    relevant_total = sum(score > 0 for score in all_grades)
    gold = [(interval.start, interval.end) for interval in evidence]
    metrics = {"mrr": reciprocal_rank(grades)}
    for cutoff in (1, 3, 5, 10):
        intervals = [
            (chunk.start, chunk.end)
            for chunk in selected[:cutoff]
            if chunk.document_id == str(question["document_id"])
        ]
        metrics.update(
            {
                f"precision_at_{cutoff}": precision_at_k(grades, cutoff),
                f"recall_at_{cutoff}": recall_at_k(grades, relevant_total, cutoff),
                f"hit_rate_at_{cutoff}": hit_rate_at_k(grades, cutoff),
                f"context_precision_at_{cutoff}": context_precision_at_k(grades, cutoff),
                f"context_recall_at_{cutoff}": context_recall(gold, intervals),
            }
        )
        if cutoff in {3, 5, 10}:
            metrics[f"map_at_{cutoff}"] = average_precision_at_k(grades, relevant_total, cutoff)
            metrics[f"ndcg_at_{cutoff}"] = ndcg_at_k(grades, cutoff, all_grades)
    return metrics, sum(chunk.tokens for chunk in selected)


def _grade(chunk: Chunk, evidence: Sequence[EvidenceInterval]) -> float:
    matching = [
        (interval.start, interval.end)
        for interval in evidence
        if chunk.document_id == interval.document_id
    ]
    return relevance_grades(matching, [(chunk.start, chunk.end)])[0]


def reranker_for(
    method: str,
    model_lock: Mapping[str, Mapping[str, object]],
    *,
    device: str = "cpu",
    dtype: str = "float32",
    retrieval_protocol: RetrievalProtocol | None = None,
) -> Reranker | None:
    from experiments.benchmarks.rag.retrieval.profiles import RERANKER_MODELS

    reranker_name = method.split("|", 1)[1] if "|" in method else method
    model = RERANKER_MODELS.get(reranker_name)
    if model is None:
        return None
    protocol = retrieval_protocol or default_protocol()
    entry = model_lock[model]
    return Reranker(
        model,
        str(entry["revision"]),
        str(entry["model_path"]),
        device=device,
        dtype=dtype,
        batch_size=protocol.reranker_batch_size,
        maximum_length=protocol.maximum_tokens(reranker_name),
    )


RETRIEVAL_QUALITY_DIRECTIONS = {
    "mrr": "max",
    "precision_at_1": "max",
    "precision_at_3": "max",
    "precision_at_5": "max",
    "precision_at_10": "max",
    "recall_at_1": "max",
    "recall_at_3": "max",
    "recall_at_5": "max",
    "recall_at_10": "max",
    "hit_rate_at_1": "max",
    "hit_rate_at_3": "max",
    "hit_rate_at_5": "max",
    "hit_rate_at_10": "max",
    "map_at_3": "max",
    "map_at_5": "max",
    "map_at_10": "max",
    "ndcg_at_3": "max",
    "ndcg_at_5": "max",
    "ndcg_at_10": "max",
    "context_precision_at_1": "max",
    "context_recall_at_3": "max",
    "context_recall_at_5": "max",
    "context_precision_at_3": "max",
    "context_precision_at_5": "max",
    "context_precision_at_10": "max",
    "context_recall_at_1": "max",
    "context_recall_at_10": "max",
    "determinism": "max",
}
