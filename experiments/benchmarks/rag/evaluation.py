"""Shared exact-search evaluator for chunking and retrieval experiments."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from edumind.rag.contracts import embedding_spec
from edumind.rag.embedder import Embedder
from edumind.rag.text_chunker import build_chunking_strategy
from edumind.rag.tokenizers import HuggingFaceOffsetTokenizer

from experiments.benchmarks.common.contracts import DatasetManifest, SampleResult
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
from experiments.benchmarks.rag.methods import BM25, Reranker, reciprocal_rank_fusion


@dataclass(frozen=True)
class Chunk:
    identifier: str
    document_id: str
    text: str
    start: int
    end: int
    tokens: int


@dataclass
class ExactIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    embedder: Embedder
    tokenizer: HuggingFaceOffsetTokenizer
    bm25: BM25 | None


def evaluate(
    manifest: DatasetManifest,
    chunker_name: str,
    embedding_name: str,
    retrieval_name: str,
    revisions: Mapping[str, str],
    repetitions: int = 1,
) -> tuple[list[SampleResult], Mapping[str, float]]:
    indexed_at = time.perf_counter()
    index = build_index(manifest, chunker_name, embedding_name, revisions, retrieval_name != "dense")
    indexing_seconds = time.perf_counter() - indexed_at
    reranker = reranker_for(retrieval_name, revisions)
    questions = [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("answerable") and row.get("evidence")
    ]
    random.Random(42).shuffle(questions)
    samples: list[SampleResult] = []
    latencies: list[float] = []
    for question in questions:
        orders: list[list[int]] = []
        item_latencies: list[float] = []
        for _ in range(repetitions):
            started = time.perf_counter()
            orders.append(rank(index, str(question["question"]), retrieval_name, reranker))
            item_latencies.append(time.perf_counter() - started)
        order = orders[0]
        selected = [index.chunks[position] for position in order[:10]]
        latency = float(np.median(item_latencies))
        latencies.extend(item_latencies)
        metrics, retrieved_tokens = retrieval_metrics(question, selected, index.chunks, index.tokenizer)
        metrics["determinism"] = float(all(candidate == order for candidate in orders))
        samples.append(
            SampleResult(
                str(question["id"]),
                metrics,
                latency,
                {"retrieved_tokens": retrieved_tokens, "measured_repetitions": repetitions},
            )
        )
    token_counts = np.asarray([chunk.tokens for chunk in index.chunks], dtype=float)
    return samples, {
        "indexing_seconds": indexing_seconds,
        "chunk_count": float(len(index.chunks)),
        "mean_chunk_tokens": float(token_counts.mean()),
        "p95_chunk_tokens": float(np.quantile(token_counts, 0.95)),
        "p50_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
        "storage_bytes": float(index.vectors.nbytes),
    }


def build_index(manifest, chunker_name, embedding_name, revisions, with_bm25=True) -> ExactIndex:
    revision = revisions[embedding_name]
    spec = embedding_spec(
        embedding_name, revision=revision, document_device="cpu", query_device="cpu"
    )
    embedder = Embedder(spec=spec)
    tokenizer = HuggingFaceOffsetTokenizer(spec.tokenizer, revision=revision)
    chunker = build_chunking_strategy(
        chunker_name, tokenizer=tokenizer, embed_sentences=embedder.embed_texts
    )
    chunks: list[Chunk] = []
    for document in (row for row in manifest.samples if row.get("kind") == "document"):
        text = str(document["text"])
        for index, (start, end, tokens) in enumerate(chunker.split(text)):
            chunks.append(
                Chunk(f"{document['id']}:{index}", str(document["id"]), text[start:end], start, end, tokens)
            )
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    return ExactIndex(
        chunks,
        vectors,
        embedder,
        tokenizer,
        BM25([chunk.text for chunk in chunks]) if with_bm25 else None,
    )


def rank(index: ExactIndex, query: str, method: str, reranker: Reranker | None = None) -> list[int]:
    dense_scores = index.vectors @ index.embedder.embed_query(query)
    dense = list(np.argsort(-dense_scores)[:20])
    if method == "dense":
        return dense
    if index.bm25 is None:
        raise RuntimeError("This index was built without BM25")
    lexical = [identifier for identifier, _ in index.bm25.rank(query, 20)]
    if method == "bm25":
        return lexical
    fused = reciprocal_rank_fusion([dense, lexical], 20)
    if reranker is None:
        return fused
    local_order = reranker.rank(query, [index.chunks[position].text for position in fused])
    return [fused[position] for position in local_order]


def retrieval_metrics(question, selected, all_chunks, tokenizer) -> tuple[dict[str, float], int]:
    evidence = [row for row in question.get("evidence", []) if isinstance(row, Mapping)]
    all_grades = [_grade(chunk, evidence) for chunk in all_chunks]
    grades = [_grade(chunk, evidence) for chunk in selected]
    relevant_total = sum(score > 0 for score in all_grades)
    gold = [(int(row["start"]), int(row["end"])) for row in evidence]
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
    budget_intervals: list[tuple[int, int]] = []
    token_total = 0
    for chunk in selected:
        remaining = 2048 - token_total
        if remaining <= 0:
            break
        if chunk.tokens <= remaining:
            if chunk.document_id == str(question["document_id"]):
                budget_intervals.append((chunk.start, chunk.end))
            token_total += chunk.tokens
        else:
            spans = tokenizer.spans(chunk.text)
            truncated_end = spans[min(remaining, len(spans)) - 1][1] if spans and remaining else 0
            if chunk.document_id == str(question["document_id"]):
                budget_intervals.append((chunk.start, chunk.start + truncated_end))
            token_total += min(remaining, len(spans))
            break
    metrics["context_recall_at_2048_tokens"] = context_recall(gold, budget_intervals)
    return metrics, token_total


def _grade(chunk: Chunk, evidence: Sequence[Mapping[str, object]]) -> float:
    matching = [
        (int(row["start"]), int(row["end"]))
        for row in evidence
        if chunk.document_id == str(row["document_id"])
    ]
    return relevance_grades(matching, [(chunk.start, chunk.end)])[0]


def reranker_for(method: str, revisions: Mapping[str, str]) -> Reranker | None:
    model = {
        "rrf-minilm-reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rrf-qwen3-reranker": "Qwen/Qwen3-Reranker-0.6B",
    }.get(method)
    return Reranker(model, revisions[model]) if model else None


RETRIEVAL_DIRECTIONS = {
    "ndcg_at_3": "max",
    "ndcg_at_5": "max",
    "context_recall_at_3": "max",
    "context_recall_at_5": "max",
    "context_precision_at_3": "max",
    "context_precision_at_5": "max",
    "context_recall_at_2048_tokens": "max",
    "operational.p95_latency_seconds": "min",
    "operational.storage_bytes": "min",
}
