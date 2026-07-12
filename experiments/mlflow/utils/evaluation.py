"""Shared evaluation utilities for maintained MLflow experiments."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

import numpy as np

logger = logging.getLogger(__name__)
T = TypeVar("T")


def compute_recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Recall@K."""
    if not relevant_ids:
        logger.warning("No relevant IDs provided for Recall@K calculation")
        return 0.0
    return len(set(retrieved_ids[:k]).intersection(relevant_ids)) / len(set(relevant_ids))


def compute_mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate mean reciprocal rank for one ranked result list."""
    if not relevant_ids:
        logger.warning("No relevant IDs provided for MRR calculation")
        return 0.0

    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]).intersection(relevant_ids)) / k


def compute_ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate normalized discounted cumulative gain at K."""
    if not relevant_ids or k <= 0:
        return 0.0

    relevant_set = set(relevant_ids)
    relevance_scores = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved_ids[:k]]
    dcg = sum(score / np.log2(index + 2) for index, score in enumerate(relevance_scores))

    ideal_length = min(len(relevant_set), k)
    ideal_scores = [1.0] * ideal_length + [0.0] * max(0, k - ideal_length)
    idcg = sum(score / np.log2(index + 2) for index, score in enumerate(ideal_scores))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_map(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate average precision for one ranked result list."""
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    precisions: list[float] = []
    hits = 0
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            precisions.append(hits / index)
    if not precisions:
        return 0.0
    return float(sum(precisions) / len(relevant_set))


def compute_hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Hit Rate@K."""
    if not relevant_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]).intersection(relevant_ids) else 0.0


def compute_diversity(embeddings: np.ndarray) -> float:
    """Calculate embedding diversity as one minus mean pairwise similarity."""
    if len(embeddings) <= 1:
        return 1.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    upper_triangle_indices = np.triu_indices(len(embeddings), k=1)
    pairwise_similarities = similarity_matrix[upper_triangle_indices]
    avg_similarity = float(np.mean(pairwise_similarities)) if pairwise_similarities.size else 0.0
    return max(0.0, min(1.0, 1.0 - avg_similarity))


def compute_chunk_coherence(chunk_embeddings: np.ndarray, boundary_embeddings: np.ndarray) -> float:
    """Calculate a simple coherence ratio from intra- and cross-boundary similarities."""
    if len(chunk_embeddings) == 0:
        return 0.0

    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = chunk_embeddings / norms
    similarity_matrix = np.dot(normalized, normalized.T)
    upper_triangle = np.triu_indices(len(chunk_embeddings), k=1)
    intra_similarities = similarity_matrix[upper_triangle]
    avg_intra = float(np.mean(intra_similarities)) if intra_similarities.size else 0.0

    boundary_scores: list[float] = []
    for pair in boundary_embeddings:
        left_norm = float(np.linalg.norm(pair[0]))
        right_norm = float(np.linalg.norm(pair[1]))
        if left_norm > 0 and right_norm > 0:
            boundary_scores.append(float(np.dot(pair[0], pair[1]) / (left_norm * right_norm)))
    avg_boundary = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    if avg_boundary == 0.0:
        return avg_intra
    return avg_intra / avg_boundary


@contextmanager
def measure_latency() -> dict[str, float]:
    """Measure execution time and expose it through a mutable timer dict."""
    timer = {"latency_ms": 0.0}
    start_time = time.perf_counter()
    try:
        yield timer
    finally:
        timer["latency_ms"] = (time.perf_counter() - start_time) * 1000


def measure_function_latency(
    func: Callable[..., T],
    *args: object,
    **kwargs: object,
) -> tuple[T, float]:
    """Measure one function call and return the result with latency in milliseconds."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start_time) * 1000


def evaluate_answer_quality(
    answer: str,
    reference_answer: str | None = None,
    context: str | None = None,
) -> dict[str, float]:
    """Evaluate answer quality with lightweight heuristics."""
    del reference_answer

    word_count = len(answer.split())
    metrics: dict[str, float] = {
        "response_length_words": float(word_count),
        "is_non_empty": 1.0 if answer.strip() else 0.0,
    }
    if context:
        metrics["is_original"] = 0.0 if answer.strip() in context else 1.0

    quality_score = 0.0
    if 5 <= word_count <= 200:
        quality_score += 0.5
    if metrics["is_non_empty"] > 0:
        quality_score += 0.25
    if metrics.get("is_original", 1.0) > 0:
        quality_score += 0.25
    metrics["basic_quality_score"] = quality_score
    return metrics


def evaluate_faithfulness(answer: str, context: str) -> float:
    """Evaluate whether answer terms are grounded in the provided context."""
    if not answer or not context:
        return 0.0

    answer_terms = {
        word.lower().strip(".,!?;:")
        for word in answer.split()
        if len(word) > 4
    }
    if not answer_terms:
        return 0.5

    context_lower = context.lower()
    found_terms = sum(1 for term in answer_terms if term in context_lower)
    return found_terms / len(answer_terms)


def evaluate_correctness(answer: str, reference_answer: str) -> float:
    """Evaluate correctness with a simple token-overlap F1 approximation."""
    if not answer or not reference_answer:
        return 0.0

    answer_tokens = {word.lower().strip(".,!?;:") for word in answer.split()}
    reference_tokens = {word.lower().strip(".,!?;:") for word in reference_answer.split()}
    answer_tokens.discard("")
    reference_tokens.discard("")
    if not answer_tokens or not reference_tokens:
        return 0.0

    overlap = len(answer_tokens.intersection(reference_tokens))
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_completeness(answer: str, reference_answer: str) -> float:
    """Evaluate whether an answer covers key terms from the reference text."""
    if not reference_answer:
        return 1.0
    if not answer:
        return 0.0

    reference_terms = {
        word.lower().strip(".,!?;:")
        for word in reference_answer.split()
        if len(word) > 4
    }
    if not reference_terms:
        return 1.0
    answer_lower = answer.lower()
    covered_terms = sum(1 for term in reference_terms if term in answer_lower)
    return covered_terms / len(reference_terms)


def evaluate_conciseness(answer: str, reference_answer: str | None = None) -> float:
    """Evaluate whether the answer length stays within a reasonable range."""
    if not answer:
        return 0.0

    answer_words = len(answer.split())
    if reference_answer:
        reference_words = len(reference_answer.split())
        if reference_words == 0:
            return 1.0
        ratio = answer_words / reference_words
        if 0.8 <= ratio <= 1.2:
            return 1.0
        if ratio < 0.8:
            return ratio / 0.8
        return max(0.0, 1.0 - ((ratio - 1.2) / 2))

    if 20 <= answer_words <= 150:
        return 1.0
    if answer_words < 20:
        return answer_words / 20
    return max(0.0, 1.0 - (((answer_words - 150) / 150) / 2))


def evaluate_context_precision(
    answer: str,
    contexts: list[str],
    context_ids: list[str] | None = None,
) -> dict[str, object]:
    """Estimate how many provided context chunks were actually used."""
    del context_ids

    if not contexts or not answer:
        return {
            "context_precision": 0.0,
            "contexts_used": 0,
            "contexts_provided": len(contexts) if contexts else 0,
            "used_context_indices": [],
        }

    answer_lower = answer.lower()
    contexts_used = 0
    used_indices: list[int] = []
    for index, context in enumerate(contexts):
        context_terms = {
            word.lower().strip(".,!?;:")
            for word in context.split()
            if len(word) > 5
        }
        terms_found = sum(1 for term in context_terms if term in answer_lower)
        if context_terms and (terms_found / len(context_terms)) >= 0.2:
            contexts_used += 1
            used_indices.append(index)

    return {
        "context_precision": contexts_used / len(contexts),
        "contexts_used": contexts_used,
        "contexts_provided": len(contexts),
        "used_context_indices": used_indices,
    }


def compute_chunk_size_statistics(chunks: list[str]) -> dict[str, float]:
    """Compute descriptive chunk-length statistics."""
    if not chunks:
        return {
            "num_chunks": 0.0,
            "mean_chars": 0.0,
            "median_chars": 0.0,
            "std_chars": 0.0,
            "min_chars": 0.0,
            "max_chars": 0.0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "std_tokens": 0.0,
            "min_tokens": 0.0,
            "max_tokens": 0.0,
        }

    char_counts = [len(chunk) for chunk in chunks]
    token_counts = [len(chunk.split()) for chunk in chunks]
    return {
        "num_chunks": float(len(chunks)),
        "mean_chars": float(np.mean(char_counts)),
        "median_chars": float(np.median(char_counts)),
        "std_chars": float(np.std(char_counts)),
        "min_chars": float(np.min(char_counts)),
        "max_chars": float(np.max(char_counts)),
        "mean_tokens": float(np.mean(token_counts)),
        "median_tokens": float(np.median(token_counts)),
        "std_tokens": float(np.std(token_counts)),
        "min_tokens": float(np.min(token_counts)),
        "max_tokens": float(np.max(token_counts)),
    }


def compute_mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean and standard deviation across repeated metric payloads."""
    if not metrics_list:
        return {}

    all_keys = set().union(*(metrics.keys() for metrics in metrics_list))
    mean_metrics: dict[str, float] = {}
    for key in all_keys:
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if values:
            mean_metrics[f"mean_{key}"] = float(np.mean(values))
            mean_metrics[f"std_{key}"] = float(np.std(values))
    return mean_metrics


def evaluate_retrieval_quality(
    query: str,
    retrieved_docs: list[dict[str, object]],
    ground_truth_ids: list[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute a small retrieval-quality summary for one query."""
    del query
    resolved_k_values = k_values or [1, 3, 5, 10]
    retrieved_ids = [str(doc.get("id", doc.get("document_id", ""))) for doc in retrieved_docs]

    metrics: dict[str, float] = {
        f"recall_at_{k}": compute_recall_at_k(retrieved_ids, ground_truth_ids, k)
        for k in resolved_k_values
    }
    metrics["mrr"] = compute_mrr(retrieved_ids, ground_truth_ids)
    metrics["num_retrieved"] = float(len(retrieved_docs))
    return metrics
