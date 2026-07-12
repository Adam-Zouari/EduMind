"""Embedding-model experiments aligned with the current evaluation fixtures."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

from edumind.rag.embedder import Embedder
from edumind.rag.types import EmbeddingSettings
from experiments.mlflow.mlflow_config import configure_mlflow
from experiments.mlflow.utils import (
    MLflowExperiment,
    compute_map,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    get_gpu_memory_usage,
    is_cuda_available,
    load_evaluation_dataset,
    measure_latency,
    resolve_query_relevant_ids,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingExperiment:
    """One maintained embedding-model experiment configuration."""

    model_name: str
    embedding_dim: int
    description: str


EMBEDDING_MODELS = (
    EmbeddingExperiment(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        description="Lightweight baseline",
    ),
    EmbeddingExperiment(
        model_name="sentence-transformers/all-mpnet-base-v2",
        embedding_dim=768,
        description="Balanced English encoder",
    ),
    EmbeddingExperiment(
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        embedding_dim=768,
        description="Question-answer retrieval encoder",
    ),
    EmbeddingExperiment(
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
        description="Compact BGE encoder",
    ),
    EmbeddingExperiment(
        model_name="BAAI/bge-base-en-v1.5",
        embedding_dim=768,
        description="Larger BGE encoder",
    ),
)


def evaluate_embedding_model(
    experiment: EmbeddingExperiment,
    *,
    test_mode: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate one maintained embedding model on the shared fixtures."""
    queries, documents = load_evaluation_dataset()
    if test_mode:
        queries = queries[:10]
        documents = documents[:100]

    documents_by_id = {document.id: document for document in documents}
    chunk_ids = [document.id for document in documents]
    chunk_texts = [document.text for document in documents]
    query_texts = [query.query for query in queries]

    device = "cuda" if is_cuda_available() else "cpu"
    embedder = Embedder(
        settings=EmbeddingSettings(
            model_name=experiment.model_name,
            embedding_dim=experiment.embedding_dim,
            device=device,
            batch_size=32,
        )
    )

    load_start = time.perf_counter()
    chunk_embeddings = embedder.embed_texts(chunk_texts, show_progress=False)
    model_load_and_encode_time = time.perf_counter() - load_start
    chunk_throughput = len(chunk_texts) / model_load_and_encode_time if chunk_texts else 0.0

    query_embeddings: list[np.ndarray] = []
    query_latencies: list[float] = []
    for query_text in query_texts:
        with measure_latency() as timer:
            query_embedding = embedder.embed_text(query_text)
        query_latencies.append(float(timer["latency_ms"]))
        query_embeddings.append(query_embedding)

    normalized_chunk_embeddings = _normalize_embeddings(chunk_embeddings)
    precision_scores: list[float] = []
    ndcg_at_5_scores: list[float] = []
    ndcg_at_10_scores: list[float] = []
    map_scores: list[float] = []
    mrr_scores: list[float] = []

    for query, query_embedding in zip(queries, query_embeddings, strict=False):
        relevant_ids = resolve_query_relevant_ids(query, documents_by_id)
        normalized_query = _normalize_embeddings(query_embedding.reshape(1, -1))[0]
        similarities = np.dot(normalized_chunk_embeddings, normalized_query)
        top_indices = np.argsort(similarities)[::-1][:10]
        retrieved_ids = [chunk_ids[index] for index in top_indices]

        precision_scores.append(compute_precision_at_k(retrieved_ids, relevant_ids, 5))
        ndcg_at_5_scores.append(compute_ndcg_at_k(retrieved_ids, relevant_ids, 5))
        ndcg_at_10_scores.append(compute_ndcg_at_k(retrieved_ids, relevant_ids, 10))
        map_scores.append(compute_map(retrieved_ids, relevant_ids))
        mrr_scores.append(compute_mrr(retrieved_ids, relevant_ids))

    gpu_metrics = get_gpu_memory_usage()
    metrics = {
        "throughput_sent_per_sec": float(chunk_throughput),
        "avg_query_latency_ms": float(np.mean(query_latencies)),
        "gpu_memory_mb": float(gpu_metrics.get("allocated_mb", 0.0)),
        "model_load_time_sec": float(model_load_and_encode_time),
        "embedding_dim": float(experiment.embedding_dim),
        "precision_at_5": float(np.mean(precision_scores)),
        "precision_at_5_std": float(np.std(precision_scores)),
        "ndcg_at_5": float(np.mean(ndcg_at_5_scores)),
        "ndcg_at_5_std": float(np.std(ndcg_at_5_scores)),
        "ndcg_at_10": float(np.mean(ndcg_at_10_scores)),
        "ndcg_at_10_std": float(np.std(ndcg_at_10_scores)),
        "map": float(np.mean(map_scores)),
        "map_std": float(np.std(map_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "mrr_std": float(np.std(mrr_scores)),
        "num_queries": float(len(queries)),
        "num_chunks": float(len(chunk_texts)),
    }
    artifacts = {
        "evaluation_results.json": {
            "per_query_precision_at_5": precision_scores,
            "per_query_ndcg_at_5": ndcg_at_5_scores,
            "per_query_ndcg_at_10": ndcg_at_10_scores,
            "per_query_map": map_scores,
            "per_query_mrr": mrr_scores,
            "per_query_latency_ms": query_latencies,
        },
        "sample_embeddings.npy": np.asarray(query_embeddings[:5], dtype=float),
    }
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False) -> int:
    """Run all maintained embedding experiments."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    experiments = EMBEDDING_MODELS[:2] if test_mode else EMBEDDING_MODELS
    for experiment in experiments:
        run_name = experiment.model_name.split("/")[-1].replace(":", "_").replace(".", "_")
        with MLflowExperiment("embedding_experiments", f"embedding_{run_name}") as run:
            run.log_params(
                {
                    "model_name": experiment.model_name,
                    "embedding_dim": experiment.embedding_dim,
                    "description": experiment.description,
                    "device": "cuda" if is_cuda_available() else "cpu",
                    "batch_size": 32,
                    "test_mode": test_mode,
                }
            )
            metrics, artifacts = evaluate_embedding_model(experiment, test_mode=test_mode)
            run.log_metrics(metrics)
            for filename, content in artifacts.items():
                run.log_artifact(filename, content)
            logger.info(
                "Completed %s: precision@5=%.4f mrr=%.4f latency=%.2f ms",
                experiment.model_name,
                metrics["precision_at_5"],
                metrics["mrr"],
                metrics["avg_query_latency_ms"],
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the embedding-experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run maintained embedding experiments.")
    parser.add_argument("--test-mode", action="store_true", help="Run a reduced model subset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for maintained embedding experiments."""
    args = build_parser().parse_args(argv)
    return run_all_experiments(test_mode=args.test_mode)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embedding vectors for cosine scoring."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


if __name__ == "__main__":
    raise SystemExit(main())
