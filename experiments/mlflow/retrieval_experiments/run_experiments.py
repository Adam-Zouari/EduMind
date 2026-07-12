"""Retrieval-strategy experiments aligned with the current typed RAG APIs."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace

import numpy as np

from edumind.common.config import load_yaml_config
from edumind.common.paths import ARTIFACTS_DIR
from edumind.rag.embedder import Embedder
from edumind.rag.types import RAGConfig
from edumind.rag.vector_store import VectorStore
from experiments.mlflow.mlflow_config import configure_mlflow
from experiments.mlflow.utils import (
    MLflowExperiment,
    build_reference_chunk_records,
    compute_diversity,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    index_documents_by_id,
    load_evaluation_dataset,
    measure_latency,
    resolve_query_relevant_ids,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalStrategy:
    """One maintained retrieval strategy configuration."""

    name: str
    description: str
    bm25_alpha: float


RETRIEVAL_STRATEGIES = (
    RetrievalStrategy("pure_vector", "Dense retrieval only", 0.0),
    RetrievalStrategy("hybrid_light_bm25", "Dense + 30% BM25", 0.3),
    RetrievalStrategy("hybrid_balanced", "Dense + 50% BM25", 0.5),
    RetrievalStrategy("hybrid_heavy_bm25", "Dense + 70% BM25", 0.7),
)
TOP_K = 5


def build_vector_store(strategy: RetrievalStrategy) -> VectorStore:
    """Build an experiment-local vector store for one retrieval strategy."""
    raw_config = load_yaml_config()
    rag_config = RAGConfig.from_mapping(raw_config)
    persist_directory = (
        ARTIFACTS_DIR / "experiments" / "mlflow" / "vector_store" / "retrieval" / strategy.name
    )
    settings = replace(
        rag_config.vector_store,
        collection_name=f"retrieval_{strategy.name}",
        persist_directory=persist_directory,
        bm25_alpha=strategy.bm25_alpha,
    )
    store = VectorStore(settings=settings)
    store.reset_collection()
    return store


def evaluate_retrieval_strategy(
    strategy: RetrievalStrategy,
    *,
    test_mode: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate one maintained retrieval strategy."""
    queries, documents = load_evaluation_dataset()
    if test_mode:
        queries = queries[:10]

    documents_by_id = index_documents_by_id(documents)
    base_chunks = build_reference_chunk_records(documents)
    embedder = Embedder()
    embedded_chunks = embedder.embed_chunks(base_chunks)
    embedding_lookup = {
        chunk.id: np.asarray(chunk.embedding or [], dtype=float)
        for chunk in embedded_chunks
    }

    vector_store = build_vector_store(strategy)
    vector_store.upsert_chunks(embedded_chunks)

    precision_scores: list[float] = []
    ndcg_scores: list[float] = []
    hit_rate_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    diversity_scores: list[float] = []
    latency_scores: list[float] = []
    query_results: list[dict[str, object]] = []

    try:
        for query in queries:
            relevant_ids = resolve_query_relevant_ids(query, documents_by_id)
            query_embedding = embedder.embed_text(query.query).tolist()
            with measure_latency() as timer:
                hits = vector_store.query_hybrid(
                    query.query,
                    query_embedding,
                    top_k=TOP_K,
                )

            retrieved_ids = [hit.id for hit in hits]
            retrieved_embeddings = [
                embedding_lookup[hit.id]
                for hit in hits
                if hit.id in embedding_lookup and embedding_lookup[hit.id].size > 0
            ]
            diversity = (
                compute_diversity(np.vstack(retrieved_embeddings))
                if len(retrieved_embeddings) >= 2
                else 1.0
            )

            precision = compute_precision_at_k(retrieved_ids, relevant_ids, TOP_K)
            ndcg = compute_ndcg_at_k(retrieved_ids, relevant_ids, TOP_K)
            hit_rate = compute_hit_rate_at_k(retrieved_ids, relevant_ids, TOP_K)
            recall = compute_recall_at_k(retrieved_ids, relevant_ids, TOP_K)
            mrr = compute_mrr(retrieved_ids, relevant_ids)

            precision_scores.append(precision)
            ndcg_scores.append(ndcg)
            hit_rate_scores.append(hit_rate)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            diversity_scores.append(diversity)
            latency_scores.append(float(timer["latency_ms"]))
            query_results.append(
                {
                    "query": query.query,
                    "relevant_chunk_ids": relevant_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "precision_at_5": precision,
                    "ndcg_at_5": ndcg,
                    "hit_rate_at_5": hit_rate,
                    "recall_at_5": recall,
                    "mrr": mrr,
                    "diversity": diversity,
                    "latency_ms": float(timer["latency_ms"]),
                }
            )
    finally:
        vector_store.reset_collection()

    metrics = {
        "precision_at_5": float(np.mean(precision_scores)),
        "precision_at_5_std": float(np.std(precision_scores)),
        "ndcg_at_5": float(np.mean(ndcg_scores)),
        "ndcg_at_5_std": float(np.std(ndcg_scores)),
        "hit_rate_at_5": float(np.mean(hit_rate_scores)),
        "hit_rate_at_5_std": float(np.std(hit_rate_scores)),
        "recall_at_5": float(np.mean(recall_scores)),
        "recall_at_5_std": float(np.std(recall_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "mrr_std": float(np.std(mrr_scores)),
        "diversity": float(np.mean(diversity_scores)),
        "diversity_std": float(np.std(diversity_scores)),
        "latency_ms": float(np.mean(latency_scores)),
        "latency_std_ms": float(np.std(latency_scores)),
        "num_queries": float(len(queries)),
    }
    artifacts = {
        "query_results.json": query_results,
        "failure_cases.json": [result for result in query_results if result["recall_at_5"] < 0.5],
    }
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False) -> int:
    """Run all maintained retrieval experiments."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    strategies = RETRIEVAL_STRATEGIES[:2] if test_mode else RETRIEVAL_STRATEGIES
    summaries: list[tuple[str, dict[str, float]]] = []

    for strategy in strategies:
        with MLflowExperiment("retrieval_experiments", f"retrieval_{strategy.name}") as experiment:
            experiment.log_params(
                {
                    "strategy_name": strategy.name,
                    "description": strategy.description,
                    "bm25_alpha": strategy.bm25_alpha,
                    "top_k": TOP_K,
                    "test_mode": test_mode,
                }
            )
            metrics, artifacts = evaluate_retrieval_strategy(strategy, test_mode=test_mode)
            experiment.log_metrics(metrics)
            for filename, content in artifacts.items():
                experiment.log_artifact(filename, content)
            summaries.append((strategy.name, metrics))
            logger.info(
                "Completed %s: recall@5=%.4f, mrr=%.4f, latency=%.2f ms",
                strategy.name,
                metrics["recall_at_5"],
                metrics["mrr"],
                metrics["latency_ms"],
            )

    for strategy_name, metrics in summaries:
        logger.info(
            "Summary %-24s recall@5=%.4f mrr=%.4f latency=%.2f ms",
            strategy_name,
            metrics["recall_at_5"],
            metrics["mrr"],
            metrics["latency_ms"],
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the retrieval-experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run maintained retrieval experiments.")
    parser.add_argument("--test-mode", action="store_true", help="Run a smaller query subset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for maintained retrieval experiments."""
    args = build_parser().parse_args(argv)
    return run_all_experiments(test_mode=args.test_mode)


if __name__ == "__main__":
    raise SystemExit(main())
