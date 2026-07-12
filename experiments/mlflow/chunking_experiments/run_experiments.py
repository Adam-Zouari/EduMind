"""Chunking-strategy experiments aligned with the current typed RAG APIs."""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, replace

import numpy as np

from edumind.common.config import load_yaml_config
from edumind.common.paths import ARTIFACTS_DIR
from edumind.rag.embedder import Embedder
from edumind.rag.text_chunker import TextChunker
from edumind.rag.types import ChunkingSettings, IngestDocument, RAGConfig, sanitize_filter_metadata
from edumind.rag.vector_store import VectorStore
from experiments.mlflow.mlflow_config import configure_mlflow
from experiments.mlflow.utils import (
    MLflowExperiment,
    build_chunk_record,
    compute_chunk_size_statistics,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    index_documents_by_id,
    load_evaluation_dataset,
    resolve_query_relevant_ids,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkingStrategy:
    """One maintained chunking strategy configuration."""

    name: str
    description: str
    kind: str
    chunk_size: int
    chunk_overlap: int
    child_size: int | None = None


CHUNKING_STRATEGIES = (
    ChunkingStrategy(
        name="fixed_character_baseline",
        description="Fixed 1000 characters with 200 overlap",
        kind="fixed_character",
        chunk_size=1000,
        chunk_overlap=200,
    ),
    ChunkingStrategy(
        name="fixed_character_large",
        description="Fixed 1500 characters with 300 overlap",
        kind="fixed_character",
        chunk_size=1500,
        chunk_overlap=300,
    ),
    ChunkingStrategy(
        name="sentence_window",
        description="Sentence windows with sentence overlap",
        kind="sentence_window",
        chunk_size=10,
        chunk_overlap=2,
    ),
    ChunkingStrategy(
        name="semantic_rag_chunker",
        description="Current RAG semantic chunker with active overlap",
        kind="semantic_rag",
        chunk_size=1000,
        chunk_overlap=200,
    ),
    ChunkingStrategy(
        name="hierarchical",
        description="Parent and child fixed windows",
        kind="hierarchical",
        chunk_size=2000,
        chunk_overlap=0,
        child_size=500,
    ),
)
TOP_K = 5


def build_vector_store(strategy: ChunkingStrategy) -> VectorStore:
    """Build an experiment-local vector store for one chunking strategy."""
    raw_config = load_yaml_config()
    rag_config = RAGConfig.from_mapping(raw_config)
    persist_directory = (
        ARTIFACTS_DIR / "experiments" / "mlflow" / "vector_store" / "chunking" / strategy.name
    )
    settings = replace(
        rag_config.vector_store,
        collection_name=f"chunking_{strategy.name}",
        persist_directory=persist_directory,
    )
    store = VectorStore(settings=settings)
    store.reset_collection()
    return store


def build_strategy_chunks(
    strategy: ChunkingStrategy,
    *,
    embedder: Embedder,
) -> list[object]:
    """Build derived chunk records for every evaluation document."""
    _, documents = load_evaluation_dataset()
    if strategy.kind == "semantic_rag":
        return _build_semantic_rag_chunks(documents, strategy, embedder)

    chunk_records: list[object] = []
    for document in documents:
        if strategy.kind == "fixed_character":
            segments = _chunk_fixed_character(
                document.text,
                strategy.chunk_size,
                strategy.chunk_overlap,
            )
        elif strategy.kind == "sentence_window":
            segments = _chunk_sentence_window(
                document.text,
                strategy.chunk_size,
                strategy.chunk_overlap,
            )
        elif strategy.kind == "hierarchical":
            segments = _chunk_hierarchical(
                document.text,
                strategy.chunk_size,
                strategy.child_size or 500,
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy.kind}")

        total_chunks = len(segments)
        for chunk_index, segment in enumerate(segments):
            chunk_records.append(
                build_chunk_record(
                    document,
                    chunk_text=segment["text"],
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    chunk_id=f"{document.id}:{chunk_index}",
                    extra_metadata={
                        "strategy": strategy.name,
                        **segment["metadata"],
                    },
                )
            )
    return chunk_records


def evaluate_chunking_strategy(
    strategy: ChunkingStrategy,
    *,
    test_mode: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate one maintained chunking strategy."""
    queries, documents = load_evaluation_dataset()
    if test_mode:
        queries = queries[:10]
        documents = documents[:100]

    documents_by_id = index_documents_by_id(documents)
    embedder = Embedder()
    chunk_records = build_strategy_chunks(strategy, embedder=embedder)
    if test_mode:
        chunk_records = chunk_records[: min(len(chunk_records), 800)]

    vector_store = build_vector_store(strategy)
    embedded_chunks = embedder.embed_chunks(chunk_records)  # type: ignore[arg-type]
    vector_store.upsert_chunks(embedded_chunks)

    precision_scores: list[float] = []
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    per_query_results: list[dict[str, object]] = []

    try:
        for query in queries:
            relevant_ids = resolve_query_relevant_ids(query, documents_by_id)
            hits = vector_store.query_by_text(query.query, embedder, top_k=TOP_K)
            retrieved_original_ids = _unique_preserve_order(
                [
                    str(hit.metadata.get("original_chunk_id", hit.id))
                    for hit in hits
                ]
            )
            precision = compute_precision_at_k(retrieved_original_ids, relevant_ids, TOP_K)
            ndcg = compute_ndcg_at_k(retrieved_original_ids, relevant_ids, TOP_K)
            recall = compute_recall_at_k(retrieved_original_ids, relevant_ids, TOP_K)
            mrr = compute_mrr(retrieved_original_ids, relevant_ids)

            precision_scores.append(precision)
            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            mrr_scores.append(mrr)
            per_query_results.append(
                {
                    "query": query.query,
                    "relevant_chunk_ids": relevant_ids,
                    "retrieved_original_chunk_ids": retrieved_original_ids,
                    "precision_at_5": precision,
                    "ndcg_at_5": ndcg,
                    "recall_at_5": recall,
                    "mrr": mrr,
                }
            )
    finally:
        vector_store.reset_collection()

    chunk_texts = [chunk.text for chunk in chunk_records]  # type: ignore[attr-defined]
    chunk_stats = compute_chunk_size_statistics(chunk_texts)
    metrics = {
        "precision_at_5": float(np.mean(precision_scores)),
        "precision_at_5_std": float(np.std(precision_scores)),
        "ndcg_at_5": float(np.mean(ndcg_scores)),
        "ndcg_at_5_std": float(np.std(ndcg_scores)),
        "recall_at_5": float(np.mean(recall_scores)),
        "recall_at_5_std": float(np.std(recall_scores)),
        "mrr": float(np.mean(mrr_scores)),
        "mrr_std": float(np.std(mrr_scores)),
        "total_chunks": chunk_stats["num_chunks"],
        "avg_chunk_size_chars": chunk_stats["mean_chars"],
        "median_chunk_size_chars": chunk_stats["median_chars"],
        "std_chunk_size_chars": chunk_stats["std_chars"],
        "avg_chunk_size_tokens": chunk_stats["mean_tokens"],
        "median_chunk_size_tokens": chunk_stats["median_tokens"],
        "num_queries_evaluated": float(len(queries)),
    }
    artifacts = {
        "query_results.json": per_query_results,
        "sample_chunks.json": [
            {
                "id": chunk.id,  # type: ignore[attr-defined]
                "text": (chunk.text[:200] + "...") if len(chunk.text) > 200 else chunk.text,  # type: ignore[attr-defined]
                "metadata": chunk.metadata,  # type: ignore[attr-defined]
            }
            for chunk in chunk_records[:5]
        ],
        "chunk_statistics.json": chunk_stats,
    }
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False) -> int:
    """Run all maintained chunking experiments."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    strategies = CHUNKING_STRATEGIES[:2] if test_mode else CHUNKING_STRATEGIES
    for strategy in strategies:
        with MLflowExperiment("chunking_experiments", f"chunking_{strategy.name}") as run:
            run.log_params(
                {
                    "strategy_name": strategy.name,
                    "description": strategy.description,
                    "kind": strategy.kind,
                    "chunk_size": strategy.chunk_size,
                    "chunk_overlap": strategy.chunk_overlap,
                    "child_size": strategy.child_size or 0,
                    "test_mode": test_mode,
                }
            )
            metrics, artifacts = evaluate_chunking_strategy(strategy, test_mode=test_mode)
            run.log_metrics(metrics)
            for filename, content in artifacts.items():
                run.log_artifact(filename, content)
            logger.info(
                "Completed %s: recall@5=%.4f mrr=%.4f total_chunks=%.0f",
                strategy.name,
                metrics["recall_at_5"],
                metrics["mrr"],
                metrics["total_chunks"],
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the chunking-experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run maintained chunking experiments.")
    parser.add_argument("--test-mode", action="store_true", help="Run a reduced strategy subset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for maintained chunking experiments."""
    args = build_parser().parse_args(argv)
    return run_all_experiments(test_mode=args.test_mode)


def _build_semantic_rag_chunks(
    documents: list[object],
    strategy: ChunkingStrategy,
    embedder: Embedder,
) -> list[object]:
    """Build chunks with the current RAG chunker implementation."""
    settings = ChunkingSettings(
        chunk_size=strategy.chunk_size,
        chunk_overlap=strategy.chunk_overlap,
        separators=("\n\n", "\n", " ", ""),
    )
    chunker = TextChunker(settings=settings, embedder=embedder)
    chunk_records: list[object] = []
    for document in documents:
        metadata = dict(document.metadata)  # type: ignore[attr-defined]
        metadata["original_chunk_id"] = document.id  # type: ignore[attr-defined]
        metadata["strategy"] = strategy.name
        ingest_document = IngestDocument(
            text=document.text,  # type: ignore[attr-defined]
            source_id=document.source_id,  # type: ignore[attr-defined]
            source=document.source,  # type: ignore[attr-defined]
            format_type="evaluation",
            metadata=metadata,
            filter_metadata=sanitize_filter_metadata(
                metadata,
                source=document.source,  # type: ignore[attr-defined]
                format_type="evaluation",
            ),
        )
        chunks = chunker.chunk_document(ingest_document)
        total_chunks = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_records.append(
                replace(
                    chunk,
                    id=f"{document.id}:{chunk_index}",  # type: ignore[attr-defined]
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
            )
    return chunk_records


def _chunk_fixed_character(text: str, chunk_size: int, overlap: int) -> list[dict[str, object]]:
    """Split text into fixed character windows."""
    segments: list[dict[str, object]] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        segment_text = text[start:end].strip()
        if segment_text:
            segments.append(
                {
                    "text": segment_text,
                    "metadata": {"start_char": start, "end_char": min(end, len(text))},
                }
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return segments


def _chunk_sentence_window(text: str, window_size: int, overlap: int) -> list[dict[str, object]]:
    """Split text into overlapping sentence windows."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    if not sentences:
        return [{"text": text.strip(), "metadata": {"num_sentences": 1}}] if text.strip() else []

    segments: list[dict[str, object]] = []
    index = 0
    while index < len(sentences):
        chunk_sentences = sentences[index : index + window_size]
        segments.append(
            {
                "text": " ".join(chunk_sentences),
                "metadata": {"num_sentences": len(chunk_sentences)},
            }
        )
        if index + window_size >= len(sentences):
            break
        index += max(1, window_size - overlap)
    return segments


def _chunk_hierarchical(text: str, parent_size: int, child_size: int) -> list[dict[str, object]]:
    """Split text into parent windows and then child windows within each parent."""
    segments: list[dict[str, object]] = []
    for parent_index, parent_start in enumerate(range(0, len(text), parent_size)):
        parent_text = text[parent_start : parent_start + parent_size]
        for child_start in range(0, len(parent_text), child_size):
            chunk_text = parent_text[child_start : child_start + child_size].strip()
            if not chunk_text:
                continue
            segments.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "parent_index": parent_index,
                        "child_start": child_start,
                        "is_child": True,
                    },
                }
            )
    return segments


def _unique_preserve_order(items: list[str]) -> list[str]:
    """Return unique values without changing their original order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
