"""Shared execution helpers for staged experiments."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TypeVar

import numpy as np

from edumind.common.paths import ARTIFACTS_DIR
from edumind.rag.embedder import Embedder
from edumind.rag.text_chunker import TextChunker
from edumind.rag.types import (
    ChunkRecord,
    ChunkingSettings,
    EmbeddingSettings,
    IngestDocument,
    RetrievalHit,
    build_chunk_id,
    sanitize_filter_metadata,
)
from experiments.mlflow.benchmark import BenchmarkDataset, build_chunk_relevance_map, build_source_relevance_map
from experiments.mlflow.stage_specs import ChunkingCandidate, EmbeddingCandidate
from experiments.mlflow.utils.evaluation import (
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_completeness,
    evaluate_conciseness,
    evaluate_context_precision,
    evaluate_correctness,
    evaluate_faithfulness,
)
from experiments.mlflow.vector_backends import ExperimentVectorBackend, VectorBackendSpec

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
T = TypeVar("T")


def slice_candidates(candidates: Sequence[T], stage_limit: int | None) -> list[T]:
    """Return a bounded candidate list when stage_limit is provided."""
    items = list(candidates)
    if stage_limit is None or stage_limit <= 0:
        return items
    return items[:stage_limit]


def build_embedder(candidate: EmbeddingCandidate) -> Embedder:
    """Construct an embedder from one embedding candidate."""
    return Embedder(
        settings=EmbeddingSettings(
            model_name=candidate.model_name,
            embedding_dim=candidate.embedding_dim,
            device="cuda",
            batch_size=32,
        )
    )


def build_chunk_records(
    dataset: BenchmarkDataset,
    chunk_candidate: ChunkingCandidate,
    *,
    embedder: Embedder,
) -> list[ChunkRecord]:
    """Build deterministic chunk records for a benchmark dataset."""
    documents = dataset.to_ingest_documents()
    if chunk_candidate.kind == "semantic":
        settings = ChunkingSettings(
            chunk_size=chunk_candidate.chunk_size,
            chunk_overlap=chunk_candidate.chunk_overlap,
            separators=("\n\n", "\n", " ", ""),
        )
        chunker = TextChunker(settings=settings, embedder=embedder)
        chunks = chunker.chunk_documents(documents)
        return [_with_original_source_metadata(chunk) for chunk in chunks]

    chunk_records: list[ChunkRecord] = []
    for document in documents:
        if chunk_candidate.kind == "token":
            segments = _chunk_token_windows(
                document.text,
                chunk_candidate.chunk_size,
                chunk_candidate.chunk_overlap,
            )
        elif chunk_candidate.kind == "sentence_window":
            segments = _chunk_sentence_windows(
                document.text,
                chunk_candidate.chunk_size,
                chunk_candidate.chunk_overlap,
            )
        elif chunk_candidate.kind == "hierarchical":
            segments = _chunk_hierarchical(
                document.text,
                chunk_candidate.chunk_size,
                chunk_candidate.child_size or 500,
            )
        else:
            raise ValueError(f"Unsupported chunker kind: {chunk_candidate.kind}")

        total_chunks = len(segments)
        for chunk_index, segment in enumerate(segments):
            metadata = dict(document.metadata)
            metadata.update(segment["metadata"])
            metadata["source_id"] = document.source_id
            metadata["original_source_id"] = document.source_id
            metadata["chunk_index"] = chunk_index
            metadata["total_chunks"] = total_chunks
            chunk_text = str(segment["text"]).strip()
            chunk_records.append(
                ChunkRecord(
                    id=build_chunk_id(document.source_id, chunk_index, chunk_text),
                    source_id=document.source_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    metadata=metadata,
                    filter_metadata=sanitize_filter_metadata(
                        metadata,
                        source=document.source,
                        format_type=document.format_type,
                        file_path=document.file_path,
                    ),
                )
            )
    return chunk_records


def build_backend(
    *,
    backend_spec: VectorBackendSpec,
    stage: str,
    dataset_name: str,
    split: str,
    candidate_suffix: str,
    bm25_alpha: float,
) -> ExperimentVectorBackend:
    """Construct one experiment backend under the shared artifacts tree."""
    persist_directory = (
        ARTIFACTS_DIR
        / "experiments"
        / "mlflow"
        / "vector_store"
        / dataset_name
        / split
        / stage
        / backend_spec.name
        / candidate_suffix
    )
    return ExperimentVectorBackend(
        spec=backend_spec,
        persist_directory=persist_directory,
        collection_name=f"{stage}_{backend_spec.name}_{candidate_suffix}",
        bm25_alpha=bm25_alpha,
    )


def evaluate_retrieval_stack(
    *,
    dataset: BenchmarkDataset,
    chunk_records: Sequence[ChunkRecord],
    embedder: Embedder,
    backend: ExperimentVectorBackend,
    retrieval_mode: str,
    bm25_alpha: float,
    top_k: int,
    include_filters: bool = False,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate one retrieval stack and return aggregate metrics and query details."""
    embedded_chunks = embedder.embed_chunks(list(chunk_records))
    backend.upsert_chunks(embedded_chunks)
    chunk_relevance = build_chunk_relevance_map(dataset.questions, embedded_chunks)
    source_relevance = build_source_relevance_map(dataset.questions)

    chunk_precision_scores: list[float] = []
    chunk_recall_scores: list[float] = []
    chunk_ndcg_scores: list[float] = []
    chunk_mrr_scores: list[float] = []
    source_recall_scores: list[float] = []
    source_mrr_scores: list[float] = []
    source_hit_scores: list[float] = []
    latency_scores: list[float] = []
    filter_success_scores: list[float] = []
    per_query_results: list[dict[str, object]] = []

    for question in dataset.questions:
        filter_metadata = question.filter_metadata if include_filters else None
        start_time = time.perf_counter()
        hits = backend.query(
            mode=retrieval_mode,
            query_text=question.query_text,
            query_embedding=embedder.embed_text(question.query_text).tolist(),
            top_k=top_k,
            filter_metadata=filter_metadata,
            bm25_alpha=bm25_alpha,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        latency_scores.append(latency_ms)

        chunk_ids = [hit.id for hit in hits]
        source_ids = unique_preserve_order([str(hit.metadata.get("source_id", hit.source)) for hit in hits])
        relevant_chunk_ids = sorted(chunk_relevance.get(question.question_id, set()))
        relevant_source_ids = sorted(source_relevance.get(question.question_id, set()))

        chunk_precision_scores.append(
            compute_precision_at_k(chunk_ids, relevant_chunk_ids, top_k)
        )
        chunk_recall_scores.append(compute_recall_at_k(chunk_ids, relevant_chunk_ids, top_k))
        chunk_ndcg_scores.append(compute_ndcg_at_k(chunk_ids, relevant_chunk_ids, top_k))
        chunk_mrr_scores.append(compute_mrr(chunk_ids, relevant_chunk_ids))
        source_recall_scores.append(compute_recall_at_k(source_ids, relevant_source_ids, top_k))
        source_mrr_scores.append(compute_mrr(source_ids, relevant_source_ids))
        source_hit_scores.append(compute_hit_rate_at_k(source_ids, relevant_source_ids, top_k))

        if include_filters and question.filter_metadata:
            filter_success_scores.append(
                1.0 if set(source_ids[:top_k]).intersection(relevant_source_ids) else 0.0
            )

        per_query_results.append(
            {
                "question_id": question.question_id,
                "query": question.query_text,
                "difficulty": question.difficulty,
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_source_ids": source_ids,
                "retrieved_documents": [hit.document for hit in hits],
                "relevant_chunk_ids": relevant_chunk_ids,
                "relevant_source_ids": relevant_source_ids,
                "latency_ms": latency_ms,
                "context": build_context_from_hits(hits),
            }
        )

    backend.reset()
    metrics = {
        "chunk_precision_at_5": float(np.mean(chunk_precision_scores)) if chunk_precision_scores else 0.0,
        "chunk_recall_at_5": float(np.mean(chunk_recall_scores)) if chunk_recall_scores else 0.0,
        "chunk_ndcg_at_5": float(np.mean(chunk_ndcg_scores)) if chunk_ndcg_scores else 0.0,
        "chunk_mrr": float(np.mean(chunk_mrr_scores)) if chunk_mrr_scores else 0.0,
        "source_recall_at_5": float(np.mean(source_recall_scores)) if source_recall_scores else 0.0,
        "source_mrr": float(np.mean(source_mrr_scores)) if source_mrr_scores else 0.0,
        "source_hit_rate_at_5": float(np.mean(source_hit_scores)) if source_hit_scores else 0.0,
        "query_latency_ms": float(np.mean(latency_scores)) if latency_scores else 0.0,
        "filter_success_rate": (
            float(np.mean(filter_success_scores)) if filter_success_scores else 0.0
        ),
        "num_questions": float(len(dataset.questions)),
        "num_chunks": float(len(chunk_records)),
        "storage_size_mb": backend.storage_size_mb(),
    }
    return metrics, per_query_results


def evaluate_llm_answers(
    *,
    dataset: BenchmarkDataset,
    query_payloads: Sequence[Mapping[str, object]],
    answers: Mapping[str, str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate generated answers against benchmark gold references."""
    by_question = {question.question_id: question for question in dataset.questions}
    faithfulness_scores: list[float] = []
    correctness_scores: list[float] = []
    completeness_scores: list[float] = []
    conciseness_scores: list[float] = []
    context_precision_scores: list[float] = []
    answer_lengths: list[float] = []
    per_question_rows: list[dict[str, object]] = []

    for payload in query_payloads:
        question_id = payload.get("question_id")
        if not isinstance(question_id, str):
            continue
        question = by_question.get(question_id)
        if question is None:
            continue

        answer = answers.get(question_id, "")
        context = str(payload.get("context", ""))
        raw_contexts = payload.get("retrieved_documents", [])
        contexts = [str(item) for item in raw_contexts] if isinstance(raw_contexts, list) else []
        faithfulness = evaluate_faithfulness(answer, context)
        correctness = evaluate_correctness(answer, question.gold_answer)
        completeness = evaluate_completeness(answer, question.gold_answer)
        conciseness = evaluate_conciseness(answer, question.gold_answer)
        context_precision = evaluate_context_precision(answer, contexts)

        faithfulness_scores.append(faithfulness)
        correctness_scores.append(correctness)
        completeness_scores.append(completeness)
        conciseness_scores.append(conciseness)
        context_precision_scores.append(float(context_precision["context_precision"]))
        answer_lengths.append(float(len(answer.split())))
        per_question_rows.append(
            {
                "question_id": question_id,
                "query": question.query_text,
                "answer": answer,
                "gold_answer": question.gold_answer,
                "faithfulness": faithfulness,
                "correctness": correctness,
                "completeness": completeness,
                "conciseness": conciseness,
                "context_precision": context_precision["context_precision"],
            }
        )

    metrics = {
        "faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
        "correctness": float(np.mean(correctness_scores)) if correctness_scores else 0.0,
        "completeness": float(np.mean(completeness_scores)) if completeness_scores else 0.0,
        "conciseness": float(np.mean(conciseness_scores)) if conciseness_scores else 0.0,
        "context_precision": (
            float(np.mean(context_precision_scores)) if context_precision_scores else 0.0
        ),
        "avg_answer_length_words": float(np.mean(answer_lengths)) if answer_lengths else 0.0,
        "num_answers": float(len(per_question_rows)),
    }
    return metrics, per_question_rows


def build_context_from_hits(hits: Sequence[RetrievalHit]) -> str:
    """Render one retrieval context block for LLM evaluation."""
    return "\n\n".join(
        f"[Source {index}] {hit.document}\nSource: {hit.metadata.get('source', 'unknown')}"
        for index, hit in enumerate(hits, start=1)
    )


def retrieval_hits_for_payload(
    payload: Mapping[str, object],
    *,
    top_k: int,
) -> list[str]:
    """Return the top-k retrieved source ids from a per-query payload."""
    source_ids = payload.get("retrieved_source_ids", [])
    if not isinstance(source_ids, list):
        return []
    return [str(source_id) for source_id in source_ids[:top_k]]


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    """Return unique values without changing their original order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _with_original_source_metadata(chunk: ChunkRecord) -> ChunkRecord:
    metadata = dict(chunk.metadata)
    metadata.setdefault("original_source_id", chunk.source_id)
    filter_metadata = dict(chunk.filter_metadata)
    return replace(chunk, metadata=metadata, filter_metadata=filter_metadata)


def _chunk_token_windows(text: str, chunk_size: int, overlap: int) -> list[dict[str, object]]:
    words = text.split()
    if not words:
        return []
    segments: list[dict[str, object]] = []
    index = 0
    while index < len(words):
        window = words[index : index + chunk_size]
        segments.append(
            {
                "text": " ".join(window),
                "metadata": {"token_start": index, "token_count": len(window)},
            }
        )
        if index + chunk_size >= len(words):
            break
        index += max(1, chunk_size - overlap)
    return segments


def _chunk_sentence_windows(text: str, window_size: int, overlap: int) -> list[dict[str, object]]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]
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
