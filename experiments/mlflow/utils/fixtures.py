"""Evaluation-fixture loading and adapters for maintained experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from edumind.rag.types import ChunkRecord, build_source_id, sanitize_filter_metadata
from experiments.mlflow.mlflow_config import EVALUATION_DIR


@dataclass(frozen=True)
class EvaluationDocument:
    """One reference document loaded from the evaluation ground truth."""

    id: str
    text: str
    source: str
    source_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationQuery:
    """One evaluation query plus its known relevant reference ids."""

    query: str
    relevant_chunk_ids: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)


def load_evaluation_documents() -> list[EvaluationDocument]:
    """Load normalized evaluation documents from ground-truth fixtures."""
    with (EVALUATION_DIR / "ground_truth.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        return []

    documents: list[EvaluationDocument] = []
    for document_id, raw_document in payload.items():
        if not isinstance(raw_document, Mapping):
            continue
        text = str(raw_document.get("text", "")).strip()
        if not text:
            continue

        metadata = {key: value for key, value in raw_document.items() if key != "text"}
        source = str(metadata.get("source", "evaluation"))
        source_id = build_source_id(
            text=text,
            source=source,
            file_path=None,
            format_type="evaluation",
            metadata=metadata,
        )
        documents.append(
            EvaluationDocument(
                id=str(document_id),
                text=text,
                source=source,
                source_id=source_id,
                metadata=metadata,
            )
        )
    return documents


def load_evaluation_queries() -> list[EvaluationQuery]:
    """Load evaluation queries from the maintained fixture file."""
    with (EVALUATION_DIR / "eval_queries.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        return []

    queries: list[EvaluationQuery] = []
    for raw_query in payload:
        if not isinstance(raw_query, Mapping):
            continue
        query_text = str(raw_query.get("query", "")).strip()
        if not query_text:
            continue

        relevant = raw_query.get("relevant_chunks", [])
        if not isinstance(relevant, list):
            relevant = []

        metadata = {
            key: value
            for key, value in raw_query.items()
            if key not in {"query", "relevant_chunks"}
        }
        queries.append(
            EvaluationQuery(
                query=query_text,
                relevant_chunk_ids=tuple(str(value) for value in relevant),
                metadata=metadata,
            )
        )
    return queries


def load_evaluation_dataset() -> tuple[list[EvaluationQuery], list[EvaluationDocument]]:
    """Load both maintained evaluation fixture sets."""
    return load_evaluation_queries(), load_evaluation_documents()


def build_reference_chunk_records(documents: Sequence[EvaluationDocument]) -> list[ChunkRecord]:
    """Convert reference documents into single-chunk ChunkRecord values."""
    chunk_records: list[ChunkRecord] = []
    for document in documents:
        metadata = dict(document.metadata)
        metadata.setdefault("source", document.source)
        metadata.setdefault("source_id", document.source_id)
        chunk_records.append(
            ChunkRecord(
                id=document.id,
                source_id=document.source_id,
                text=document.text,
                chunk_index=0,
                total_chunks=1,
                metadata=metadata,
                filter_metadata=sanitize_filter_metadata(
                    metadata,
                    source=document.source,
                    format_type="evaluation",
                ),
            )
        )
    return chunk_records


def build_chunk_record(
    document: EvaluationDocument,
    *,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    extra_metadata: Mapping[str, object] | None = None,
    chunk_id: str | None = None,
) -> ChunkRecord:
    """Build one derived ChunkRecord while preserving the original reference id."""
    metadata = dict(document.metadata)
    metadata.setdefault("source", document.source)
    metadata["original_chunk_id"] = document.id
    metadata["source_id"] = document.source_id
    if extra_metadata:
        metadata.update(extra_metadata)

    return ChunkRecord(
        id=chunk_id or f"{document.id}:{chunk_index}",
        source_id=document.source_id,
        text=chunk_text,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        metadata=metadata,
        filter_metadata=sanitize_filter_metadata(
            metadata,
            source=document.source,
            format_type="evaluation",
        ),
    )


def index_documents_by_id(
    documents: Sequence[EvaluationDocument],
) -> dict[str, EvaluationDocument]:
    """Build a lookup table for reference documents."""
    return {document.id: document for document in documents}


def resolve_query_relevant_ids(
    query: EvaluationQuery,
    documents_by_id: Mapping[str, EvaluationDocument],
) -> list[str]:
    """Expand sampled relevance labels into full matching groups when possible."""
    if not query.relevant_chunk_ids:
        return []

    domain = query.metadata.get("domain")
    expected_variant = query.metadata.get("expected_variant")
    sample_document = documents_by_id.get(query.relevant_chunk_ids[0])
    topic = sample_document.metadata.get("topic") if sample_document is not None else None

    if domain is None or expected_variant is None or topic is None:
        return list(query.relevant_chunk_ids)

    expanded = [
        document.id
        for document in documents_by_id.values()
        if document.metadata.get("domain") == domain
        and document.metadata.get("topic") == topic
        and document.metadata.get("variant") == expected_variant
    ]
    return expanded or list(query.relevant_chunk_ids)
