"""Serialization helpers for RAG dataclasses and API payloads."""

from __future__ import annotations

from edumind.common.schemas import (
    AnswerResponse,
    IngestResponse,
    QueryResponse,
    RetrievalHitResponse,
)

from .types import AnswerResult, IngestReport, RetrievalHit


def serialize_retrieval_hit(result: RetrievalHit) -> RetrievalHitResponse:
    """Convert one retrieval hit into the shared response schema."""
    return RetrievalHitResponse(
        id=result.id,
        text=result.document,
        metadata=dict(result.metadata),
        score=result.score,
        source=result.source,
        page=result.page,
        rank=result.rank,
        retrieval_method=result.retrieval_method,
        token_count=result.token_count,
    )


def serialize_query_results(query: str, results: list[RetrievalHit]) -> QueryResponse:
    """Convert retrieval results into the shared query response schema."""
    return QueryResponse(
        query=query,
        results=[serialize_retrieval_hit(result) for result in results],
    )


def serialize_answer_result(result: AnswerResult) -> AnswerResponse:
    """Convert one generated answer payload into the shared response schema."""
    return AnswerResponse(
        answer=result.answer,
        sources=[serialize_retrieval_hit(source) for source in result.sources],
        context=result.context,
        retrieval_seconds=result.retrieval_seconds,
        generation_seconds=result.generation_seconds,
        warnings=list(result.warnings),
    )


def serialize_ingest_report(report: IngestReport) -> IngestResponse:
    """Convert one ingest report into the shared ingest response schema."""
    return IngestResponse(
        success=True,
        chunks=report.chunks_created,
        source_id=report.source_id,
        chunks_replaced=report.chunks_replaced,
        elapsed_seconds=report.elapsed_seconds,
    )
