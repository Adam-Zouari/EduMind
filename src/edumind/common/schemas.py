"""Validated HTTP request and response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedDocumentPayload(StrictModel):
    text: str = Field(min_length=1)
    source: str | None = None
    format_type: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(StrictModel):
    document: ExtractedDocumentPayload


class IngestResponse(StrictModel):
    success: bool
    chunks: int = Field(ge=0)
    source_id: str
    chunks_replaced: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)


class QueryRequest(StrictModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=10)
    generate_answer: bool = True
    filter_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievalHitResponse(StrictModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    source: str
    page: str
    rank: int = 0
    retrieval_method: str = "dense"
    token_count: int = 0


class QueryResponse(StrictModel):
    query: str
    results: list[RetrievalHitResponse] = Field(default_factory=list)


class AnswerResponse(StrictModel):
    answer: str
    sources: list[RetrievalHitResponse] = Field(default_factory=list)
    context: str
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ExtractionResponse(StrictModel):
    success: bool
    document: dict[str, Any]


class ErrorResponse(StrictModel):
    error: Literal[True] = True
    code: str
    message: str
    request_id: str


class ServiceHealth(StrictModel):
    status: Literal["alive", "ready", "not_ready"]
    checks: dict[str, bool | str] = Field(default_factory=dict)
