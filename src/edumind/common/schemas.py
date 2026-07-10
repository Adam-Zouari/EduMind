"""Shared request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExtractedDocumentPayload(BaseModel):
    text: str
    source: str | None = None
    format_type: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    document: ExtractedDocumentPayload


class IngestResponse(BaseModel):
    success: bool
    chunks: int
    source_id: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    generate_answer: bool = True
    filter_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievalHitResponse(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    source: str
    page: str


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievalHitResponse] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[RetrievalHitResponse] = Field(default_factory=list)
    context: str


class OCRExtractResponse(BaseModel):
    success: bool
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    format_type: str
    extraction_time: float


class ServiceHealth(BaseModel):
    status: str
