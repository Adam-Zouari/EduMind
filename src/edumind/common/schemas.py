"""Shared request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    generate_answer: bool = True


class OCRExtractResponse(BaseModel):
    success: bool
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    format_type: str
    extraction_time: float


class ServiceHealth(BaseModel):
    status: str
