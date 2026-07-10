"""FastAPI service for the RAG pipeline."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from edumind.common.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    ServiceHealth,
)
from edumind.rag.errors import MetadataFilterError
from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.serializers import (
    serialize_answer_result,
    serialize_ingest_report,
    serialize_query_results,
)

app = FastAPI(title="EduMind RAG Service", version="0.1.0")


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Lazily construct the heavy RAG runtime on first real use."""
    return RAGPipeline(use_llm=True)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "RAG Service", "status": "running"}


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    return ServiceHealth(status="healthy")


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest) -> IngestResponse:
    try:
        report = get_rag_pipeline().ingest_document(_model_dump(request.document))
        return serialize_ingest_report(report)
    except MetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        if request.generate_answer:
            answer = get_rag_pipeline().generate_answer(
                query=request.query,
                top_k=request.top_k,
                filter_metadata=request.filter_metadata,
            )
            return _model_dump(serialize_answer_result(answer))

        results = get_rag_pipeline().query(
            query_text=request.query,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
        )
        return _model_dump(serialize_query_results(request.query, results))
    except MetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def get_stats():
    try:
        return get_rag_pipeline().get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/reset")
def reset_database() -> dict[str, object]:
    try:
        get_rag_pipeline().reset()
        return {"success": True, "message": "Database reset"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _model_dump(model: object) -> dict[str, object]:
    """Return a dict from either a Pydantic v1 or v2 model."""
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")()
    if hasattr(model, "dict"):
        return getattr(model, "dict")()
    raise TypeError(f"Unsupported model type: {type(model).__name__}")
