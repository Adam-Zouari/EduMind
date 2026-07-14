"""Local-only FastAPI boundary for indexing and cited RAG answers."""

from __future__ import annotations

import logging
import threading
import uuid
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from edumind.common.schemas import IngestRequest, QueryRequest
from edumind.rag.errors import MetadataFilterError, RAGConfigurationError
from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.serializers import (
    serialize_answer_result,
    serialize_ingest_report,
    serialize_query_results,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="EduMind RAG Service", version="0.2.0")
_destructive_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline(use_llm=True)


def _dump(model: object) -> dict[str, object]:
    typed_model: Any = model
    return typed_model.model_dump() if hasattr(typed_model, "model_dump") else typed_model.dict()


def _error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": True, "code": code, "message": message, "request_id": str(uuid.uuid4())},
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("Invalid RAG request: %s", exc)
    return _error("invalid_request", "The request payload is invalid.", 422)


@app.get("/health/live")
def liveness() -> dict[str, object]:
    return {"status": "alive", "checks": {}}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    try:
        pipeline = get_rag_pipeline()
        stats = pipeline.get_stats()
        ollama = bool(pipeline.llm_generator and pipeline.llm_generator.health_check())
    except Exception as exc:
        logger.exception("RAG readiness failed: %s", exc)
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "checks": {"index": False}}
        )
    return JSONResponse(
        content={"status": "ready", "checks": {"index": True, "ollama": ollama, "stats": stats}}
    )


@app.get("/health")
def health() -> JSONResponse:
    return readiness()


@app.post("/ingest")
def ingest_document(request: IngestRequest) -> JSONResponse:
    try:
        report = get_rag_pipeline().ingest_document(_dump(request.document))
        return JSONResponse(content=_dump(serialize_ingest_report(report)))
    except (MetadataFilterError, ValueError) as exc:
        return _error("invalid_document", str(exc), 400)
    except RAGConfigurationError as exc:
        logger.error("RAG configuration error: %s", exc)
        return _error("not_ready", str(exc), 503)
    except Exception as exc:
        logger.exception("RAG ingest failed: %s", exc)
        return _error("internal_error", "Indexing failed. Check service logs for details.", 500)


@app.post("/query")
def query_documents(request: QueryRequest) -> JSONResponse:
    try:
        pipeline = get_rag_pipeline()
        if request.generate_answer:
            response: object = serialize_answer_result(
                pipeline.generate_answer(request.query, request.top_k, request.filter_metadata)
            )
        else:
            response = serialize_query_results(
                request.query, pipeline.query(request.query, request.top_k, request.filter_metadata)
            )
        return JSONResponse(content=_dump(response))
    except MetadataFilterError as exc:
        return _error("invalid_filter", str(exc), 400)
    except RAGConfigurationError as exc:
        return _error("not_ready", str(exc), 503)
    except Exception as exc:
        logger.exception("RAG query failed: %s", exc)
        return _error("internal_error", "Query failed. Check service logs for details.", 500)


@app.get("/stats")
def get_stats() -> JSONResponse:
    try:
        return JSONResponse(content=get_rag_pipeline().get_stats())
    except Exception as exc:
        logger.exception("Stats failed: %s", exc)
        return _error("not_ready", "RAG runtime is not ready.", 503)


@app.delete("/reset")
def reset_database() -> JSONResponse:
    if not _destructive_lock.acquire(blocking=False):
        return _error("operation_in_progress", "Another destructive operation is running.", 409)
    try:
        get_rag_pipeline().reset()
        return JSONResponse(content={"success": True, "message": "Index reset"})
    except Exception as exc:
        logger.exception("Index reset failed: %s", exc)
        return _error("internal_error", "Index reset failed.", 500)
    finally:
        _destructive_lock.release()
