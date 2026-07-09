"""FastAPI service for the RAG pipeline."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from edumind.common.schemas import IngestRequest, QueryRequest, ServiceHealth
from edumind.rag.rag_pipeline import RAGPipeline

app = FastAPI(title="EduMind RAG Service", version="0.1.0")
rag_pipeline = RAGPipeline(use_llm=True)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "RAG Service", "status": "running"}


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    return ServiceHealth(status="healthy")


@app.post("/ingest")
def ingest_document(request: IngestRequest) -> dict[str, object]:
    try:
        document = {"text": request.text, **request.metadata}
        chunks = rag_pipeline.ingest_document(document)
        return {"success": True, "chunks": chunks}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        if request.generate_answer:
            return rag_pipeline.generate_answer(query=request.query, top_k=request.top_k)
        return rag_pipeline.query(query_text=request.query, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
def get_stats():
    try:
        return rag_pipeline.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/reset")
def reset_database() -> dict[str, object]:
    try:
        rag_pipeline.reset()
        return {"success": True, "message": "Database reset"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
