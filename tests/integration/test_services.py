from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import httpx

from edumind.common.config import load_settings
from edumind.extraction import ExtractedDocument, ExtractedSegment, ExtractionProfile, SourceKind
from edumind.rag.types import AnswerResult, IngestReport, RetrievalHit
from services import extraction_service, rag_service


class ASGIClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

    def delete(self, path, **kwargs):
        return self.request("DELETE", path, **kwargs)


class FakeExtractionPipeline:
    def __init__(self):
        self.path = None

    def supported_sources(self):
        return {"image": ["fake"]}

    def extract(self, path):
        self.path = Path(path)
        return ExtractedDocument(
            "upload.png",
            str(path),
            SourceKind.IMAGE,
            "checksum",
            "image/png",
            "text",
            (ExtractedSegment("text", 0, 4, page_number=1),),
            ExtractionProfile("fake", "fake", "1"),
        )


def test_extraction_service_stream_limit_and_cleanup(monkeypatch) -> None:
    settings = load_settings()
    monkeypatch.setattr(
        extraction_service,
        "load_settings",
        lambda: replace(settings, extraction=replace(settings.extraction, maximum_upload_bytes=5)),
    )
    fake = FakeExtractionPipeline()
    monkeypatch.setattr(extraction_service, "get_extraction_pipeline", lambda: fake)
    client = ASGIClient(extraction_service.app)
    too_large = client.post("/extract", files={"file": ("x.png", b"123456", "image/png")})
    assert too_large.status_code == 413
    success = client.post("/extract", files={"file": ("x.png", b"1234", "image/png")})
    assert success.status_code == 200
    assert fake.path is not None and not fake.path.exists()
    assert success.json()["document"]["source_path"] == "x.png"


def test_services_separate_liveness_and_readiness(monkeypatch) -> None:
    extraction_client = ASGIClient(extraction_service.app)
    assert extraction_client.get("/health/live").json()["status"] == "alive"
    rag_client = ASGIClient(rag_service.app)
    assert rag_client.get("/health/live").json()["status"] == "alive"


class FakeGenerator:
    def health_check(self):
        return True


class FakeRAGPipeline:
    llm_generator = FakeGenerator()

    def __init__(self):
        self.reset_called = False

    def get_stats(self):
        return {"total_chunks": 1}

    def ingest_document(self, document):
        assert document["text"] == "alpha"
        return IngestReport("source-id", "notes.pdf", 1, elapsed_seconds=0.1)

    def query(self, query, top_k, filters):
        assert query == "alpha" and top_k == 3 and filters == {}
        return [RetrievalHit("1", "alpha", {"source": "notes.pdf", "page": 1}, 0.9)]

    def generate_answer(self, query, top_k, filters):
        hits = self.query(query, top_k, filters)
        return AnswerResult("alpha [1]", hits, "[1] alpha")

    def reset(self):
        self.reset_called = True


def test_rag_service_full_success_path_and_validation(monkeypatch) -> None:
    fake = FakeRAGPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda: fake)
    client = ASGIClient(rag_service.app)
    assert client.get("/health/ready").json()["checks"]["ollama"] is True
    assert client.get("/health").json()["status"] == "ready"
    assert client.get("/stats").json()["total_chunks"] == 1
    ingest = client.post(
        "/ingest",
        json={"document": {"text": "alpha", "source": "notes.pdf"}},
    )
    assert ingest.status_code == 200 and ingest.json()["chunks"] == 1
    query = client.post(
        "/query",
        json={"query": "alpha", "top_k": 3, "generate_answer": False},
    )
    assert query.json()["results"][0]["source"] == "notes.pdf"
    answer = client.post(
        "/query",
        json={"query": "alpha", "top_k": 3, "generate_answer": True},
    )
    assert answer.json()["answer"] == "alpha [1]"
    invalid = client.post("/query", json={"query": "", "unknown": True})
    assert invalid.status_code == 422 and invalid.json()["code"] == "invalid_request"
    assert client.delete("/reset").json()["success"] is True
    assert fake.reset_called is True


def test_rag_service_safe_failure_paths_and_reset_serialization(monkeypatch) -> None:
    monkeypatch.setattr(
        rag_service,
        "get_rag_pipeline",
        lambda: (_ for _ in ()).throw(RuntimeError("private details")),
    )
    client = ASGIClient(rag_service.app)
    assert client.get("/health/ready").status_code == 503
    assert client.get("/stats").json()["message"] == "RAG runtime is not ready."
    assert client.post("/ingest", json={"document": {"text": "alpha"}}).status_code == 500
    assert client.post("/query", json={"query": "alpha"}).status_code == 500
    assert client.delete("/reset").json()["message"] == "Index reset failed."

    assert rag_service._destructive_lock.acquire(blocking=False)
    try:
        assert client.delete("/reset").status_code == 409
    finally:
        rag_service._destructive_lock.release()
