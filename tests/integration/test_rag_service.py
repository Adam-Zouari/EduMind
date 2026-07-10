from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import services.rag_service as rag_service
from edumind.common.schemas import ExtractedDocumentPayload, IngestRequest, QueryRequest
from edumind.rag.types import AnswerResult, IngestReport, RetrievalHit


class FakeRAGPipeline:
    def __init__(self) -> None:
        self.ingested_document: dict[str, object] | None = None
        self.query_calls: list[tuple[str, int, dict[str, object]]] = []

    def ingest_document(self, document: dict[str, object]) -> IngestReport:
        self.ingested_document = document
        return IngestReport(source_id="doc-1", source="lesson.pdf", chunks_created=3)

    def query(
        self,
        *,
        query_text: str,
        top_k: int,
        filter_metadata: dict[str, object] | None = None,
    ) -> list[RetrievalHit]:
        self.query_calls.append((query_text, top_k, filter_metadata or {}))
        return [
            RetrievalHit(
                id="chunk-1",
                document="Important source text",
                metadata={"source": "lesson.pdf", "page": 2},
                score=0.88,
            )
        ]

    def generate_answer(
        self,
        *,
        query: str,
        top_k: int,
        filter_metadata: dict[str, object] | None = None,
    ) -> AnswerResult:
        self.query_calls.append((query, top_k, filter_metadata or {}))
        return AnswerResult(
            answer="Generated answer",
            sources=[
                RetrievalHit(
                    id="chunk-1",
                    document="Important source text",
                    metadata={"source": "lesson.pdf", "page": 2},
                    score=0.88,
                )
            ],
            context="[Document 1]\nImportant source text",
        )

    def get_stats(self) -> dict[str, object]:
        return {"total_chunks": 3}

    def reset(self) -> None:
        return None


def test_health_does_not_touch_pipeline(monkeypatch) -> None:
    def fail_if_called():
        raise AssertionError("health should not initialize the RAG pipeline")

    monkeypatch.setattr(rag_service, "get_rag_pipeline", fail_if_called)

    assert rag_service.health().status == "healthy"


def test_ingest_endpoint_uses_nested_document_contract(monkeypatch) -> None:
    fake_pipeline = FakeRAGPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda: fake_pipeline)
    request = IngestRequest(
        document=ExtractedDocumentPayload(
            text="Study this chapter",
            source="lesson.pdf",
            format_type="pdf",
            file_path="lesson.pdf",
            metadata={"page": 4},
        )
    )

    response = rag_service.ingest_document(request)

    assert response.success is True
    assert response.chunks == 3
    assert fake_pipeline.ingested_document is not None
    assert fake_pipeline.ingested_document["source"] == "lesson.pdf"
    assert fake_pipeline.ingested_document["metadata"] == {"page": 4}


def test_query_endpoint_serializes_results(monkeypatch) -> None:
    fake_pipeline = FakeRAGPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda: fake_pipeline)
    request = QueryRequest(
        query="What should I revise?",
        top_k=2,
        generate_answer=False,
        filter_metadata={"page": 2},
    )

    payload = rag_service.query_documents(request)

    assert payload["query"] == "What should I revise?"
    assert payload["results"][0]["source"] == "lesson.pdf"
    assert fake_pipeline.query_calls[0][2] == {"page": 2}


def test_query_endpoint_serializes_answers(monkeypatch) -> None:
    fake_pipeline = FakeRAGPipeline()
    monkeypatch.setattr(rag_service, "get_rag_pipeline", lambda: fake_pipeline)
    request = QueryRequest(
        query="Summarize this lesson",
        top_k=1,
        generate_answer=True,
    )

    payload = rag_service.query_documents(request)

    assert payload["answer"] == "Generated answer"
    assert payload["sources"][0]["page"] == "2"
