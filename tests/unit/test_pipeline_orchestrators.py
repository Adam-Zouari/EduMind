from __future__ import annotations

from pathlib import Path

import requests

from edumind.ocr.core.base_extractor import ExtractionResult
from edumind.pipeline.orchestrator import OCRRAGOrchestrator
from edumind.pipeline.orchestrator_api import APIOrchestrator
from edumind.rag.types import IngestReport


class FakeOCRPipeline:
    def __init__(self, results: list[ExtractionResult]) -> None:
        self.results = results
        self.extractors = {"image": object(), "pdf": object()}
        self.batch_calls: list[tuple[list[str | Path], dict[str, object]]] = []
        self.file_calls: list[tuple[str | Path, dict[str, object]]] = []

    def process_file(self, file_path: str | Path, **kwargs: object) -> ExtractionResult:
        self.file_calls.append((file_path, dict(kwargs)))
        return self.results[0]

    def process_batch(
        self,
        file_paths: list[str | Path],
        **kwargs: object,
    ) -> list[ExtractionResult]:
        self.batch_calls.append((list(file_paths), dict(kwargs)))
        return list(self.results)


class FakeRAGPipeline:
    def __init__(self, *, fail_ingest: bool = False) -> None:
        self.fail_ingest = fail_ingest
        self.ingested_documents: list[dict[str, object]] = []

    def ingest_document(self, document: dict[str, object]) -> IngestReport:
        if self.fail_ingest:
            raise RuntimeError("rag ingest failed")
        self.ingested_documents.append(document)
        return IngestReport(source_id="source-1", source="lesson.pdf", chunks_created=2)

    def get_stats(self) -> dict[str, object]:
        return {"total_chunks": len(self.ingested_documents)}

    def reset(self) -> None:
        self.ingested_documents.clear()


class FakeResponse:
    def __init__(
        self,
        payload: object | None = None,
        *,
        request_error: requests.RequestException | None = None,
        json_error: ValueError | None = None,
    ) -> None:
        self.payload = payload
        self.request_error = request_error
        self.json_error = json_error

    def raise_for_status(self) -> None:
        if self.request_error is not None:
            raise self.request_error

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    def __init__(
        self,
        responses: dict[tuple[str, str], FakeResponse | requests.RequestException],
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, int, dict[str, object]]] = []

    def request(self, method: str, url: str, timeout: int, **kwargs: object) -> FakeResponse:
        self.calls.append((method, url, timeout, dict(kwargs)))
        response = self.responses[(method, url)]
        if isinstance(response, requests.RequestException):
            raise response
        return response


def _make_result(
    *,
    file_path: str,
    text: str = "Study chapter one",
    success: bool = True,
    error: str | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        text=text,
        metadata={"page": 2},
        format_type="pdf",
        file_path=file_path,
        extraction_time=0.42,
        success=success,
        error=error,
    )


def test_local_orchestrator_process_file_ingests_and_returns_stable_payload() -> None:
    orchestrator = OCRRAGOrchestrator.__new__(OCRRAGOrchestrator)
    orchestrator.ocr_pipeline = FakeOCRPipeline([_make_result(file_path="notes/lesson.pdf")])
    orchestrator.rag_pipeline = FakeRAGPipeline()

    payload = OCRRAGOrchestrator.process_file(
        orchestrator,
        "notes/lesson.pdf",
        ingest_to_rag=True,
        clean_text=False,
    )

    assert payload["ocr_success"] is True
    assert payload["rag_ingested"] is True
    assert payload["rag_chunks"] == 2
    assert payload["rag_source_id"] == "source-1"
    assert payload["rag_error"] is None
    assert orchestrator.rag_pipeline.ingested_documents[0]["source"] == "lesson.pdf"
    assert orchestrator.ocr_pipeline.file_calls[0][1]["clean_text"] is False


def test_local_orchestrator_process_file_keeps_ocr_result_when_rag_fails() -> None:
    orchestrator = OCRRAGOrchestrator.__new__(OCRRAGOrchestrator)
    orchestrator.ocr_pipeline = FakeOCRPipeline([_make_result(file_path="lesson.pdf")])
    orchestrator.rag_pipeline = FakeRAGPipeline(fail_ingest=True)

    payload = OCRRAGOrchestrator.process_file(orchestrator, "lesson.pdf", ingest_to_rag=True)

    assert payload["ocr_success"] is True
    assert payload["rag_ingested"] is False
    assert payload["rag_chunks"] == 0
    assert payload["rag_error"] == "rag ingest failed"


def test_local_orchestrator_process_batch_uses_ocr_batch_path_and_preserves_order() -> None:
    orchestrator = OCRRAGOrchestrator.__new__(OCRRAGOrchestrator)
    first = _make_result(file_path="one.pdf", text="first")
    second = _make_result(file_path="two.pdf", text="second")
    orchestrator.ocr_pipeline = FakeOCRPipeline([first, second])
    orchestrator.rag_pipeline = FakeRAGPipeline()

    results = OCRRAGOrchestrator.process_batch(
        orchestrator,
        ["one.pdf", "two.pdf"],
        ingest_to_rag=False,
        clean_text=True,
    )

    assert [result["text"] for result in results] == ["first", "second"]
    assert orchestrator.ocr_pipeline.batch_calls == [
        (["one.pdf", "two.pdf"], {"clean_text": True})
    ]
    assert orchestrator.ocr_pipeline.file_calls == []


def test_local_orchestrator_stats_and_reset_aliases_are_consistent() -> None:
    orchestrator = OCRRAGOrchestrator.__new__(OCRRAGOrchestrator)
    orchestrator.ocr_pipeline = FakeOCRPipeline([_make_result(file_path="lesson.pdf")])
    orchestrator.rag_pipeline = FakeRAGPipeline()

    stats = OCRRAGOrchestrator.get_stats(orchestrator)
    OCRRAGOrchestrator.reset_database(orchestrator)

    assert stats["ocr_extractors"] == ["image", "pdf"]
    assert stats["ocr_formats"] == ["image", "pdf"]
    assert stats["rag"] == {"total_chunks": 0}


def test_api_orchestrator_process_file_returns_stable_payload(tmp_path: Path) -> None:
    file_path = tmp_path / "lesson.pdf"
    file_path.write_text("content", encoding="utf-8")
    session = FakeSession(
        {
            ("POST", "http://ocr/extract"): FakeResponse(
                {
                    "success": True,
                    "text": "Remote OCR text",
                    "metadata": {"page": 3},
                    "format_type": "pdf",
                    "extraction_time": 1.25,
                }
            ),
            ("POST", "http://rag/ingest"): FakeResponse(
                {"success": True, "chunks": 4, "source_id": "remote-source"}
            ),
        }
    )
    orchestrator = APIOrchestrator(
        ocr_url="http://ocr",
        rag_url="http://rag",
        session=session,
        verify_on_init=False,
    )

    payload = orchestrator.process_file(file_path, ingest_to_rag=True)

    assert payload["ocr_success"] is True
    assert payload["rag_ingested"] is True
    assert payload["rag_chunks"] == 4
    assert payload["rag_source_id"] == "remote-source"
    assert payload["rag_error"] is None
    assert session.calls[1][3]["json"]["document"]["source"] == "lesson.pdf"


def test_api_orchestrator_keeps_ocr_result_when_remote_rag_ingest_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "lesson.pdf"
    file_path.write_text("content", encoding="utf-8")
    session = FakeSession(
        {
            ("POST", "http://ocr/extract"): FakeResponse(
                {
                    "success": True,
                    "text": "Remote OCR text",
                    "metadata": {"page": 1},
                    "format_type": "pdf",
                    "extraction_time": 0.5,
                }
            ),
            ("POST", "http://rag/ingest"): requests.ConnectionError("rag unavailable"),
        }
    )
    orchestrator = APIOrchestrator(
        ocr_url="http://ocr",
        rag_url="http://rag",
        session=session,
        verify_on_init=False,
    )

    payload = orchestrator.process_file(file_path, ingest_to_rag=True)

    assert payload["ocr_success"] is True
    assert payload["rag_ingested"] is False
    assert payload["rag_chunks"] == 0
    assert payload["rag_source_id"] is None
    assert payload["rag_error"] is not None
    assert "Request failed: POST http://rag/ingest" in payload["rag_error"]


def test_api_orchestrator_health_stats_and_reset_alias_use_same_session() -> None:
    session = FakeSession(
        {
            ("GET", "http://ocr/health"): FakeResponse({"status": "healthy"}),
            ("GET", "http://rag/health"): FakeResponse({"status": "healthy"}),
            ("GET", "http://rag/stats"): FakeResponse({"total_chunks": 12}),
            ("GET", "http://ocr/formats"): FakeResponse({"formats": ["pdf", "png"]}),
            ("DELETE", "http://rag/reset"): FakeResponse({"success": True}),
        }
    )
    orchestrator = APIOrchestrator(
        ocr_url="http://ocr",
        rag_url="http://rag",
        session=session,
        verify_on_init=True,
    )

    stats = orchestrator.get_stats()
    reset_payload = orchestrator.reset_database()

    assert stats["rag"] == {"total_chunks": 12}
    assert stats["ocr_formats"] == ["pdf", "png"]
    assert stats["ocr_extractors"] == ["pdf", "png"]
    assert reset_payload == {"success": True}
    assert session.calls[0][:3] == ("GET", "http://ocr/health", 5)
    assert session.calls[1][:3] == ("GET", "http://rag/health", 5)
