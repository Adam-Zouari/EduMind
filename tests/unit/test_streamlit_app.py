from __future__ import annotations

from pathlib import Path

import apps.streamlit_app as streamlit_app


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


class FakeBatchOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Path], bool, bool]] = []

    def process_batch(
        self,
        file_paths: list[str | Path],
        ingest_to_rag: bool = True,
        **kwargs: object,
    ) -> list[streamlit_app.ProcessedDocumentPayload]:
        self.calls.append(
            (
                [Path(file_path) for file_path in file_paths],
                ingest_to_rag,
                bool(kwargs.get("clean_text", True)),
            )
        )
        return [
            {
                "ocr_success": True,
                "ocr_error": None,
                "text": "First document",
                "metadata": {"page": 1},
                "file_path": str(file_paths[0]),
                "format_type": "pdf",
                "extraction_time": 0.3,
                "rag_ingested": True,
                "rag_chunks": 2,
                "rag_source_id": "doc-1",
                "rag_error": None,
            },
            {
                "ocr_success": True,
                "ocr_error": None,
                "text": "Second document",
                "metadata": {"page": 2},
                "file_path": str(file_paths[1]),
                "format_type": "docx",
                "extraction_time": 0.5,
                "rag_ingested": False,
                "rag_chunks": 0,
                "rag_source_id": None,
                "rag_error": "RAG unavailable",
            },
        ]


class FakeQueryOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []

    def query(
        self,
        *,
        query_text: str,
        top_k: int,
        generate_answer: bool,
    ) -> dict[str, object]:
        self.calls.append((query_text, top_k, generate_answer))
        if generate_answer:
            raise RuntimeError("ollama offline")
        return {
            "query": query_text,
            "results": [
                {
                    "source": "lesson.pdf",
                    "page": "3",
                    "score": 0.91,
                    "text": "Important retrieval result",
                    "metadata": {"page": 3},
                }
            ],
        }


def test_build_processed_file_record_adds_ui_fields() -> None:
    payload: streamlit_app.ProcessedDocumentPayload = {
        "ocr_success": True,
        "ocr_error": None,
        "text": "Study notes",
        "metadata": {"page": 4},
        "file_path": "tmp/lesson.pdf",
        "format_type": "pdf",
        "extraction_time": 0.25,
        "rag_ingested": True,
        "rag_chunks": 3,
        "rag_source_id": "source-1",
        "rag_error": None,
    }

    record = streamlit_app._build_processed_file_record(
        payload,
        filename="lesson.pdf",
        timestamp="2026-07-12 15:00:00",
    )

    assert record["filename"] == "lesson.pdf"
    assert record["timestamp"] == "2026-07-12 15:00:00"
    assert record["metadata"] == {"page": 4}
    assert record["rag_chunks"] == 3


def test_process_uploaded_files_uses_batch_path_and_preserves_order() -> None:
    orchestrator = FakeBatchOrchestrator()
    uploads = [
        FakeUploadedFile("lesson.pdf", b"first"),
        FakeUploadedFile("summary.docx", b"second"),
    ]

    records = streamlit_app._process_uploaded_files(
        orchestrator,
        uploads,
        ingest_to_rag=True,
        clean_text=False,
    )

    assert [record["filename"] for record in records] == ["lesson.pdf", "summary.docx"]
    assert [record["text"] for record in records] == ["First document", "Second document"]
    assert records[1]["rag_error"] == "RAG unavailable"
    assert len(orchestrator.calls) == 1
    batch_paths, ingest_to_rag, clean_text = orchestrator.calls[0]
    assert [path.name for path in batch_paths] == ["lesson.pdf", "summary.docx"]
    assert ingest_to_rag is True
    assert clean_text is False


def test_run_query_with_fallback_returns_retrieval_results() -> None:
    orchestrator = FakeQueryOrchestrator()

    display = streamlit_app._run_query_with_fallback(
        orchestrator,
        query="What should I revise?",
        top_k=4,
    )

    assert display["fallback_used"] is True
    assert display["warning"] is not None
    assert display["sources"][0]["source"] == "lesson.pdf"
    assert display["answer"] == "Answer generation is unavailable. Showing retrieved sources only."
    assert orchestrator.calls == [
        ("What should I revise?", 4, True),
        ("What should I revise?", 4, False),
    ]


def test_build_chat_history_record_preserves_sources_and_fallback() -> None:
    display: streamlit_app.QueryDisplayRecord = {
        "answer": "Fallback answer text",
        "sources": [
            {
                "source": "lesson.pdf",
                "page": "2",
                "score": 0.88,
                "text": "retrieved text",
                "metadata": {"page": 2},
            }
        ],
        "warning": "Answer generation failed.",
        "fallback_used": True,
    }

    record = streamlit_app._build_chat_history_record(
        query="Summarize this lesson",
        display=display,
    )

    assert record["query"] == "Summarize this lesson"
    assert record["fallback_used"] is True
    assert record["warning"] == "Answer generation failed."
    assert record["sources"][0]["page"] == "2"
