from __future__ import annotations

from textwrap import dedent

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest

import edumind.pipeline.orchestrator as orchestrator_module


class FakeAppOrchestrator:
    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    def get_stats(self) -> dict[str, object]:
        return {
            "rag": {
                "total_chunks": 0,
                "embedding_model": "fake-embedder",
                "llm_enabled": self.use_llm,
                "model_loaded": False,
            },
            "ocr_formats": ["pdf", "png"],
        }

    def reset_rag(self) -> None:
        return None


def test_streamlit_app_renders_pre_initialization_state() -> None:
    app = AppTest.from_file("apps/streamlit_app.py")
    app.run()

    assert app.title[0].value == "EduMind Study Assistant"
    assert (
        app.info[0].value
        == "Initialize the workspace from the sidebar to start processing files."
    )


def test_streamlit_app_successful_initialization_shows_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator_module, "OCRRAGOrchestrator", FakeAppOrchestrator)

    app = AppTest.from_file("apps/streamlit_app.py")
    app.run()
    app.sidebar.button[0].click().run()
    app.run()

    assert app.session_state["initialized"] is True
    assert app.sidebar.success[0].value == "Pipeline ready"
    assert app.sidebar.subheader[0].value == "Runtime"


def test_streamlit_app_initialization_failure_shows_sidebar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingOrchestrator:
        def __init__(self, use_llm: bool = True) -> None:
            del use_llm
            raise RuntimeError("init failed")

    monkeypatch.setattr(orchestrator_module, "OCRRAGOrchestrator", FailingOrchestrator)

    app = AppTest.from_file("apps/streamlit_app.py")
    app.run()
    app.sidebar.button[0].click().run()

    assert app.session_state["initialized"] is False
    assert "Initialization failed: init failed" in app.sidebar.error[0].value


def test_render_processing_results_shows_ingest_warning_and_preview() -> None:
    script = dedent(
        """
        import apps.streamlit_app as appmod

        appmod._render_processing_results(
            [
                {
                    "filename": "lesson.pdf",
                    "timestamp": "2026-07-12 15:00:00",
                    "ocr_success": True,
                    "ocr_error": None,
                    "text": "Preview text for the study material.",
                    "metadata": {"page": 1},
                    "file_path": "lesson.pdf",
                    "format_type": "pdf",
                    "extraction_time": 0.4,
                    "rag_ingested": False,
                    "rag_chunks": 0,
                    "rag_source_id": None,
                    "rag_error": "RAG ingest failed",
                }
            ]
        )
        """
    )
    app = AppTest.from_string(script)
    app.run()

    assert "RAG ingest failed" in app.warning[0].value
    assert app.text_area[0].value == "Preview text for the study material."


def test_render_query_tab_handles_empty_index() -> None:
    script = dedent(
        """
        import streamlit as st
        import apps.streamlit_app as appmod

        appmod._ensure_session_state()
        st.session_state["processed_files"] = []

        class FakeOrchestrator:
            pass

        appmod._render_query_tab(FakeOrchestrator())
        """
    )
    app = AppTest.from_string(script)
    app.run()

    assert app.info[0].value == "Process at least one file before asking questions."
