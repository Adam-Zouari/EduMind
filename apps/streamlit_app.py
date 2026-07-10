"""Primary Streamlit app for the study assistant workflow."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from edumind.pipeline.orchestrator import OCRRAGOrchestrator

SUPPORTED_UPLOAD_TYPES = [
    "pdf",
    "docx",
    "png",
    "jpg",
    "jpeg",
    "html",
    "mp3",
    "wav",
    "mp4",
    "avi",
]


def _ensure_session_state() -> None:
    """Create the Streamlit session keys used by the primary UI."""
    st.session_state.setdefault("orchestrator", None)
    st.session_state.setdefault("initialized", False)
    st.session_state.setdefault("processed_files", [])
    st.session_state.setdefault("chat_history", [])


def _initialize_pipeline() -> None:
    """Initialize the OCR plus RAG runtime on demand."""
    with st.spinner("Initializing OCR and RAG runtimes..."):
        st.session_state.orchestrator = OCRRAGOrchestrator(use_llm=True)
        st.session_state.initialized = True


def _get_orchestrator() -> OCRRAGOrchestrator | None:
    """Return the active orchestrator, if the user has initialized it."""
    orchestrator = st.session_state.get("orchestrator")
    if isinstance(orchestrator, OCRRAGOrchestrator):
        return orchestrator
    return None


def _processed_file_count() -> int:
    """Return the number of successfully tracked processed files."""
    processed_files = st.session_state.get("processed_files", [])
    return len(processed_files) if isinstance(processed_files, list) else 0


def _total_chunk_count() -> int:
    """Return the total number of ingested chunks across the session."""
    processed_files = st.session_state.get("processed_files", [])
    if not isinstance(processed_files, list):
        return 0
    return sum(int(file_info.get("rag_chunks", 0)) for file_info in processed_files)


def _render_sidebar() -> None:
    """Render app controls and lightweight runtime stats."""
    st.sidebar.title("EduMind Study Assistant")
    st.sidebar.caption("OCR -> RAG -> answer generation")

    if not st.session_state.initialized:
        if st.sidebar.button("Initialize workspace", type="primary"):
            try:
                _initialize_pipeline()
                st.sidebar.success("Pipeline ready.")
            except Exception as exc:  # pragma: no cover - UI-only path
                st.sidebar.error(f"Initialization failed: {exc}")
                st.sidebar.info("Make sure your local OCR and Ollama dependencies are installed.")
        return

    orchestrator = _get_orchestrator()
    st.sidebar.success("Pipeline ready")

    if orchestrator is not None:
        stats = orchestrator.get_stats()
        rag_stats = stats["rag"]
        st.sidebar.subheader("Runtime")
        st.sidebar.metric("Indexed chunks", int(rag_stats.get("total_chunks", 0)))
        st.sidebar.metric("Processed files", _processed_file_count())
        st.sidebar.write(f"Embedding model: `{rag_stats.get('embedding_model', 'unknown')}`")
        st.sidebar.write(f"LLM enabled: `{rag_stats.get('llm_enabled', False)}`")
        st.sidebar.write(f"Model loaded: `{rag_stats.get('model_loaded', False)}`")

    if st.sidebar.button("Reset RAG index"):
        if orchestrator is not None:
            orchestrator.reset_rag()
        st.session_state.processed_files = []
        st.session_state.chat_history = []
        st.sidebar.success("RAG index cleared.")
        st.rerun()


def _render_upload_tab(orchestrator: OCRRAGOrchestrator) -> None:
    """Render document upload plus OCR ingest flow."""
    st.subheader("Upload study material")
    uploaded_files = st.file_uploader(
        "Choose documents or media files",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
    )

    clean_text = st.checkbox("Clean extracted text", value=True)
    ingest_to_rag = st.checkbox("Add extracted text to RAG", value=True)

    if not uploaded_files:
        st.info("Upload one or more files to build your study knowledge base.")
        return

    if not st.button("Process files", type="primary"):
        return

    progress_bar = st.progress(0.0)
    status = st.empty()
    batch_results: list[dict[str, object]] = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"Processing `{uploaded_file.name}`...")
        temp_path = _save_upload_to_temp(uploaded_file.name, uploaded_file.getbuffer())
        try:
            result = orchestrator.process_file(
                file_path=temp_path,
                ingest_to_rag=ingest_to_rag,
                clean_text=clean_text,
            )
            result["filename"] = uploaded_file.name
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch_results.append(result)

            if bool(result.get("ocr_success")):
                st.session_state.processed_files.append(result)
        except Exception as exc:  # pragma: no cover - UI-only path
            batch_results.append(
                {
                    "filename": uploaded_file.name,
                    "ocr_success": False,
                    "ocr_error": str(exc),
                    "text": "",
                    "metadata": {},
                    "format_type": Path(uploaded_file.name).suffix.lstrip("."),
                    "extraction_time": 0.0,
                    "rag_ingested": False,
                    "rag_chunks": 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        finally:
            temp_path.unlink(missing_ok=True)

        progress_bar.progress(index / len(uploaded_files))

    status.write("Processing complete.")
    _render_processing_results(batch_results)


def _render_processing_results(results: list[dict[str, object]]) -> None:
    """Render the OCR and ingest outcome for each processed file."""
    st.subheader("Processing results")
    for result in results:
        filename = str(result.get("filename", "file"))
        success = bool(result.get("ocr_success"))
        label = f"{'Success' if success else 'Failed'}: {filename}"
        with st.expander(label, expanded=not success):
            col1, col2, col3 = st.columns(3)
            col1.metric("Format", str(result.get("format_type", "unknown")))
            col2.metric("Extraction time", f"{float(result.get('extraction_time', 0.0)):.2f}s")
            col3.metric("RAG chunks", int(result.get("rag_chunks", 0)))

            if success:
                preview = str(result.get("text", ""))
                st.text_area(
                    "Extracted text preview",
                    preview[:1000],
                    height=180,
                    key=f"preview_{filename}_{result.get('timestamp', '')}",
                )
            else:
                st.error(str(result.get("ocr_error", "Unknown OCR error")))


def _render_query_tab(orchestrator: OCRRAGOrchestrator) -> None:
    """Render the retrieval and answer-generation workflow."""
    st.subheader("Ask questions")
    if _total_chunk_count() == 0:
        st.info("Process at least one file before asking questions.")
        return

    st.caption(
        "Current session index: "
        f"{_total_chunk_count()} chunks from {_processed_file_count()} files."
    )
    query = st.text_input("Question", placeholder="What should I study first from these notes?")
    top_k = st.slider("Number of sources", min_value=1, max_value=10, value=5)

    if not st.button("Ask", type="primary"):
        return
    if not query.strip():
        st.warning("Enter a question first.")
        return

    with st.spinner("Searching and generating an answer..."):
        result = orchestrator.query(query_text=query, top_k=top_k, generate_answer=True)

    st.subheader("Answer")
    st.markdown(str(result.get("answer", "")))

    sources = result.get("sources", [])
    if isinstance(sources, list) and sources:
        st.subheader("Sources")
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                continue
            source_name = str(source.get("source", "Unknown"))
            page = str(source.get("page", "N/A"))
            score = float(source.get("score", 0.0))
            with st.expander(
                f"Source {index}: {source_name} | page {page} | score {score:.3f}"
            ):
                st.write(source.get("text", ""))
                metadata = source.get("metadata", {})
                if isinstance(metadata, dict) and metadata:
                    st.json(metadata)

    st.session_state.chat_history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": result.get("answer", ""),
            "sources": sources if isinstance(sources, list) else [],
        }
    )


def _render_history_tab() -> None:
    """Render prior question and answer history for the current session."""
    st.subheader("Chat history")
    chat_history = st.session_state.get("chat_history", [])
    if not isinstance(chat_history, list) or not chat_history:
        st.info("No questions asked yet.")
        return

    for chat in reversed(chat_history):
        if not isinstance(chat, dict):
            continue
        query = str(chat.get("query", "Question"))
        timestamp = str(chat.get("timestamp", ""))
        with st.expander(f"{timestamp} | {query[:80]}"):
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {chat.get('answer', '')}")
            sources = chat.get("sources", [])
            if isinstance(sources, list) and sources:
                st.markdown("**Sources:**")
                for index, source in enumerate(sources, start=1):
                    if isinstance(source, dict):
                        st.write(
                            f"{index}. {source.get('source', 'Unknown')} "
                            "(page "
                            f"{source.get('page', 'N/A')}, "
                            f"score {float(source.get('score', 0.0)):.3f})"
                        )

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()


def _render_processed_files_tab() -> None:
    """Render the current session's processed file log."""
    st.subheader("Processed files")
    processed_files = st.session_state.get("processed_files", [])
    if not isinstance(processed_files, list) or not processed_files:
        st.info("No files processed yet.")
        return

    for file_info in reversed(processed_files):
        if not isinstance(file_info, dict):
            continue
        filename = str(file_info.get("filename") or Path(str(file_info.get("file_path", ""))).name)
        with st.expander(filename):
            col1, col2, col3 = st.columns(3)
            col1.metric("Format", str(file_info.get("format_type", "unknown")))
            col2.metric("RAG chunks", int(file_info.get("rag_chunks", 0)))
            col3.metric("OCR success", "yes" if file_info.get("ocr_success") else "no")
            st.caption(str(file_info.get("timestamp", "")))
            preview = str(file_info.get("text", ""))
            if preview:
                st.text_area(
                    "Preview",
                    preview[:1000],
                    height=180,
                    key=f"processed_{filename}_{file_info.get('timestamp', '')}",
                )


def _save_upload_to_temp(filename: str, content: bytes) -> Path:
    """Persist one uploaded file to a temporary path for OCR processing."""
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(content)
        return Path(handle.name)


def main() -> None:
    """Run the primary Streamlit application."""
    st.set_page_config(page_title="EduMind Study Assistant", layout="wide")
    _ensure_session_state()

    st.title("EduMind Study Assistant")
    st.caption("Upload study material, build a local knowledge base, and ask grounded questions.")
    _render_sidebar()

    if not st.session_state.initialized:
        st.info("Initialize the workspace from the sidebar to start processing files.")
        return

    orchestrator = _get_orchestrator()
    if orchestrator is None:
        st.error("The application state is out of sync. Reinitialize the workspace.")
        return

    upload_tab, query_tab, history_tab, files_tab = st.tabs(
        ["Upload", "Ask", "History", "Processed files"]
    )
    with upload_tab:
        _render_upload_tab(orchestrator)
    with query_tab:
        _render_query_tab(orchestrator)
    with history_tab:
        _render_history_tab()
    with files_tab:
        _render_processed_files_tab()


if __name__ == "__main__":
    main()
