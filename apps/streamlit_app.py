"""Primary Streamlit app for the study assistant workflow."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypedDict, cast

import streamlit as st

from edumind.pipeline.orchestrator import OCRRAGOrchestrator, ProcessedDocumentPayload

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


class SourceRecord(TypedDict):
    """Normalized source payload for UI rendering and chat history."""

    source: str
    page: str
    score: float
    text: str
    metadata: dict[str, object]


class ProcessedFileRecord(TypedDict):
    """UI-facing representation of one processed upload."""

    filename: str
    timestamp: str
    ocr_success: bool
    ocr_error: str | None
    text: str
    metadata: dict[str, object]
    file_path: str
    format_type: str
    extraction_time: float
    rag_ingested: bool
    rag_chunks: int
    rag_source_id: str | None
    rag_error: str | None


class QueryDisplayRecord(TypedDict):
    """Normalized query result for answer or retrieval-only rendering."""

    answer: str
    sources: list[SourceRecord]
    warning: str | None
    fallback_used: bool


class ChatHistoryRecord(TypedDict):
    """Persisted chat interaction for the session history tab."""

    timestamp: str
    query: str
    answer: str
    sources: list[SourceRecord]
    fallback_used: bool
    warning: str | None


class UploadedFileLike(Protocol):
    """Minimal protocol used from Streamlit uploaded files."""

    name: str

    def getbuffer(self) -> memoryview: ...


def _current_timestamp() -> str:
    """Return the standard display timestamp for UI records."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def _coerce_str(value: object, default: str = "") -> str:
    """Normalize arbitrary payload values into strings."""
    return value if isinstance(value, str) else default


def _coerce_float(value: object, default: float = 0.0) -> float:
    """Normalize arbitrary payload values into floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_int(value: object, default: int = 0) -> int:
    """Normalize arbitrary payload values into integers."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_metadata(value: object) -> dict[str, object]:
    """Normalize arbitrary payload values into metadata dictionaries."""
    return dict(value) if isinstance(value, dict) else {}


def _ensure_session_state() -> None:
    """Create the Streamlit session keys used by the primary UI."""
    st.session_state.setdefault("orchestrator", None)
    st.session_state.setdefault("initialized", False)
    st.session_state.setdefault("processed_files", [])
    st.session_state.setdefault("chat_history", [])


def _initialize_pipeline() -> None:
    """Initialize the OCR plus RAG runtime on demand."""
    with st.spinner("Initializing OCR and RAG runtimes..."):
        st.session_state["orchestrator"] = OCRRAGOrchestrator(use_llm=True)
        st.session_state["initialized"] = True


def _get_orchestrator() -> OCRRAGOrchestrator | None:
    """Return the active orchestrator, if the user has initialized it."""
    orchestrator = st.session_state.get("orchestrator")
    if isinstance(orchestrator, OCRRAGOrchestrator):
        return orchestrator
    return None


def _get_processed_files() -> list[ProcessedFileRecord]:
    """Return the normalized processed-file records stored in session state."""
    processed_files = st.session_state.get("processed_files")
    if isinstance(processed_files, list):
        return cast(list[ProcessedFileRecord], processed_files)
    return []


def _get_chat_history() -> list[ChatHistoryRecord]:
    """Return the normalized chat-history records stored in session state."""
    chat_history = st.session_state.get("chat_history")
    if isinstance(chat_history, list):
        return cast(list[ChatHistoryRecord], chat_history)
    return []


def _set_processed_files(records: list[ProcessedFileRecord]) -> None:
    """Persist processed-file records back into session state."""
    st.session_state["processed_files"] = records


def _set_chat_history(records: list[ChatHistoryRecord]) -> None:
    """Persist chat-history records back into session state."""
    st.session_state["chat_history"] = records


def _processed_file_count() -> int:
    """Return the number of successfully tracked processed files."""
    return len(_get_processed_files())


def _total_chunk_count() -> int:
    """Return the total number of ingested chunks across the session."""
    return sum(file_info["rag_chunks"] for file_info in _get_processed_files())


def _supported_upload_types(orchestrator: OCRRAGOrchestrator) -> list[str] | None:
    """Read supported OCR upload types from orchestrator stats."""
    raw_formats = orchestrator.get_stats().get("ocr_formats", [])
    if not isinstance(raw_formats, list):
        return None
    formats = [str(value).lower() for value in raw_formats if str(value).strip()]
    return formats or None


def _build_processed_file_record(
    payload: ProcessedDocumentPayload,
    *,
    filename: str,
    timestamp: str,
) -> ProcessedFileRecord:
    """Convert one backend payload into the stable UI record shape."""
    return {
        "filename": filename,
        "timestamp": timestamp,
        "ocr_success": payload["ocr_success"],
        "ocr_error": payload["ocr_error"],
        "text": payload["text"],
        "metadata": dict(payload["metadata"]),
        "file_path": payload["file_path"],
        "format_type": payload["format_type"],
        "extraction_time": payload["extraction_time"],
        "rag_ingested": payload["rag_ingested"],
        "rag_chunks": payload["rag_chunks"],
        "rag_source_id": payload["rag_source_id"],
        "rag_error": payload["rag_error"],
    }


def _build_failed_file_record(
    *,
    filename: str,
    timestamp: str,
    error: Exception,
) -> ProcessedFileRecord:
    """Build a stable UI record for a batch-level upload failure."""
    return {
        "filename": filename,
        "timestamp": timestamp,
        "ocr_success": False,
        "ocr_error": str(error),
        "text": "",
        "metadata": {},
        "file_path": "",
        "format_type": Path(filename).suffix.lstrip("."),
        "extraction_time": 0.0,
        "rag_ingested": False,
        "rag_chunks": 0,
        "rag_source_id": None,
        "rag_error": None,
    }


def _build_source_record(payload: Mapping[str, object]) -> SourceRecord:
    """Normalize one answer or retrieval hit payload into a UI source record."""
    return {
        "source": _coerce_str(payload.get("source"), "Unknown"),
        "page": _coerce_str(payload.get("page"), "N/A"),
        "score": _coerce_float(payload.get("score")),
        "text": _coerce_str(payload.get("text")),
        "metadata": _coerce_metadata(payload.get("metadata")),
    }


def _normalize_sources(value: object) -> list[SourceRecord]:
    """Normalize query payload sources or retrieval hits into source records."""
    if not isinstance(value, list):
        return []
    return [
        _build_source_record(item)
        for item in value
        if isinstance(item, Mapping)
    ]


def _normalize_query_display(
    payload: Mapping[str, object],
    *,
    warning: str | None,
    fallback_used: bool,
) -> QueryDisplayRecord:
    """Convert answer or retrieval payloads into one stable UI query result."""
    sources = _normalize_sources(payload.get("sources"))
    if not sources:
        sources = _normalize_sources(payload.get("results"))

    answer = _coerce_str(payload.get("answer"))
    if fallback_used and not answer and sources:
        answer = "Answer generation is unavailable. Showing retrieved sources only."

    return {
        "answer": answer,
        "sources": sources,
        "warning": warning,
        "fallback_used": fallback_used,
    }


def _build_chat_history_record(
    *,
    query: str,
    display: QueryDisplayRecord,
) -> ChatHistoryRecord:
    """Build one chat-history record from a normalized query result."""
    return {
        "timestamp": _current_timestamp(),
        "query": query,
        "answer": display["answer"],
        "sources": list(display["sources"]),
        "fallback_used": display["fallback_used"],
        "warning": display["warning"],
    }


def _save_upload_to_temp_directory(
    filename: str,
    content: bytes | bytearray | memoryview,
    directory: Path,
) -> Path:
    """Persist one uploaded file to a unique temporary path for OCR processing."""
    base_name = Path(filename).name or "upload"
    target = directory / base_name
    counter = 1
    while target.exists():
        target = directory / f"{Path(base_name).stem}_{counter}{Path(base_name).suffix}"
        counter += 1

    target.write_bytes(bytes(content))
    return target


def _process_uploaded_files(
    orchestrator: OCRRAGOrchestrator,
    uploaded_files: Sequence[UploadedFileLike],
    *,
    ingest_to_rag: bool,
    clean_text: bool,
) -> list[ProcessedFileRecord]:
    """Persist uploads once, run batch OCR/RAG processing, and normalize results."""
    if not uploaded_files:
        return []

    timestamps = [_current_timestamp() for _ in uploaded_files]
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        temp_paths = [
            _save_upload_to_temp_directory(
                uploaded_file.name,
                uploaded_file.getbuffer(),
                directory,
            )
            for uploaded_file in uploaded_files
        ]

        try:
            file_paths: list[str | Path] = list(temp_paths)
            payloads = orchestrator.process_batch(
                file_paths,
                ingest_to_rag=ingest_to_rag,
                clean_text=clean_text,
            )
        except Exception as exc:
            return [
                _build_failed_file_record(
                    filename=uploaded_file.name,
                    timestamp=timestamps[index],
                    error=exc,
                )
                for index, uploaded_file in enumerate(uploaded_files)
            ]

    if len(payloads) != len(uploaded_files):
        mismatch_error = RuntimeError("Batch processing returned an unexpected result count.")
        return [
            _build_failed_file_record(
                filename=uploaded_file.name,
                timestamp=timestamps[index],
                error=mismatch_error,
            )
            for index, uploaded_file in enumerate(uploaded_files)
        ]

    return [
        _build_processed_file_record(
            payloads[index],
            filename=uploaded_file.name,
            timestamp=timestamps[index],
        )
        for index, uploaded_file in enumerate(uploaded_files)
    ]


def _run_query_with_fallback(
    orchestrator: OCRRAGOrchestrator,
    *,
    query: str,
    top_k: int,
) -> QueryDisplayRecord:
    """Prefer answer generation but fall back to retrieval-only results on failure."""
    try:
        payload = orchestrator.query(query_text=query, top_k=top_k, generate_answer=True)
        return _normalize_query_display(payload, warning=None, fallback_used=False)
    except Exception as exc:
        warning = f"Answer generation failed: {exc}. Showing retrieved sources only."
        payload = orchestrator.query(query_text=query, top_k=top_k, generate_answer=False)
        return _normalize_query_display(payload, warning=warning, fallback_used=True)


def _render_sidebar() -> None:
    """Render app controls and lightweight runtime stats."""
    st.sidebar.title("EduMind Study Assistant")
    st.sidebar.caption("OCR -> RAG -> answer generation")

    if not bool(st.session_state.get("initialized")):
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
        rag_stats = stats.get("rag", {})
        if isinstance(rag_stats, Mapping):
            st.sidebar.subheader("Runtime")
            st.sidebar.metric("Indexed chunks", _coerce_int(rag_stats.get("total_chunks")))
            st.sidebar.metric("Processed files", _processed_file_count())
            st.sidebar.write(
                f"Embedding model: `{_coerce_str(rag_stats.get('embedding_model'), 'unknown')}`"
            )
            st.sidebar.write(f"LLM enabled: `{bool(rag_stats.get('llm_enabled', False))}`")
            st.sidebar.write(f"Model loaded: `{bool(rag_stats.get('model_loaded', False))}`")

    if st.sidebar.button("Reset RAG index"):
        if orchestrator is not None:
            orchestrator.reset_rag()
        _set_processed_files([])
        _set_chat_history([])
        st.sidebar.success("RAG index cleared.")
        st.rerun()


def _render_upload_tab(orchestrator: OCRRAGOrchestrator) -> None:
    """Render document upload plus OCR ingest flow."""
    st.subheader("Upload study material")
    uploaded_files = st.file_uploader(
        "Choose documents or media files",
        type=_supported_upload_types(orchestrator),
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
    status.write("Preparing uploads...")
    progress_bar.progress(0.2)

    with st.spinner("Processing uploaded files..."):
        batch_results = _process_uploaded_files(
            orchestrator,
            uploaded_files,
            ingest_to_rag=ingest_to_rag,
            clean_text=clean_text,
        )

    successful_records = [
        record for record in batch_results if record["ocr_success"]
    ]
    if successful_records:
        _set_processed_files(_get_processed_files() + successful_records)

    progress_bar.progress(1.0)
    status.write("Processing complete.")
    _render_processing_results(batch_results)


def _render_result_details(record: ProcessedFileRecord, *, key_prefix: str) -> None:
    """Render the normalized details for one processed file record."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Format", record["format_type"] or "unknown")
    col2.metric("Extraction time", f"{record['extraction_time']:.2f}s")
    col3.metric("RAG chunks", record["rag_chunks"])

    if record["ocr_success"]:
        if record["rag_error"]:
            st.warning(f"OCR succeeded but RAG ingest failed: {record['rag_error']}")
        preview = record["text"]
        if preview:
            st.text_area(
                "Extracted text preview",
                preview[:1000],
                height=180,
                key=f"{key_prefix}_{record['filename']}_{record['timestamp']}",
            )
    else:
        st.error(record["ocr_error"] or "Unknown OCR error")


def _render_processing_results(results: Sequence[ProcessedFileRecord]) -> None:
    """Render the OCR and ingest outcome for each processed file."""
    st.subheader("Processing results")
    for record in results:
        label = f"{'Success' if record['ocr_success'] else 'Failed'}: {record['filename']}"
        with st.expander(label, expanded=not record["ocr_success"]):
            _render_result_details(record, key_prefix="preview")


def _render_sources(sources: Sequence[SourceRecord]) -> None:
    """Render normalized source entries for either answers or retrieval-only results."""
    if not sources:
        return

    st.subheader("Sources")
    for index, source in enumerate(sources, start=1):
        with st.expander(
            "Source "
            f"{index}: {source['source']} | page {source['page']} | "
            f"score {source['score']:.3f}"
        ):
            st.write(source["text"])
            if source["metadata"]:
                st.json(source["metadata"])


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

    try:
        with st.spinner("Searching and generating an answer..."):
            display = _run_query_with_fallback(orchestrator, query=query, top_k=top_k)
    except Exception as exc:  # pragma: no cover - UI-only path
        st.error(f"Query failed: {exc}")
        return

    if display["warning"]:
        st.warning(display["warning"])

    if display["answer"]:
        st.subheader("Answer")
        st.markdown(display["answer"])
    elif not display["sources"]:
        st.info("No relevant sources were found for this question.")

    _render_sources(display["sources"])

    chat_history = _get_chat_history()
    chat_history.append(_build_chat_history_record(query=query, display=display))
    _set_chat_history(chat_history)


def _render_history_tab() -> None:
    """Render prior question and answer history for the current session."""
    st.subheader("Chat history")
    chat_history = _get_chat_history()
    if not chat_history:
        st.info("No questions asked yet.")
        return

    for chat in reversed(chat_history):
        with st.expander(f"{chat['timestamp']} | {chat['query'][:80]}"):
            st.markdown(f"**Question:** {chat['query']}")
            if chat["warning"]:
                st.warning(chat["warning"])
            if chat["answer"]:
                st.markdown(f"**Answer:** {chat['answer']}")
            elif chat["fallback_used"]:
                st.markdown("**Answer:** Retrieval-only fallback.")

            if chat["sources"]:
                st.markdown("**Sources:**")
                for index, source in enumerate(chat["sources"], start=1):
                    st.write(
                        f"{index}. {source['source']} "
                        f"(page {source['page']}, score {source['score']:.3f})"
                    )

    if st.button("Clear chat history"):
        _set_chat_history([])
        st.rerun()


def _render_processed_files_tab() -> None:
    """Render the current session's processed file log."""
    st.subheader("Processed files")
    processed_files = _get_processed_files()
    if not processed_files:
        st.info("No files processed yet.")
        return

    for record in reversed(processed_files):
        with st.expander(record["filename"] or Path(record["file_path"]).name):
            col1, col2, col3 = st.columns(3)
            col1.metric("Format", record["format_type"] or "unknown")
            col2.metric("RAG chunks", record["rag_chunks"])
            col3.metric("OCR success", "yes" if record["ocr_success"] else "no")
            st.caption(record["timestamp"])
            _render_result_details(record, key_prefix="processed")


def main() -> None:
    """Run the primary Streamlit application."""
    st.set_page_config(page_title="EduMind Study Assistant", layout="wide")
    _ensure_session_state()

    st.title("EduMind Study Assistant")
    st.caption("Upload study material, build a local knowledge base, and ask grounded questions.")
    _render_sidebar()

    if not bool(st.session_state.get("initialized")):
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
