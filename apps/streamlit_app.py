"""Thin Streamlit view over the tested EduMind application controller."""

from __future__ import annotations

import logging

import streamlit as st

from edumind.pipeline import EduMindPipeline, ProgressEvent

from apps.controller import AppController, safe_error

LOGGER = logging.getLogger(__name__)


@st.cache_resource
def _controller() -> AppController:
    return AppController(EduMindPipeline(use_llm=True))


def _records() -> dict[str, dict[str, object]]:
    if "document_records" not in st.session_state:
        st.session_state.document_records = {}
    return st.session_state.document_records


def _render_readiness(controller: AppController) -> None:
    try:
        readiness = controller.readiness()
        generation_ready = bool(readiness.get("generation_ready"))
        if generation_ready:
            st.success("Extraction, index, and Ollama are ready.")
        else:
            st.warning(
                "Index is ready, but Ollama is unavailable. Start Ollama and install "
                "the configured model."
            )
    except Exception as exc:
        LOGGER.exception("Runtime readiness check failed")
        st.error(f"Runtime not ready: {exc}")


def _render_upload(controller: AppController) -> None:
    uploads = st.file_uploader(
        "Study material",
        type=[
            "pdf",
            "docx",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "wav",
            "mp3",
            "m4a",
            "flac",
            "mp4",
            "mkv",
            "mov",
        ],
        accept_multiple_files=True,
    )
    if st.button("Extract and index", type="primary", disabled=not uploads):
        progress_bar = st.progress(0.0)
        status = st.empty()

        def update(event: ProgressEvent) -> None:
            progress_bar.progress(event.progress)
            status.caption(event.message)

        for upload in uploads or []:
            record, processed = controller.process_upload(
                upload.name, bytes(upload.getbuffer()), _records(), progress=update
            )
            if not processed:
                st.info(f"{record.filename} was already indexed; duplicate upload skipped.")
            elif record.error:
                st.error(f"{record.filename}: {record.error}")
            else:
                st.success(
                    f"{record.filename}: {record.characters:,} characters, {record.chunks} chunks, "
                    f"{record.timings.get('total_seconds', 0):.2f}s"
                )
                for warning in record.warnings:
                    st.warning(warning)


def _render_query(controller: AppController) -> None:
    question = st.text_area("Question", placeholder="Ask a question grounded in your documents")
    top_k = st.select_slider("Maximum evidence blocks", options=[1, 3, 5, 10], value=5)
    if st.button("Answer", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving evidence and generating a cited answer..."):
                result = controller.query(question, top_k=top_k)
        except Exception as exc:
            LOGGER.exception("Query failed")
            st.error(f"Query failed: {safe_error(exc)}")
            return
        if result.answer:
            st.markdown(result.answer.answer)
            for warning in result.answer.warnings:
                st.warning(warning)
        st.caption(
            f"Retrieval {result.timings.get('retrieval_seconds', 0):.2f}s · "
            f"Generation {result.timings.get('generation_seconds', 0):.2f}s · "
            f"Total {result.timings.get('total_seconds', 0):.2f}s"
        )
        st.subheader("Cited evidence")
        for index, hit in enumerate(result.hits, start=1):
            with st.expander(f"[{index}] {hit.source} · page {hit.page}"):
                st.write(hit.document)
                st.caption(f"{hit.retrieval_method} · rank {hit.rank} · {hit.token_count} tokens")


def _render_documents(controller: AppController) -> None:
    st.subheader("Document status")
    if not _records():
        st.caption("No documents indexed in this session.")
    for record in _records().values():
        st.write(
            f"{record.get('filename')} — {record.get('status')} — {record.get('chunks', 0)} chunks"
        )
    confirmation = st.checkbox("I understand that reset deletes the Chroma collection")
    if st.button("Reset Chroma collection", disabled=not confirmation):
        controller.reset(_records())
        st.success("Chroma collection reset.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="EduMind", page_icon="📚", layout="wide")
    st.title("EduMind")
    st.caption("Local multimodal extraction and citation-grounded study assistance")
    try:
        controller = _controller()
    except Exception as exc:
        LOGGER.exception("Application startup failed")
        st.error(safe_error(exc))
        st.info(
            "Start the provisional database with `docker compose -f "
            "infrastructure/chroma.yml up -d`, then reload this page."
        )
        st.stop()
    _render_readiness(controller)
    upload_tab, query_tab, documents_tab = st.tabs(["Extract & index", "Ask", "Documents"])
    with upload_tab:
        _render_upload(controller)
    with query_tab:
        _render_query(controller)
    with documents_tab:
        _render_documents(controller)


if __name__ == "__main__":
    main()
