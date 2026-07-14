"""Normalize extraction documents and API payloads into RAG ingest documents."""

from __future__ import annotations

from collections.abc import Mapping

from .types import IngestDocument, build_source_id, sanitize_filter_metadata


def normalize_ingest_document(document: Mapping[str, object]) -> IngestDocument:
    text = str(document.get("text", ""))
    if not text.strip():
        raise ValueError("Cannot ingest an empty document")
    metadata_value = document.get("metadata", {})
    metadata = dict(metadata_value) if isinstance(metadata_value, Mapping) else {}
    source = str(document.get("source") or metadata.get("source") or "uploaded-document")
    file_path = str(document.get("file_path") or metadata.get("source_path") or "") or None
    format_type = str(document.get("format_type") or metadata.get("source_kind") or "") or None
    source_id = str(
        document.get("source_id")
        or build_source_id(
            text=text,
            source=source,
            file_path=file_path,
            format_type=format_type,
            metadata=metadata,
        )
    )
    supplied_filters = document.get("filter_metadata")
    filters = (
        sanitize_filter_metadata(
            supplied_filters, source=source, format_type=format_type, file_path=file_path
        )
        if isinstance(supplied_filters, Mapping)
        else sanitize_filter_metadata(
            metadata, source=source, format_type=format_type, file_path=file_path
        )
    )
    return IngestDocument(text, source_id, source, format_type, file_path, metadata, filters)
