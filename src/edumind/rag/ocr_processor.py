"""Normalization helpers for OCR-to-RAG ingest payloads."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from .types import IngestDocument, build_source_id, sanitize_filter_metadata

logger = logging.getLogger(__name__)

DOCUMENT_BASE_FIELDS = {"text", "source", "format_type", "file_path", "metadata"}


class OCRProcessor:
    """Normalize OCR output into typed ingest documents."""

    def normalize_document(
        self,
        document: Mapping[str, object],
        *,
        default_source: str | None = None,
        document_index: int = 0,
    ) -> IngestDocument:
        """Normalize one OCR payload into a typed ingest document."""
        text_value = document.get("text", "")
        text = str(text_value).strip() if text_value is not None else ""
        if not text:
            raise ValueError("Document text is required for RAG ingestion")

        raw_metadata: dict[str, object] = {}
        nested_metadata = document.get("metadata")
        if isinstance(nested_metadata, Mapping):
            raw_metadata.update(dict(nested_metadata))

        for key, value in document.items():
            if key not in DOCUMENT_BASE_FIELDS:
                raw_metadata[key] = value

        source = _resolve_string(document.get("source")) or _resolve_string(
            raw_metadata.get("source")
        )
        if not source:
            source = default_source or f"document-{document_index}"

        format_type = _resolve_string(document.get("format_type")) or _resolve_string(
            raw_metadata.get("format_type")
        )
        file_path = _resolve_string(document.get("file_path")) or _resolve_string(
            raw_metadata.get("file_path")
        )

        normalized_metadata = dict(raw_metadata)
        normalized_metadata.setdefault("source", source)
        if format_type:
            normalized_metadata.setdefault("format_type", format_type)
        if file_path:
            normalized_metadata.setdefault("file_path", file_path)

        source_id = build_source_id(
            text=text,
            source=source,
            file_path=file_path,
            format_type=format_type,
            metadata=normalized_metadata,
        )

        return IngestDocument(
            text=text,
            source_id=source_id,
            source=source,
            format_type=format_type,
            file_path=file_path,
            metadata=normalized_metadata,
            filter_metadata=sanitize_filter_metadata(
                normalized_metadata,
                source=source,
                format_type=format_type,
                file_path=file_path,
            ),
        )

    def load_from_json(self, json_path: str | Path) -> list[IngestDocument]:
        """Load and normalize OCR JSON payloads from disk."""
        path = Path(json_path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        raw_documents: list[Mapping[str, object]]
        if isinstance(payload, list):
            raw_documents = [item for item in payload if isinstance(item, Mapping)]
        elif isinstance(payload, Mapping):
            raw_documents = [payload]
        else:
            logger.warning("Unsupported OCR JSON payload type: %s", type(payload).__name__)
            return []

        documents: list[IngestDocument] = []
        for index, raw_document in enumerate(raw_documents):
            try:
                documents.append(
                    self.normalize_document(
                        raw_document,
                        default_source=path.name,
                        document_index=index,
                    )
                )
            except ValueError as exc:
                logger.warning("Skipping OCR JSON document %s: %s", index, exc)

        logger.info("Loaded %s normalized documents from %s", len(documents), path)
        return documents


def _resolve_string(value: object) -> str | None:
    """Normalize optional string-like values."""
    if isinstance(value, str) and value:
        return value
    return None

