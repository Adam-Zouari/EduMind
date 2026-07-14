"""Helpers for building offset-correct extraction documents."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..contracts import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    ExtractionRequest,
    ExtractionWarning,
    SourceKind,
)


def build_document(
    request: ExtractionRequest,
    kind: SourceKind,
    profile: ExtractionProfile,
    texts: list[str],
    *,
    pages: list[int | None] | None = None,
    timestamps: Sequence[tuple[float | None, float | None]] | None = None,
    metadata: dict[str, object] | None = None,
    warnings: list[ExtractionWarning] | None = None,
    seconds: float = 0.0,
) -> ExtractedDocument:
    pieces: list[str] = []
    segments: list[ExtractedSegment] = []
    offset = 0
    for index, text in enumerate(texts):
        if pieces:
            pieces.append("\n")
            offset += 1
        start = offset
        pieces.append(text)
        offset += len(text)
        time_range = timestamps[index] if timestamps and index < len(timestamps) else (None, None)
        page = pages[index] if pages and index < len(pages) else None
        segments.append(
            ExtractedSegment(
                text=text,
                start=start,
                end=offset,
                page_number=page,
                timestamp_start=time_range[0],
                timestamp_end=time_range[1],
            )
        )
    return ExtractedDocument(
        source_name=Path(request.source_path).name,
        source_path=str(request.source_path),
        source_kind=kind,
        source_checksum=request.checksum,
        mime_type=request.mime_type,
        text="".join(pieces),
        segments=tuple(segments),
        profile=profile,
        metadata=metadata or {},
        warnings=tuple(warnings or []),
        extraction_seconds=seconds,
    )
