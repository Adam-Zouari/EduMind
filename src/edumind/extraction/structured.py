"""Canonical structured-document conversion for extraction backends."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    BoundingBox,
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    ExtractionRequest,
    ExtractionWarning,
    SegmentKind,
    SourceKind,
)


def build_docling_document(
    request: ExtractionRequest,
    kind: SourceKind,
    profile: ExtractionProfile,
    document: Any,
    *,
    metadata: Mapping[str, object] | None = None,
    warnings: Sequence[ExtractionWarning] = (),
    seconds: float = 0.0,
) -> ExtractedDocument:
    """Convert a native DoclingDocument without inferring structure from Markdown."""

    elements: list[dict[str, object]] = []
    for order, (item, level) in enumerate(
        document.iterate_items(with_groups=True, traverse_pictures=True)
    ):
        label = _enum_value(getattr(item, "label", None)) or _enum_value(
            getattr(item, "name", None)
        )
        text, structured = _docling_content(item, document, label)
        page_number, bounding_box, provenance = _docling_provenance(item, document)
        parent = getattr(item, "parent", None)
        elements.append(
            {
                "text": text,
                "element_id": str(getattr(item, "self_ref", f"element-{order}")),
                "parent_id": str(getattr(parent, "cref", "")) or None,
                "order": order,
                "page_number": page_number,
                "bounding_box": bounding_box,
                "kind": _segment_kind(label),
                "structured_content": structured,
                "metadata": {
                    "label": label or "group",
                    "hierarchy_level": int(level),
                    "provenance": provenance,
                },
            }
        )
    return build_structured_document(
        request,
        kind,
        profile,
        elements,
        metadata=metadata,
        warnings=warnings,
        seconds=seconds,
    )


def build_structured_document(
    request: ExtractionRequest,
    kind: SourceKind,
    profile: ExtractionProfile,
    elements: Iterable[Mapping[str, object]],
    *,
    metadata: Mapping[str, object] | None = None,
    warnings: Sequence[ExtractionWarning] = (),
    seconds: float = 0.0,
) -> ExtractedDocument:
    """Build offset-correct text and segments from ordered native elements."""

    pieces: list[str] = []
    segments: list[ExtractedSegment] = []
    offset = 0
    page_numbers: set[int] = set()
    for fallback_order, raw in enumerate(elements):
        text = str(raw.get("text", ""))
        if text and pieces and not pieces[-1].endswith("\n\n"):
            pieces.append("\n\n")
            offset += 2
        start = offset
        if text:
            pieces.append(text)
            offset += len(text)
        page_number = _optional_int(raw.get("page_number"))
        if page_number is not None:
            page_numbers.add(page_number)
        structured = raw.get("structured_content", {})
        element_metadata = raw.get("metadata", {})
        order = _optional_int(raw.get("order"))
        segments.append(
            ExtractedSegment(
                text=text,
                start=start,
                end=offset,
                element_id=_optional_string(raw.get("element_id")),
                parent_id=_optional_string(raw.get("parent_id")),
                order=fallback_order if order is None else order,
                page_number=page_number,
                bounding_box=_bounding_box(raw.get("bounding_box")),
                kind=_coerce_kind(raw.get("kind")),
                structured_content=(
                    dict(structured) if isinstance(structured, Mapping) else {}
                ),
                metadata=(
                    dict(element_metadata)
                    if isinstance(element_metadata, Mapping)
                    else {}
                ),
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
        metadata={
            **dict(metadata or {}),
            "structured_output": True,
            "page_count": len(page_numbers) or (1 if kind is SourceKind.IMAGE else 0),
            "bounding_box_coordinates": "normalized-top-left",
        },
        warnings=tuple(warnings),
        extraction_seconds=seconds,
    )


def _docling_content(
    item: Any, document: Any, label: str | None
) -> tuple[str, dict[str, object]]:
    if label == "table" and hasattr(item, "data"):
        cells = []
        rows: list[list[str]] = [
            ["" for _ in range(int(getattr(item.data, "num_cols", 0)))]
            for _ in range(int(getattr(item.data, "num_rows", 0)))
        ]
        for cell in getattr(item.data, "table_cells", []):
            value = str(getattr(cell, "text", ""))
            row_start = int(getattr(cell, "start_row_offset_idx", 0))
            row_end = int(getattr(cell, "end_row_offset_idx", row_start + 1))
            column_start = int(getattr(cell, "start_col_offset_idx", 0))
            column_end = int(getattr(cell, "end_col_offset_idx", column_start + 1))
            cells.append(
                {
                    "text": value,
                    "row_start": row_start,
                    "row_end": row_end,
                    "column_start": column_start,
                    "column_end": column_end,
                    "row_span": int(getattr(cell, "row_span", row_end - row_start)),
                    "column_span": int(
                        getattr(cell, "col_span", column_end - column_start)
                    ),
                    "column_header": bool(getattr(cell, "column_header", False)),
                    "row_header": bool(getattr(cell, "row_header", False)),
                }
            )
            for row in range(row_start, min(row_end, len(rows))):
                for column in range(column_start, min(column_end, len(rows[row]))):
                    rows[row][column] = value
        html = str(item.export_to_html(document, add_caption=False))
        markdown = str(item.export_to_markdown(document)).strip()
        return markdown, {"rows": rows, "cells": cells, "html": html}
    text = str(getattr(item, "text", "") or getattr(item, "orig", "") or "").strip()
    if label == "formula":
        return text, {"latex": text}
    return text, {}


def _docling_provenance(
    item: Any, document: Any
) -> tuple[int | None, BoundingBox | None, list[dict[str, object]]]:
    converted: list[dict[str, object]] = []
    for value in getattr(item, "prov", []) or []:
        page_number = int(getattr(value, "page_no", 0)) or None
        box = _normalized_docling_box(getattr(value, "bbox", None), page_number, document)
        converted.append(
            {
                "page_number": page_number,
                "bounding_box": list(box) if box else None,
                "character_span": list(getattr(value, "charspan", ()) or ()),
            }
        )
    first = converted[0] if converted else {}
    return (
        _optional_int(first.get("page_number")),
        _bounding_box(first.get("bounding_box")),
        converted,
    )


def _normalized_docling_box(
    box: Any, page_number: int | None, document: Any
) -> BoundingBox | None:
    if box is None or page_number is None:
        return None
    page = getattr(document, "pages", {}).get(page_number)
    size = getattr(page, "size", None)
    width = float(getattr(size, "width", 0.0))
    height = float(getattr(size, "height", 0.0))
    if width <= 0 or height <= 0:
        return None
    top_left = box.to_top_left_origin(height)
    return (
        _clamp(float(top_left.l) / width),
        _clamp(float(top_left.t) / height),
        _clamp(float(top_left.r) / width),
        _clamp(float(top_left.b) / height),
    )


def _segment_kind(label: str | None) -> SegmentKind:
    return {
        "title": SegmentKind.TITLE,
        "section_header": SegmentKind.HEADING,
        "list_item": SegmentKind.LIST_ITEM,
        "table": SegmentKind.TABLE,
        "formula": SegmentKind.FORMULA,
        "caption": SegmentKind.CAPTION,
        "picture": SegmentKind.FIGURE,
        "chart": SegmentKind.FIGURE,
        "code": SegmentKind.CODE,
        "page_header": SegmentKind.PAGE_HEADER,
        "page_footer": SegmentKind.PAGE_FOOTER,
    }.get(label or "", SegmentKind.TEXT)


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _coerce_kind(value: object) -> SegmentKind:
    if isinstance(value, SegmentKind):
        return value
    try:
        return SegmentKind(str(value))
    except ValueError:
        return SegmentKind.TEXT


def _bounding_box(value: object) -> BoundingBox | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _optional_string(value: object) -> str | None:
    return str(value) if value not in (None, "") else None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
