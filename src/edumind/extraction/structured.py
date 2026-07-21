"""Turn parser Markdown into offset-correct typed document elements."""

from __future__ import annotations

import re
from pathlib import Path

from .contracts import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    ExtractionRequest,
    ExtractionWarning,
    SegmentKind,
    SourceKind,
)


def build_markdown_document(
    request: ExtractionRequest,
    kind: SourceKind,
    profile: ExtractionProfile,
    pages: list[str],
    *,
    metadata: dict[str, object] | None = None,
    warnings: list[ExtractionWarning] | None = None,
    seconds: float = 0.0,
) -> ExtractedDocument:
    pieces: list[str] = []
    segments: list[ExtractedSegment] = []
    offset = 0
    for page_number, raw_page in enumerate(pages, 1):
        page = raw_page.strip()
        if pieces:
            pieces.append("\n")
            offset += 1
        page_start = offset
        pieces.append(page)
        offset += len(page)
        for start, end, segment_kind, structured in markdown_segments(page):
            segments.append(
                ExtractedSegment(
                    text=page[start:end],
                    start=page_start + start,
                    end=page_start + end,
                    page_number=page_number,
                    kind=segment_kind,
                    structured_content=structured,
                )
            )
        if page and not any(segment.page_number == page_number for segment in segments):
            segments.append(
                ExtractedSegment(
                    text=page,
                    start=page_start,
                    end=offset,
                    page_number=page_number,
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
        metadata={**(metadata or {}), "structured_output": True, "page_count": len(pages)},
        warnings=tuple(warnings or []),
        extraction_seconds=seconds,
    )


def markdown_segments(
    text: str,
) -> list[tuple[int, int, SegmentKind, dict[str, object]]]:
    candidates: list[tuple[int, int]] = []
    patterns = (
        r"(?s)\$\$.*?\$\$|\\\[.*?\\\]",
        r"(?is)<table\b.*?</table>",
        r"(?m)(?:^[^\n]*\|[^\n]*(?:\n|$)){2,}",
        r"(?m)^#{1,6}[ \t]+\S.*$",
        r"(?m)(?:^[ \t]*(?:[-*+] |\d+[.)] ).*(?:\n|$))+",
        r"(?im)^(?:table|figure|equation)\s+[A-Z0-9].*$",
    )
    for pattern in patterns:
        candidates.extend(match.span() for match in re.finditer(pattern, text))
    recognized: list[tuple[int, int]] = []
    for start, end in sorted(candidates, key=lambda span: (span[0], -span[1])):
        end = end - len(text[start:end]) + len(text[start:end].rstrip())
        if start >= end or any(start < used_end and end > used_start for used_start, used_end in recognized):
            continue
        block = text[start:end]
        kind, structured = _classify_block(block)
        if kind is SegmentKind.TEXT:
            continue
        recognized.append((start, end))

    result: list[tuple[int, int, SegmentKind, dict[str, object]]] = []
    cursor = 0
    for start, end in sorted(recognized):
        result.extend(_plain_segments(text, cursor, start))
        kind, structured = _classify_block(text[start:end])
        result.append((start, end, kind, structured))
        cursor = end
    result.extend(_plain_segments(text, cursor, len(text)))
    return result


def _plain_segments(
    text: str, start_offset: int, end_offset: int
) -> list[tuple[int, int, SegmentKind, dict[str, object]]]:
    result: list[tuple[int, int, SegmentKind, dict[str, object]]] = []
    fragment = text[start_offset:end_offset]
    for match in re.finditer(r"(?s)(?<!\S)\S.*?(?=\n[ \t]*\n|\Z)", fragment):
        start, end = match.span()
        block = match.group(0).strip()
        leading = len(match.group(0)) - len(match.group(0).lstrip())
        trailing = len(match.group(0)) - len(match.group(0).rstrip())
        start += start_offset + leading
        end += start_offset - trailing
        if start >= end:
            continue
        kind, structured = _classify_block(block)
        result.append((start, end, kind, structured))
    return result


def _classify_block(block: str) -> tuple[SegmentKind, dict[str, object]]:
    stripped = block.strip()
    if (stripped.startswith("$$") and stripped.endswith("$$")) or (
        stripped.startswith("\\[") and stripped.endswith("\\]")
    ):
        latex = stripped[2:-2].strip()
        return SegmentKind.FORMULA, {"latex": latex, "display": True}
    rows = _markdown_table_rows(stripped) or _html_table_rows(stripped)
    if rows is not None:
        return SegmentKind.TABLE, {
            "rows": rows,
            "row_count": len(rows),
            "column_count": max((len(row) for row in rows), default=0),
            "format": "html" if stripped.casefold().startswith("<table") else "markdown",
        }
    if re.match(r"^#{1,6}[ \t]+", stripped):
        level = len(stripped) - len(stripped.lstrip("#"))
        return SegmentKind.HEADING, {"level": level}
    lines = [line for line in stripped.splitlines() if line.strip()]
    if lines and all(re.match(r"^[ \t]*(?:[-*+] |\d+[.)] )", line) for line in lines):
        return SegmentKind.LIST_ITEM, {}
    if re.match(r"(?i)^(?:table|figure|equation)\s+[A-Z0-9]", stripped):
        return SegmentKind.CAPTION, {}
    return SegmentKind.TEXT, {}


def _markdown_table_rows(block: str) -> list[list[str]] | None:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 2 or not all("|" in line for line in lines):
        return None
    separator_index = next(
        (
            index
            for index, line in enumerate(lines[1:3], 1)
            if all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in line.strip("|").split("|"))
        ),
        None,
    )
    if separator_index is None:
        return None
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for index, line in enumerate(lines)
        if index != separator_index
    ]


def _html_table_rows(block: str) -> list[list[str]] | None:
    if not re.match(r"(?is)^<table\b", block) or not re.search(r"(?is)</table>\s*$", block):
        return None
    rows: list[list[str]] = []
    for row in re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", block):
        cells = [
            re.sub(r"\s+", " ", re.sub(r"(?is)<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"(?is)<t[dh]\b[^>]*>(.*?)</t[dh]>", row)
        ]
        if cells:
            rows.append(cells)
    return rows or None
