"""Benchmarkable text normalization with offset reconstruction."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from .contracts import ExtractedDocument, ExtractedSegment


def normalize_text(text: str, profile: str) -> str:
    if profile not in {"minimal", "conservative", "aggressive"}:
        raise ValueError(f"Unknown normalization profile: {profile}")
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u0000", "").replace("\u00ad", "")
    if profile in {"conservative", "aggressive"}:
        normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)
        normalized = re.sub(r"[\t\f\v]+", " ", normalized)
        normalized = re.sub(r"[ ]{2,}", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    if profile == "aggressive":
        normalized = re.sub(r"(?m)^\s*(?:page\s+)?\d+\s*$", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\n{2,}", "\n", normalized)
    return normalized.strip()


def normalize_document(document: ExtractedDocument, profile: str) -> ExtractedDocument:
    source_segments = document.segments or (
        ExtractedSegment(text=document.text, start=0, end=len(document.text)),
    )
    segments: list[ExtractedSegment] = []
    pieces: list[str] = []
    offset = 0
    for original in source_segments:
        text = normalize_text(original.text, profile)
        if not text:
            continue
        if pieces:
            pieces.append("\n")
            offset += 1
        start = offset
        pieces.append(text)
        offset += len(text)
        segments.append(replace(original, text=text, start=start, end=offset))
    return replace(
        document,
        text="".join(pieces),
        segments=tuple(segments),
        profile=replace(document.profile, normalization=profile),
    )
