"""Deterministic source classification without heavyweight imports."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from .contracts import SourceKind
from .errors import UnsupportedSourceError

EXTENSION_KIND = {
    ".png": SourceKind.IMAGE,
    ".jpg": SourceKind.IMAGE,
    ".jpeg": SourceKind.IMAGE,
    ".tif": SourceKind.IMAGE,
    ".tiff": SourceKind.IMAGE,
    ".bmp": SourceKind.IMAGE,
    ".webp": SourceKind.IMAGE,
    ".pdf": SourceKind.PDF,
    ".docx": SourceKind.DOCX,
    ".wav": SourceKind.AUDIO,
    ".mp3": SourceKind.AUDIO,
    ".m4a": SourceKind.AUDIO,
    ".flac": SourceKind.AUDIO,
    ".ogg": SourceKind.AUDIO,
    ".mp4": SourceKind.VIDEO,
    ".mkv": SourceKind.VIDEO,
    ".mov": SourceKind.VIDEO,
    ".avi": SourceKind.VIDEO,
    ".webm": SourceKind.VIDEO,
}


def classify_source(path: Path, mime_type: str | None = None) -> tuple[SourceKind, str | None]:
    guessed_mime = mime_type or mimetypes.guess_type(path.name)[0]
    kind = EXTENSION_KIND.get(path.suffix.lower())
    if kind is None and guessed_mime:
        major = guessed_mime.split("/", 1)[0]
        if major == "image":
            kind = SourceKind.IMAGE
        elif major == "audio":
            kind = SourceKind.AUDIO
        elif major == "video":
            kind = SourceKind.VIDEO
        elif guessed_mime == "application/pdf":
            kind = SourceKind.PDF
    if kind is None:
        raise UnsupportedSourceError(
            f"Unsupported file type '{path.suffix or guessed_mime or 'unknown'}'. "
            "Supported sources are image, PDF, DOCX, audio, and video."
        )
    return kind, guessed_mime
