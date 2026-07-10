"""Reusable cache-key builders for OCR artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .file_handler import FileHandler


def build_image_cache_key(file_path: Path) -> str:
    """Build the cache key used for direct image OCR."""
    return _hash_key(FileHandler.get_file_identity_string(file_path))


def build_pdf_page_cache_key(
    *,
    file_path: Path,
    page_index: int,
    languages: list[str],
    engine_name: str,
    confidence_threshold: float,
) -> str:
    """Build the cache key used for rendered PDF page OCR."""
    key_string = "|".join(
        [
            FileHandler.get_file_identity_string(file_path),
            str(page_index),
            ",".join(languages),
            engine_name,
            str(confidence_threshold),
            "pdf_page_v1",
        ]
    )
    return _hash_key(key_string)


def _hash_key(key_string: str) -> str:
    """Hash a human-readable cache identity into a stable file name."""
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()
