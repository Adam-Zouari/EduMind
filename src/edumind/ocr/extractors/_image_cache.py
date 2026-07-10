"""Cache storage helpers for OCR image results."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.base_extractor import ExtractionResult
from ..core.errors import CacheReadError, CacheWriteError


class ImageCacheStore:
    """Persist and retrieve OCR image cache payloads."""

    def __init__(self, cache_dir: Path | None) -> None:
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def read(self, cache_key: str) -> ExtractionResult:
        """Read a cached OCR result."""
        cache_file = self._cache_file(cache_key)
        if not cache_file.exists():
            raise CacheReadError(f"Cache file does not exist for key {cache_key}")

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CacheReadError(f"Failed to load OCR cache for key {cache_key}: {exc}") from exc

        return ExtractionResult.from_cache_dict(payload)

    def write(self, cache_key: str, result: ExtractionResult) -> None:
        """Write a cached OCR result."""
        cache_file = self._cache_file(cache_key)
        try:
            cache_file.write_text(
                json.dumps(result.to_cache_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise CacheWriteError(f"Failed to write OCR cache for key {cache_key}: {exc}") from exc

    def exists(self, cache_key: str) -> bool:
        """Return whether a cache entry exists."""
        try:
            return self._cache_file(cache_key).exists()
        except CacheReadError:
            return False

    def _cache_file(self, cache_key: str) -> Path:
        """Return the path of a cache entry file."""
        if self.cache_dir is None:
            raise CacheReadError("OCR caching is disabled")
        return self.cache_dir / f"{cache_key}.json"
