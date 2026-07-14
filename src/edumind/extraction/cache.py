"""Content-addressed extraction cache."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, local_file_lock, stable_hash

from .contracts import ExtractedDocument, ExtractionRequest

CACHE_SCHEMA = 1


class ExtractionCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def key(self, request: ExtractionRequest) -> str:
        if request.profile is None:
            raise ValueError("A resolved profile is required before caching")
        return stable_hash(
            {
                "schema": CACHE_SCHEMA,
                "source_checksum": request.checksum,
                "profile": request.profile.fingerprint,
                "options": request.options,
            }
        )

    def get(self, request: ExtractionRequest) -> ExtractedDocument | None:
        path = self.directory / f"{self.key(request)}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != CACHE_SCHEMA:
                return None
            document = ExtractedDocument.from_dict(payload["document"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        return replace(document, cache_hit=True)

    def put(self, request: ExtractionRequest, document: ExtractedDocument) -> Path:
        path = self.directory / f"{self.key(request)}.json"
        with local_file_lock(path.with_suffix(".lock")):
            atomic_write_json(path, {"schema": CACHE_SCHEMA, "document": document.to_dict()})
        return path
