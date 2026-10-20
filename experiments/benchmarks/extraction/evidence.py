"""Durable local prediction evidence, deliberately separate from uploaded tables."""

from __future__ import annotations

import json
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file


def retain_prediction(path: Path, record: object) -> dict[str, object]:
    # Malformed native outputs are evidence too. Preserve non-JSON fragments as
    # diagnostic strings rather than dropping the entire failed observation.
    atomic_write_json(path, json.loads(json.dumps(record, default=str)))
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "uploaded": False}


def public_asr_evidence(value):
    """Strip prediction bodies recursively, including nested worker error envelopes."""
    if isinstance(value, dict):
        return {
            key: public_asr_evidence(item)
            for key, item in value.items()
            if key not in {"transcript", "segments", "raw_prediction"}
        }
    if isinstance(value, (list, tuple)):
        return [public_asr_evidence(item) for item in value]
    return value
