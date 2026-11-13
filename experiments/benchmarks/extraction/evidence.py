"""Durable local prediction evidence, deliberately separate from uploaded tables."""

from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file


def rejected_diagnostic(value, *, max_nodes=4096, max_text=65536):
    """Bounded JSON-safe evidence only; never a repaired canonical prediction."""
    ancestors = set()
    remaining = max_nodes
    text_left = max_text

    def visit(item, depth=0):
        nonlocal remaining, text_left
        remaining -= 1
        if remaining < 0 or depth > 32:
            return {"diagnostic": "truncated"}
        if item is None or type(item) in (bool, int):
            return (
                {"diagnostic": "oversized_integer"}
                if type(item) is int and item.bit_length() > 4096
                else item
            )
        if isinstance(item, str):
            kept = item[:text_left]
            text_left -= len(kept)
            return (
                kept
                if len(kept) == len(item)
                else {"diagnostic": "truncated_text", "prefix": kept}
            )
        if isinstance(item, float):
            return (
                item
                if math.isfinite(item)
                else {"diagnostic": "nonfinite", "value": str(item)}
            )
        if isinstance(item, (dict, list, tuple)):
            if id(item) in ancestors:
                return {"diagnostic": "cycle"}
            ancestors.add(id(item))
            try:
                limit = max(0, remaining)
                if isinstance(item, dict):
                    entries = list(islice(item.items(), limit))
                    if any(not isinstance(key, str) for key, _ in entries):
                        return {
                            "diagnostic": "invalid_object_keys",
                            "entries": [
                                [visit(k, depth + 1), visit(v, depth + 1)]
                                for k, v in entries
                            ],
                        }
                    result = {}
                    for index, (key, child) in enumerate(entries):
                        safe_key = (
                            key if len(key) <= text_left else f"truncated-key-{index}"
                        )
                        text_left = max(0, text_left - len(safe_key))
                        result[safe_key] = visit(child, depth + 1)
                    if len(entries) < len(item):
                        result["diagnostic_truncated_entries"] = True
                    return result
                result = [visit(child, depth + 1) for child in islice(item, limit)]
                if len(result) < len(item):
                    result.append({"diagnostic": "truncated_entries"})
                return result
            finally:
                ancestors.remove(id(item))
        return {"diagnostic": "unsupported", "type": type(item).__name__[:128]}

    return visit(value)


def observed_resources(evidence, *, device):
    """Keep valid worker observations without turning a telemetry gap into success."""
    result = {}
    for name in ("peak_process_tree_ram_mb", "peak_vram_mb"):
        value = evidence.get(name)
        result[name] = None
        if name == "peak_vram_mb" and device == "cpu":
            continue
        try:
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                result[name] = float(value)
        except OverflowError:
            pass  # Impossible observations must not erase completed predictions.
    return result


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
