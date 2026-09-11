"""Frozen, versioned protocol inputs for video benchmark phases."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from edumind.common.artifacts import stable_hash


@dataclass(frozen=True)
class VideoProtocolLock:
    path: Path
    checksum: str
    window_length_seconds: float
    overlap_seconds: float
    stitching: Mapping[str, object]
    occurrence_matching: Mapping[str, object]
    hybrid_scene_threshold: float


def load_protocol_lock(
    path: Path, *, manifest_checksum: str, profile: str
) -> VideoProtocolLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"{path} must use VideoProtocolLock schema_version 1")
    required_strings = (
        "protocol_version",
        "scope",
        "manifest_checksum",
        "visible_text_unitization",
        "reviewer",
        "review_date",
    )
    missing = [name for name in required_strings if not str(payload.get(name, "")).strip()]
    if missing:
        raise ValueError(f"{path} lacks protocol fields: {', '.join(missing)}")
    if str(payload["manifest_checksum"]) != manifest_checksum:
        raise ValueError(
            f"{path} manifest checksum does not match the selected video manifest"
        )
    scope = str(payload["scope"])
    if profile == "smoke" and scope != "smoke":
        raise ValueError("Smoke video execution requires a smoke-scoped protocol lock")
    if profile != "smoke" and scope != "authoritative":
        raise ValueError(
            "Authoritative video execution requires a data-reviewed authoritative protocol lock"
        )
    try:
        date.fromisoformat(str(payload["review_date"]))
    except ValueError as exc:
        raise ValueError(f"{path} review_date must use YYYY-MM-DD") from exc

    window = _mapping(payload.get("asr_window"), "asr_window")
    length = float(window.get("length_seconds", 0))
    overlap = float(window.get("overlap_seconds", -1))
    if not 0 < length <= 30:
        raise ValueError("Video ASR window length must be greater than 0 and at most 30 seconds")
    if overlap < 0 or overlap >= length:
        raise ValueError("Video ASR overlap must satisfy 0 <= overlap < window length")

    stitching = _mapping(payload.get("stitching"), "stitching")
    if stitching.get("method") != "normalized_suffix_prefix":
        raise ValueError("Video stitching method must be normalized_suffix_prefix")
    if int(stitching.get("maximum_overlap_tokens", 0)) <= 0:
        raise ValueError("Video stitching requires a positive maximum_overlap_tokens")
    if stitching.get("normalization") != "asr-minimal-v1":
        raise ValueError("Video stitching normalization must be asr-minimal-v1")

    unitization = str(payload["visible_text_unitization"])
    if unitization != "normalized_lines_distinct_v1":
        raise ValueError(
            "Video visible-text unitization must be normalized_lines_distinct_v1"
        )
    matching = _mapping(payload.get("occurrence_matching"), "occurrence_matching")
    threshold = float(matching.get("content_f1_threshold", -1))
    tolerance = float(matching.get("frame_timestamp_tolerance_seconds", -1))
    if not 0 <= threshold <= 1 or tolerance != 0:
        raise ValueError("Video occurrence-matching thresholds are invalid")
    hybrid_threshold = float(payload.get("hybrid_scene_threshold", -1))
    if not 0 <= hybrid_threshold <= 1:
        raise ValueError("Video hybrid_scene_threshold must be between 0 and 1")
    return VideoProtocolLock(
        path.resolve(),
        stable_hash(payload),
        length,
        overlap,
        dict(stitching),
        dict(matching),
        hybrid_threshold,
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Video protocol {label} must be an object")
    return value
