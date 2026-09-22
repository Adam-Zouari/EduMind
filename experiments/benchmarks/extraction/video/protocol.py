"""Strict protocol for frozen-ASR and visual-only video benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    ProtocolMetadata,
    boolean,
    choice,
    execution_profiles,
    increasing_integers,
    integer,
    load_yaml,
    mapping,
    metadata,
    number,
    sequence,
    strict_object,
    string,
)

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class VideoProtocol:
    meta: ProtocolMetadata
    fixed_intervals: tuple[int, ...]
    scene_thresholds: tuple[float, ...]
    hybrid_gaps: tuple[int, ...]
    include_frame_zero: bool
    frame_sync: str
    smoke_scene_threshold: float
    selected_scene_threshold: float | None
    selected_scene_source_run_id: str | None
    window_length_seconds: float
    overlap_seconds: float
    stitching: Mapping[str, object]
    occurrence_matching: Mapping[str, object]
    video_counts: Mapping[str, int]
    manifest_duration_tolerance_seconds: float
    confidence_level: float
    maximum_finalists: int

    def profile(self, name: str):
        return self.meta.profile(name)

    def hybrid_threshold(self, profile: str) -> float:
        if profile == "smoke":
            return self.smoke_scene_threshold
        if (
            self.selected_scene_threshold is None
            or not self.selected_scene_source_run_id
        ):
            raise ValueError(
                "Hybrid video execution requires selected_scene_threshold and "
                "selected_scene_source_run_id in protocol.yaml"
            )
        if self.selected_scene_threshold not in self.scene_thresholds:
            raise ValueError(
                "Selected video scene threshold is outside the declared range"
            )
        return self.selected_scene_threshold


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> VideoProtocol:
    return protocol_from_mapping(load_yaml(path, "video"), source_path=path)


def protocol_from_mapping(
    value: object, *, source_path: Path = DEFAULT_PROTOCOL_PATH
) -> VideoProtocol:
    root = strict_object(
        value,
        "video protocol root",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "keyframes",
            "asr",
            "visual_text",
            "datasets",
            "statistics",
            "selection",
            "profiles",
        },
    )
    keyframes = strict_object(
        root["keyframes"],
        "keyframes",
        {
            "fixed_intervals_seconds",
            "scene_thresholds",
            "hybrid_maximum_gaps_seconds",
            "include_frame_zero",
            "frame_sync",
            "smoke_hybrid_scene_threshold",
            "selected_scene_threshold",
            "selected_scene_source_run_id",
        },
    )
    fixed = increasing_integers(
        keyframes["fixed_intervals_seconds"], "keyframes.fixed_intervals_seconds"
    )
    gaps = increasing_integers(
        keyframes["hybrid_maximum_gaps_seconds"],
        "keyframes.hybrid_maximum_gaps_seconds",
    )
    scenes = tuple(
        number(item, f"keyframes.scene_thresholds[{index}]", minimum=0, maximum=1)
        for index, item in enumerate(
            sequence(keyframes["scene_thresholds"], "keyframes.scene_thresholds")
        )
    )
    if not scenes or tuple(sorted(set(scenes))) != scenes:
        raise ValueError("Scene thresholds must be non-empty, unique, and increasing")
    include_zero = boolean(
        keyframes["include_frame_zero"], "keyframes.include_frame_zero"
    )
    if not include_zero:
        raise ValueError("Video protocol must include frame zero")
    frame_sync = choice(keyframes["frame_sync"], "keyframes.frame_sync", {"vfr"})
    smoke_threshold = number(
        keyframes["smoke_hybrid_scene_threshold"],
        "keyframes.smoke_hybrid_scene_threshold",
        minimum=0,
        maximum=1,
    )
    if smoke_threshold not in scenes:
        raise ValueError("Smoke hybrid threshold must be one of the scene candidates")
    selected_raw = keyframes["selected_scene_threshold"]
    selected = (
        None
        if selected_raw is None
        else number(
            selected_raw, "keyframes.selected_scene_threshold", minimum=0, maximum=1
        )
    )
    source_raw = keyframes["selected_scene_source_run_id"]
    source_run = (
        None
        if source_raw is None
        else string(source_raw, "keyframes.selected_scene_source_run_id")
    )
    if (selected is None) != (source_run is None):
        raise ValueError(
            "Selected scene threshold and source run ID must be set together"
        )
    if selected is not None and selected not in scenes:
        raise ValueError("Selected scene threshold must be in scene_thresholds")
    asr = strict_object(
        root["asr"],
        "asr",
        {
            "window_length_seconds",
            "overlap_seconds",
            "stitching_method",
            "stitching_normalization",
            "maximum_overlap_units",
        },
    )
    window = number(
        asr["window_length_seconds"],
        "asr.window_length_seconds",
        minimum=0,
        maximum=30,
        minimum_exclusive=True,
    )
    overlap = number(asr["overlap_seconds"], "asr.overlap_seconds", minimum=0)
    if overlap >= window:
        raise ValueError("Video ASR overlap must be smaller than its window")
    stitching = {
        "method": choice(
            asr["stitching_method"],
            "asr.stitching_method",
            {"normalized_suffix_prefix"},
        ),
        "normalization": choice(
            asr["stitching_normalization"],
            "asr.stitching_normalization",
            {"asr-minimal-v1"},
        ),
        "maximum_overlap_tokens": integer(
            asr["maximum_overlap_units"], "asr.maximum_overlap_units", minimum=1
        ),
    }
    visual = strict_object(
        root["visual_text"],
        "visual_text",
        {
            "unitization",
            "occurrence_content_f1_threshold",
            "frame_timestamp_tolerance_seconds",
        },
    )
    choice(
        visual["unitization"],
        "visual_text.unitization",
        {"normalized_lines_distinct_v1"},
    )
    occurrence = {
        "content_f1_threshold": number(
            visual["occurrence_content_f1_threshold"],
            "visual_text.occurrence_content_f1_threshold",
            minimum=0,
            maximum=1,
        ),
        "frame_timestamp_tolerance_seconds": number(
            visual["frame_timestamp_tolerance_seconds"],
            "visual_text.frame_timestamp_tolerance_seconds",
            minimum=0,
        ),
    }
    if occurrence["frame_timestamp_tolerance_seconds"] != 0:
        raise ValueError("Video occurrence timestamp tolerance must remain zero")
    datasets = strict_object(
        root["datasets"],
        "datasets",
        {"video_counts", "manifest_duration_tolerance_seconds"},
    )
    counts = strict_object(
        datasets["video_counts"],
        "datasets.video_counts",
        {"smoke", "development", "validation", "locked"},
    )
    video_counts = {
        name: integer(item, f"datasets.video_counts.{name}", minimum=1)
        for name, item in counts.items()
    }
    duration_tolerance = number(
        datasets["manifest_duration_tolerance_seconds"],
        "datasets.manifest_duration_tolerance_seconds",
        minimum=0,
    )
    statistics = strict_object(root["statistics"], "statistics", {"confidence_level"})
    confidence = number(
        statistics["confidence_level"],
        "statistics.confidence_level",
        minimum=0,
        maximum=1,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    selection = strict_object(root["selection"], "selection", {"maximum_finalists"})
    maximum_finalists = integer(
        selection["maximum_finalists"], "selection.maximum_finalists", minimum=1
    )
    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation", "locked")
    )
    if {profile.batch_size for profile in profiles.values()} != {1}:
        raise ValueError("Video visual profiles must process one sample at a time")
    return VideoProtocol(
        metadata("video", source_path, root, profiles=profiles),
        fixed,
        scenes,
        gaps,
        include_zero,
        frame_sync,
        smoke_threshold,
        selected,
        source_run,
        window,
        overlap,
        stitching,
        occurrence,
        video_counts,
        duration_tolerance,
        confidence,
        maximum_finalists,
    )


def protocol_from_worker(value: object) -> VideoProtocol:
    payload = mapping(value, "video worker protocol")
    root = mapping(payload.get("resolved"), "video resolved protocol")
    protocol = protocol_from_mapping(root)
    protocol.meta.validate_worker_payload(payload)
    return protocol
