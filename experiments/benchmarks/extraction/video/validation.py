"""Full video-data checks and cheap sealed-report verification before execution."""

from __future__ import annotations

import math
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.extraction.audio.evaluate import normalize_transcript
from experiments.benchmarks.extraction.audio.validation import (
    validate_reference_segments,
)
from experiments.benchmarks.extraction.media import media_duration
from experiments.benchmarks.extraction.video.frozen_asr_worker import _window_starts
from experiments.benchmarks.extraction.video.metrics import (
    _occurrences,
    _reference_units,
)


def requirements(protocol, profile):
    return {
        "video_count": protocol.video_counts[profile],
        "duration_tolerance_seconds": protocol.manifest_duration_tolerance_seconds,
        "window_length_seconds": protocol.window_length_seconds,
        "overlap_seconds": protocol.overlap_seconds,
        "nonspeech_unit_type": protocol.nonspeech_unit_type,
        "required_nonspeech_units": protocol.required_nonspeech_units,
        "reference_schema": "video-v2",
    }


def validate(path, profile, protocol):
    items = [
        item for item in load_manifest(path).samples if item.get("kind") == "video"
    ]
    if len(items) != protocol.video_counts[profile]:
        raise ValueError(
            f"Video {profile} requires {protocol.video_counts[profile]} samples"
        )
    if profile != "smoke" and protocol.nonspeech_unit_type is None:
        raise ValueError(
            "Freeze the reviewed video nonspeech unit type and allocation before evaluation"
        )
    metadata, nonspeech = {}, 0
    for item in items:
        for field in (
            "id",
            "source_path",
            "asset_sha256",
            "duration_seconds",
            "reference_transcript",
            "reference_segments",
            "reference_visual_text",
            "visual_occurrences",
        ):
            if field not in item or item[field] is None:
                raise ValueError(f"Video sample lacks {field}")
        if not isinstance(item["reference_transcript"], str):
            raise ValueError(
                "Spoken video reference must be explicit text, including reviewed empty text"
            )
        source = PROJECT_ROOT / str(item["source_path"])
        if sha256_file(source) != item["asset_sha256"]:
            raise ValueError("Video asset checksum mismatch")
        duration = float(item["duration_seconds"])
        if (
            not math.isfinite(duration)
            or duration <= 0
            or abs(media_duration(source) - duration)
            > protocol.manifest_duration_tolerance_seconds
        ):
            raise ValueError("Video duration is invalid or differs from its asset")
        if profile != "smoke" and any(
            not item.get(field)
            for field in ("source_license", "source_revision", "document_family")
        ):
            raise ValueError("Video source provenance is incomplete")
        validate_reference_segments(
            {**item, "reference": item["reference_transcript"]}, duration
        )
        units = set(_reference_units(item["reference_visual_text"]))
        occurrences = _occurrences(item["visual_occurrences"], duration=duration)
        occurrence_units = {
            unit
            for occurrence in occurrences
            for unit in _reference_units(occurrence["text"])
        }
        if (
            any(
                len(_reference_units(occurrence["text"])) != 1
                for occurrence in occurrences
            )
            or units != occurrence_units
        ):
            raise ValueError(
                "Visible reference lines and occurrence annotations must describe the same complete units"
            )
        windows = _window_starts(
            duration, protocol.window_length_seconds, protocol.overlap_seconds
        )
        if (
            protocol.nonspeech_unit_type == "whole_video"
            and item.get("nonspeech_reviewed") is True
        ):
            if normalize_transcript(item["reference_transcript"]):
                raise ValueError(
                    "A reviewed nonspeech video contains spoken reference text"
                )
            nonspeech += 1
        if protocol.nonspeech_unit_type == "asr_window":
            indices = item.get("nonspeech_window_indices", [])
            if (
                not isinstance(indices, list)
                or len(set(indices)) != len(indices)
                or any(
                    type(index) is not int or not 0 <= index < len(windows)
                    for index in indices
                )
            ):
                raise ValueError(
                    "Reviewed nonspeech indices must identify actual unique ASR windows"
                )
            for index in indices:
                start = windows[index]
                end = min(duration, start + protocol.window_length_seconds)
                if any(
                    float(segment["start"]) < end and float(segment["end"]) > start
                    for segment in item["reference_segments"]
                ):
                    raise ValueError(
                        "A reviewed nonspeech window overlaps spoken evidence"
                    )
            nonspeech += len(indices)
        metadata[str(item["id"])] = {
            "duration_seconds": duration,
            "scheduled_window_count": len(windows),
        }
    if (
        protocol.required_nonspeech_units is not None
        and nonspeech < protocol.required_nonspeech_units
    ):
        raise ValueError("Video corpus lacks the frozen reviewed nonspeech allocation")
    return {"samples": metadata, "nonspeech_unit_count": nonspeech}


def prepare_report(path, profile, protocol, **options):
    return write_report(
        "video",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        lambda source, phase: validate(source, phase, protocol),
        **options,
    )


def verified_inputs(path, profile, protocol, **options):
    return verify_report(
        "video",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        **options,
    )
