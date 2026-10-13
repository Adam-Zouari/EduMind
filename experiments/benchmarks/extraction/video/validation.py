"""Full video-data checks and cheap sealed-report verification before execution."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.protocol import number
from experiments.benchmarks.extraction.audio.evaluate import normalize_transcript
from experiments.benchmarks.extraction.audio.validation import (
    validate_reference_segments,
)
from experiments.benchmarks.extraction.media import (
    canonical_wav_duration,
    decode_canonical_audio,
    ffmpeg_version,
    media_duration,
)
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
        "canonical_audio": {
            "sample_rate_hz": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
        },
        "ffmpeg_version": ffmpeg_version(),
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
    assets = set()
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
        if (
            not isinstance(item["id"], str)
            or not item["id"].strip()
            or item["id"] in metadata
        ):
            raise ValueError("Video IDs must be nonempty unique strings")
        if item["asset_sha256"] in assets:
            raise ValueError(
                "Video preparation requires unique video assets, not duplicate rows"
            )
        assets.add(item["asset_sha256"])
        source = PROJECT_ROOT / str(item["source_path"])
        if sha256_file(source) != item["asset_sha256"]:
            raise ValueError("Video asset checksum mismatch")
        duration = number(
            item["duration_seconds"],
            "Video duration",
            minimum=0,
            minimum_exclusive=True,
        )
        actual_duration = media_duration(source)
        if (
            not math.isfinite(duration)
            or duration <= 0
            or not math.isfinite(actual_duration)
            or actual_duration <= 0
            or abs(actual_duration - duration)
            > protocol.manifest_duration_tolerance_seconds
        ):
            raise ValueError("Video duration is invalid or differs from its asset")
        if profile != "smoke" and any(
            not item.get(field)
            for field in ("source_license", "source_revision", "document_family")
        ):
            raise ValueError("Video source provenance is incomplete")
        if profile != "smoke":
            group = item.get("independent_source_id") or item.get("source_group_id")
            if not isinstance(group, str) or not group.strip():
                raise ValueError(
                    "Video requires a reviewed independent source/group ID"
                )
        if (
            "nonspeech_reviewed" in item
            and type(item["nonspeech_reviewed"]) is not bool
        ):
            raise ValueError("Video nonspeech review flag must be boolean")
        if item.get("nonspeech_reviewed") is True and item.get(
            "nonspeech_window_indices"
        ):
            raise ValueError(
                "Do not mix whole-video and ASR-window nonspeech review units"
            )
        if protocol.nonspeech_unit_type == "whole_video" and item.get(
            "nonspeech_window_indices"
        ):
            raise ValueError(
                "Reviewed nonspeech windows contradict the frozen whole-video unit type"
            )
        if (
            protocol.nonspeech_unit_type == "asr_window"
            and item.get("nonspeech_reviewed") is True
        ):
            raise ValueError(
                "Reviewed nonspeech video contradicts the frozen ASR-window unit type"
            )
        # Video recognition does not claim a spoken timing task without reviewed
        # segments. Missing annotations still fail above; nonempty ones remain strict.
        if item["reference_segments"] != []:
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
                or any(
                    type(index) is not int or not 0 <= index < len(windows)
                    for index in indices
                )
                or len(set(indices)) != len(indices)
            ):
                raise ValueError(
                    "Reviewed nonspeech indices must identify actual unique ASR windows"
                )
            for index in indices:
                if (
                    normalize_transcript(item["reference_transcript"])
                    and not item["reference_segments"]
                ):
                    raise ValueError(
                        "Reviewed nonspeech windows cannot be verified against untimed speech"
                    )
                start = windows[index]
                end = min(duration, start + protocol.window_length_seconds)
                if any(
                    normalize_transcript(segment["text"])
                    and float(segment["start"]) < end
                    and float(segment["end"]) > start
                    for segment in item["reference_segments"]
                ):
                    raise ValueError(
                        "A reviewed nonspeech window overlaps spoken evidence"
                    )
            nonspeech += len(indices)
        metadata[str(item["id"])] = {
            "duration_seconds": duration,
            "actual_duration_seconds": actual_duration,
            "scheduled_window_count": len(windows),
            **_media_metadata(source, duration, protocol),
        }
    if (
        protocol.required_nonspeech_units is not None
        and nonspeech < protocol.required_nonspeech_units
    ):
        raise ValueError("Video corpus lacks the frozen reviewed nonspeech allocation")
    return {"samples": metadata, "nonspeech_unit_count": nonspeech}


def _media_metadata(source, duration, protocol):
    """Validate real streams/PCM outside all model timing and monitoring."""
    command = ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout).get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) > 1 or len(audio) > 1:
        raise ValueError(
            "Video benchmark sources require one visual and one audio track; remux the intended tracks before annotation"
        )
    if not videos or any(
        type(videos[0].get(name)) is not int or videos[0][name] <= 0
        for name in ("width", "height")
    ):
        raise ValueError("Video must have a decodable positive-dimension video stream")
    if not audio:
        raise ValueError(
            "Video has no audio stream; missing audio is not reviewed silence"
        )
    with tempfile.TemporaryDirectory(prefix="video-data-validation-") as raw:
        destination = Path(raw) / "canonical.wav"
        decode_command = decode_canonical_audio(
            source, destination, sample_rate_hz=16000, channels=1
        )
        actual = canonical_wav_duration(
            destination, sample_rate_hz=16000, channels=1, sample_width_bytes=2
        )
        if (
            not math.isfinite(actual)
            or actual <= 0
            or abs(actual - duration) > protocol.manifest_duration_tolerance_seconds
        ):
            raise ValueError(
                "Decoded video audio duration is invalid or differs from the video"
            )
        checksum = sha256_file(destination)
    return {
        "width": videos[0]["width"],
        "height": videos[0]["height"],
        "audio_stream_count": len(audio),
        "canonical_sha256": checksum,
        "canonical_duration_seconds": actual,
        "ffprobe_command": command,
        "ffmpeg_command": decode_command,
    }


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
