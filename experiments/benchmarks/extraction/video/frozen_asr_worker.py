"""One measured audio preparation per video, independent of visual candidates."""

from __future__ import annotations

import math
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from edumind.common.model_placement import inspect_model_placement
from experiments.benchmarks.common.process import (
    json_worker_main,
    seed_deterministically,
)
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import build_runtime
from experiments.benchmarks.extraction.audio.evaluate import (
    METRIC_DIRECTIONS as AUDIO_DIRECTIONS,
)
from experiments.benchmarks.extraction.audio.evaluate import (
    aggregate_rows,
    normalize_transcript,
    sample_metrics,
    score_speech,
    validate_prediction,
)
from experiments.benchmarks.extraction.audio.protocol import (
    protocol_from_worker as audio_protocol_from_worker,
)
from experiments.benchmarks.extraction.audio.worker import synchronize
from experiments.benchmarks.extraction.media import decode_canonical_audio
from experiments.benchmarks.extraction.scoring import bootstrap_sources, source_id
from experiments.benchmarks.extraction.video.protocol import protocol_from_worker

METRIC_DIRECTIONS = {
    name.replace("_warm_clip_latency_", "_warm_audio_processing_latency_"): direction
    for name, direction in AUDIO_DIRECTIONS.items()
    if name != "transcript_repeatability_success_rate"
}


def execute(payload):
    protocol = protocol_from_worker(payload["protocol"])
    audio_protocol = audio_protocol_from_worker(payload["audio_protocol"])
    preflight = payload.get("mode") == "preflight"
    execution = (
        protocol.preflight if preflight else protocol.profile(str(payload["profile"]))
    )
    device = str(payload["device"])
    if device not in (
        ("cuda",) if preflight else execution.devices or (execution.device,)
    ):
        raise ValueError("Frozen ASR device violates its execution profile")
    items = list(payload["items"])
    if not items:
        raise ValueError("Frozen ASR requires videos")
    runtime = build_runtime(
        str(payload["candidate"]), payload["model_lock"], device, audio_protocol
    )
    monitor = ResourceMonitor(require_vram=device == "cuda", device=device)
    videos, commands = [], []
    before = after = None
    with tempfile.TemporaryDirectory(prefix="edumind-video-asr-") as raw:
        temporary = Path(raw)
        try:
            with monitor:
                started = time.perf_counter()
                seed_deterministically(protocol.meta.seed)
                runtime.load()
                synchronize(device)
                cold = time.perf_counter() - started
                if device == "cuda":
                    before = inspect_model_placement(
                        *runtime.placement_models(), expected_device=device
                    )
                    if before["status"] != "qualified" and not preflight:
                        raise RuntimeError(
                            "Frozen ASR violates the no-offload contract"
                        )
                if execution.warmups:
                    warmup_audio = temporary / "warmup-full.wav"
                    commands.append(
                        {
                            "phase": "warmup",
                            "command": _decode(items[0], warmup_audio, audio_protocol),
                        }
                    )
                    warmup = _slice_wav(
                        warmup_audio,
                        temporary / "warmup.wav",
                        start=0,
                        duration=min(
                            protocol.window_length_seconds,
                            float(items[0]["duration_seconds"]),
                        ),
                        **_pcm_options(audio_protocol),
                    )
                    for _ in range(execution.warmups):
                        runtime.transcribe(warmup)
                        synchronize(device)
                for index, item in enumerate(items):
                    videos.append(
                        _transcribe_video(
                            runtime,
                            item,
                            temporary / str(index),
                            protocol,
                            audio_protocol,
                            device,
                            commands,
                        )
                    )
                if device == "cuda":
                    after = inspect_model_placement(
                        *runtime.placement_models(), expected_device=device
                    )
                    if after["status"] != "qualified" and not preflight:
                        raise RuntimeError(
                            "Frozen ASR weight placement changed during inference"
                        )
        finally:
            runtime.close()
    resources = monitor.metrics()
    if preflight:
        if any(not row["success"] for row in videos):
            raise RuntimeError(
                "Frozen-ASR stress inference failed; inspect window outcomes"
            )
        return {
            "placement": after if after["status"] != "qualified" else before,
            "placement_before": before,
            "placement_after": after,
            **resources,
            "vram_measurement_method": monitor.vram_measurement_method,
            "stress_sample_ids": [str(item["id"]) for item in items],
        }
    # Scoring is outside candidate monitoring.
    for item, row in zip(items, videos, strict=True):
        if row["success"]:
            row.update(
                score_speech(
                    {
                        **item,
                        "reference": item["reference_transcript"],
                        "reference_segments": item["reference_segments"],
                    },
                    row["transcript"],
                    row["segments"],
                    quality_latency_seconds=row["latency_seconds"],
                    alignment_threshold=audio_protocol.alignment_threshold,
                    timestamp_tolerance_seconds=audio_protocol.timestamp_tolerance_seconds,
                )
            )
        else:
            row.update(
                {
                    "sample_type": "speech",
                    "first_attempt_success": False,
                    "metric_status": "unavailable",
                    "reason": row["error"],
                }
            )
        row["source_group_id"] = source_id(item)
        row["nonspeech_units"] = _nonspeech_units(item, row, protocol)
        row["planned_reference_word_count"] = len(
            normalize_transcript(item["reference_transcript"]).split()
        )
        row["planned_reference_character_count"] = len(
            normalize_transcript(item["reference_transcript"])
        )
        row["planned_reference_timed_segment_count"] = sum(
            bool(normalize_transcript(segment["text"]))
            for segment in item["reference_segments"]
        )
        row["attempt_failure_rate"] = (
            sum(not window["success"] for window in row["windows"])
            / row["scheduled_window_count"]
            if len(row["windows"]) == row["scheduled_window_count"]
            else None
        )
        row["metric_values"], row["metric_statuses"] = sample_metrics(row)
        for field in ("metric_values", "metric_statuses"):
            row[field].pop("transcript_repeatability_success_rate")
        controls = [unit for unit in row["nonspeech_units"] if unit["success"]]
        row["metric_values"]["nonspeech_false_transcription_rate"] = (
            sum(unit["false_transcription"] for unit in controls) / len(controls)
            if controls
            else None
        )
        row["metric_statuses"]["nonspeech_false_transcription_rate"] = {
            "status": "scored"
            if controls
            else "unavailable"
            if row["nonspeech_units"]
            else "inapplicable",
            "reason": None
            if controls
            else "no_completed_reviewed_nonspeech_units"
            if row["nonspeech_units"]
            else "no_reviewed_nonspeech_units",
        }
    metrics = _statistics(videos)
    for name in videos[0]["metric_values"]:
        metrics[name + ".scheduled_count"] = float(len(videos))
        metrics[name + ".eligible_count"] = float(
            sum(
                row["metric_statuses"][name]["status"] != "inapplicable"
                for row in videos
            )
        )
        metrics[name + ".contributing_count"] = float(
            sum(row["metric_values"][name] is not None for row in videos)
        )
    intervals = bootstrap_sources(
        videos,
        _statistics,
        resamples=execution.bootstrap_resamples,
        seed=protocol.meta.seed,
        confidence=protocol.confidence_level,
        minimum_sources=protocol.minimum_ci_sources,
    )
    if protocol.minimum_latency_ci_sources != protocol.minimum_ci_sources:
        latency_intervals = bootstrap_sources(
            videos,
            _statistics,
            resamples=execution.bootstrap_resamples,
            seed=protocol.meta.seed,
            confidence=protocol.confidence_level,
            minimum_sources=protocol.minimum_latency_ci_sources,
        )
        for name in (
            "p50_warm_audio_processing_latency_seconds",
            "p95_warm_audio_processing_latency_seconds",
        ):
            if name in latency_intervals:
                intervals[name] = latency_intervals[name]
    metrics.update(
        {
            "cold_model_load_seconds": cold,
            **{
                name: resources[name]
                for name in ("peak_process_tree_ram_mb", "peak_vram_mb")
            },
        }
    )
    return {
        "videos": videos,
        "metrics": metrics,
        "intervals": intervals,
        "ffmpeg_commands": commands,
        "resource_samples": monitor.samples(),
        "gpu_identity": monitor.gpu_identity,
        "parameters": {
            **runtime.parameters(),
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": protocol.frozen_asr_repetitions,
            "stitching": dict(protocol.stitching),
            "window_length_seconds": protocol.window_length_seconds,
            "overlap_seconds": protocol.overlap_seconds,
            "vram_measurement_method": monitor.vram_measurement_method,
        },
    }


def _transcribe_video(
    runtime, item, directory, protocol, audio_protocol, device, commands
):
    directory.mkdir(parents=True)
    started = time.perf_counter()
    duration = float(item["duration_seconds"])
    segments, windows = [], []
    error = None
    try:
        decoded = directory / "full.wav"
        commands.append(
            {
                "phase": "decode",
                "sample_id": str(item["id"]),
                "command": _decode(item, decoded, audio_protocol),
            }
        )
        for index, start in enumerate(
            _window_starts(
                duration, protocol.window_length_seconds, protocol.overlap_seconds
            )
        ):
            window_duration = min(protocol.window_length_seconds, duration - start)
            window_started = time.perf_counter()
            output = None
            failure = None
            try:
                window = _slice_wav(
                    decoded,
                    directory / f"{index}.wav",
                    start=start,
                    duration=window_duration,
                    **_pcm_options(audio_protocol),
                )
                seed_deterministically(protocol.meta.seed)
                output = runtime.transcribe(window)
                predictions = validate_prediction(
                    output.text,
                    output.segments,
                    window_duration,
                    audio_protocol.timestamp_tolerance_seconds,
                )
                synchronize(device)
                shifted = [
                    {
                        "text": segment["text"],
                        "start": float(segment["start"]) + start,
                        "end": min(duration, float(segment["end"]) + start),
                    }
                    for segment in predictions
                ]
                if index == 0 or protocol.overlap_seconds:
                    segments = _stitch_segments(
                        segments,
                        shifted,
                        int(protocol.stitching["maximum_overlap_tokens"]),
                        overlap_start=start,
                        overlap_end=start + protocol.overlap_seconds,
                    )
                else:
                    segments.extend(_timestamp_units(shifted))
            except Exception as exc:  # noqa: BLE001 - preserve recoverable window outcomes
                failure = f"{type(exc).__name__}: {exc}"
            windows.append(
                {
                    "index": index,
                    "start": start,
                    "end": start + window_duration,
                    "success": failure is None,
                    "error": failure,
                    "latency_seconds": time.perf_counter() - window_started,
                    "transcript": output.text if output is not None else None,
                    "segments": list(output.segments) if output is not None else None,
                }
            )
        transcript = " ".join(str(segment["text"]) for segment in segments)
        if any(not row["success"] for row in windows):
            error = "required_window_failed"
        else:
            validate_prediction(
                transcript,
                segments,
                duration,
                audio_protocol.timestamp_tolerance_seconds,
            )
            synchronize(device)
    except Exception as exc:  # noqa: BLE001 - setup failures are not invented window failures
        error = f"{type(exc).__name__}: {exc}"
    return {
        "sample_id": str(item["id"]),
        "duration_seconds": duration,
        "success": error is None,
        "first_attempt_success": error is None,
        "error": error,
        "windows": windows,
        "scheduled_window_count": len(
            _window_starts(
                duration, protocol.window_length_seconds, protocol.overlap_seconds
            )
        ),
        "transcript": " ".join(str(segment["text"]) for segment in segments),
        "segments": segments,
        "latency_seconds": time.perf_counter() - started,
    }


def _statistics(rows):
    timings = [
        {
            "sample_id": str(index),
            "sample_type": "speech",
            "success": row["success"],
            "duration_seconds": row["duration_seconds"],
            "latency_seconds": row["latency_seconds"],
        }
        for index, row in enumerate(rows)
    ]
    result = aggregate_rows(rows, timings)
    result.pop("transcript_repeatability_success_rate")
    for percentile in (50, 95):
        result.pop(f"p{percentile}_warm_clip_latency_seconds")
    windows = [window for row in rows for window in row["windows"]]
    scheduled = sum(row["scheduled_window_count"] for row in rows)
    result["attempt_failure_rate"] = (
        sum(not row["success"] for row in windows) / scheduled
        if scheduled and len(windows) == scheduled
        else None
    )
    controls = [
        unit for row in rows for unit in row["nonspeech_units"] if unit["success"]
    ]
    result["nonspeech_false_transcription_rate"] = (
        sum(unit["false_transcription"] for unit in controls) / len(controls)
        if controls
        else None
    )
    completed = [float(row["latency_seconds"]) for row in rows if row["success"]]
    for percentile in (50, 95):
        result[f"p{percentile}_warm_audio_processing_latency_seconds"] = (
            float(np.quantile(completed, percentile / 100)) if completed else None
        )
    return result


def _nonspeech_units(item, row, protocol):
    if (
        protocol.nonspeech_unit_type == "whole_video"
        and item.get("nonspeech_reviewed") is True
    ):
        return [
            {
                "success": row["success"],
                "false_transcription": bool(normalize_transcript(row["transcript"])),
            }
        ]
    if protocol.nonspeech_unit_type == "asr_window":
        reviewed = set(item.get("nonspeech_window_indices", []))
        return [
            {
                "index": window["index"],
                "success": window["success"],
                "false_transcription": bool(
                    normalize_transcript(window["transcript"] or "")
                ),
            }
            for window in row["windows"]
            if window["index"] in reviewed
        ]
    return []


def _decode(item, destination, protocol):
    return decode_canonical_audio(
        Path(str(item["source_path"])),
        destination,
        sample_rate_hz=int(protocol.audio["sample_rate_hz"]),
        channels=int(protocol.audio["channels"]),
    )


def _pcm_options(protocol):
    return {
        name: int(protocol.audio[name])
        for name in ("sample_rate_hz", "channels", "sample_width_bytes")
    }


def _window_starts(duration, length, overlap):
    if (
        not all(math.isfinite(value) for value in (duration, length, overlap))
        or duration <= 0
        or not 0 <= overlap < length
    ):
        raise ValueError("Video duration and ASR window bounds are invalid")
    starts, current = [], 0.0
    while current < duration:
        starts.append(current)
        if current + length >= duration:
            break
        current += length - overlap
    return starts


def _slice_wav(
    source,
    destination,
    *,
    start,
    duration,
    sample_rate_hz,
    channels,
    sample_width_bytes,
):
    with wave.open(str(source), "rb") as reader:
        if (reader.getnchannels(), reader.getframerate(), reader.getsampwidth()) != (
            channels,
            sample_rate_hz,
            sample_width_bytes,
        ):
            raise ValueError(
                "Decoded video audio does not match canonical PCM settings"
            )
        reader.setpos(min(reader.getnframes(), round(start * sample_rate_hz)))
        frames = reader.readframes(round(duration * sample_rate_hz))
        parameters = reader.getparams()
    if not frames:
        raise ValueError("Required audio window has no PCM frames")
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(frames)
    return destination


def _stitch_segments(
    existing, current, maximum_overlap, *, overlap_start=None, overlap_end=None
):
    left, right = _timestamp_units(existing), _timestamp_units(current)
    limit = min(len(left), len(right), maximum_overlap)
    if overlap_start is not None:
        limit = min(
            limit,
            sum(unit["end"] > overlap_start for unit in left),
            sum(unit["start"] < overlap_end for unit in right),
        )
    width = next(
        (
            width
            for width in range(limit, 0, -1)
            if [unit["text"] for unit in left[-width:]]
            == [unit["text"] for unit in right[:width]]
        ),
        0,
    )
    # Overlapping windows can disagree lexically. Retain both observations but
    # keep native timestamps in chronological order instead of inventing times.
    return sorted([*left, *right[width:]], key=lambda unit: unit["start"])


def _timestamp_units(segments):
    return [
        {"text": token, "start": float(item["start"]), "end": float(item["end"])}
        for item in segments
        for token in normalize_transcript(str(item["text"])).split()
    ]


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
