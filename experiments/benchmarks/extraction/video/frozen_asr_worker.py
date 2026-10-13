"""One measured audio preparation per video, independent of visual candidates."""

from __future__ import annotations

import math
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from edumind.common.artifacts import sha256_file
from edumind.common.model_placement import inspect_model_placement
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
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
from experiments.benchmarks.extraction.media import (
    canonical_wav_duration,
    decode_canonical_audio,
)
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
    if payload.get("mode") not in {None, "preflight"}:
        raise ValueError("Unknown frozen ASR worker mode")
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
    if len({str(item["id"]) for item in items}) != len(items):
        raise ValueError("Frozen ASR requires unique video IDs")
    runtime = None
    monitor = ResourceMonitor(
        interval_seconds=protocol.preflight.telemetry_interval_seconds,
        require_vram=device == "cuda",
        device=device,
    )
    videos, commands = [], []
    before = after = None
    cold = execution_error = None
    parameters = {}
    phase = "load"
    lifecycle_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="edumind-video-asr-") as raw:
        temporary = Path(raw)
        try:
            with monitor:
                started = time.perf_counter()
                seed_deterministically(protocol.meta.seed)
                runtime = build_runtime(
                    str(payload["candidate"]),
                    payload["model_lock"],
                    device,
                    audio_protocol,
                )
                runtime.load()
                synchronize(device)
                cold = time.perf_counter() - started
                phase = "placement_before"
                if device == "cuda":
                    before = inspect_model_placement(
                        *runtime.placement_models(), expected_device=device
                    )
                    if before["status"] != "qualified":
                        raise RuntimeError(
                            "Frozen ASR violates the no-offload contract"
                        )
                phase = "warmup"
                if execution.warmups:
                    warmup_audio = temporary / "warmup-full.wav"
                    commands.append(
                        {
                            "phase": "warmup",
                            "command": _decode(items[0], warmup_audio, audio_protocol),
                        }
                    )
                    warmup_duration = min(
                        protocol.window_length_seconds,
                        float(items[0]["duration_seconds"]),
                        canonical_wav_duration(
                            warmup_audio, **_pcm_options(audio_protocol)
                        ),
                    )
                    warmup = _slice_wav(
                        warmup_audio,
                        temporary / "warmup.wav",
                        start=0,
                        duration=warmup_duration,
                        **_pcm_options(audio_protocol),
                    )
                    for _ in range(execution.warmups):
                        seed_deterministically(protocol.meta.seed)
                        output = runtime.transcribe(warmup)
                        validate_prediction(
                            output.text,
                            output.segments,
                            warmup_duration,
                            audio_protocol.timestamp_tolerance_seconds,
                        )
                        synchronize(device)
                phase = "inference"
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
                    phase = "placement_after"
                    after = inspect_model_placement(
                        *runtime.placement_models(), expected_device=device
                    )
                    if after["status"] != "qualified":
                        raise RuntimeError(
                            "Frozen ASR weight placement changed during inference"
                        )
                phase = "telemetry"
        except Exception as exc:  # noqa: BLE001 - preserve lifecycle observations
            before = getattr(exc, "placement_report", before)
            execution_error = {
                "phase": phase,
                "error": f"{type(exc).__name__}: {exc}",
                "lifecycle_elapsed_seconds": time.perf_counter() - lifecycle_started,
                "resource_evidence": getattr(exc, "resource_evidence", {}),
            }
            if getattr(exc, "cmd", None) is not None:
                commands.append(
                    {
                        "phase": phase,
                        "command": list(exc.cmd),
                        "error": execution_error["error"],
                    }
                )
        finally:
            for name, operation in (
                (("parameters", runtime.parameters), ("close", runtime.close))
                if runtime is not None
                else ()
            ):
                try:
                    value = operation()
                    if name == "parameters":
                        parameters = value
                except Exception as exc:  # noqa: BLE001 - cleanup cannot erase prior evidence
                    error = f"{type(exc).__name__}: {exc}"
                    if execution_error is None:
                        execution_error = {"phase": name, "error": error}
                    else:
                        execution_error[name + "_error"] = error
    try:
        resources = monitor.metrics()
    except Exception as exc:  # noqa: BLE001 - retain partial telemetry
        evidence = getattr(exc, "resource_evidence", {})
        resources = {
            name: evidence.get(name)
            for name in ("peak_process_tree_ram_mb", "peak_vram_mb")
        }
        execution_error = execution_error or {
            "phase": "telemetry",
            "error": f"{type(exc).__name__}: {exc}",
        }
        execution_error["resource_evidence"] = evidence
    if device == "cpu":
        resources["peak_vram_mb"] = None
    elif (
        resources["peak_vram_mb"] is None
        or monitor.vram_measurement_method != "nvml-device-total"
    ):
        execution_error = execution_error or {
            "phase": "telemetry",
            "error": "Frozen ASR CUDA telemetry is unavailable",
        }
    observed = {row["sample_id"] for row in videos}
    videos.extend(
        _unexecuted_video(
            item,
            protocol,
            execution_error["phase"] if execution_error else "missing_video",
        )
        for item in items
        if str(item["id"]) not in observed
    )
    evidence = {
        "videos": videos,
        **resources,
        "cold_model_load_seconds": cold,
        "placement_before": before,
        "placement_after": after,
        "execution_error": execution_error,
        "ffmpeg_commands": commands,
        "resource_samples": monitor.samples(),
        "gpu_identity": monitor.gpu_identity,
        "vram_measurement_method": monitor.vram_measurement_method,
        "parameters": parameters,
        "resource_measurement_scope": "partial_lifecycle"
        if execution_error
        else "full_lifecycle",
    }
    if preflight:
        evidence.update(
            {
                "placement": after
                if after and after["status"] != "qualified"
                else before,
                "stress_sample_ids": [str(item["id"]) for item in items],
            }
        )
        if execution_error or any(not row["success"] for row in videos):
            error_type = (
                WorkerMeasurementError
                if execution_error and execution_error["phase"] == "telemetry"
                else WorkerExecutionError
            )
            reason = (
                execution_error["error"]
                if execution_error
                else next(
                    (
                        window["error"]
                        for row in videos
                        for window in row["windows"]
                        if window["error"]
                    ),
                    next(
                        (row["error"] for row in videos if row["error"]),
                        "required stress work was not completed",
                    ),
                )
            )
            raise error_type(
                f"Frozen-ASR stress lifecycle/inference failed: {reason}", evidence
            )
        return evidence
    # Scoring is outside candidate monitoring.
    try:
        return _score_videos(
            items, videos, protocol, audio_protocol, execution, evidence
        )
    except Exception as exc:
        raise WorkerExecutionError(
            f"Frozen ASR offline evaluation failed: {type(exc).__name__}: {exc}",
            {
                **evidence,
                "execution_error": {
                    "phase": "offline_evaluation",
                    "error": f"{type(exc).__name__}: {exc}",
                    "preceding_execution_error": execution_error,
                },
            },
        ) from exc


def _score_videos(items, videos, protocol, audio_protocol, execution, evidence):
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
                    warnings=[
                        warning
                        for window in row["windows"]
                        for warning in window["warnings"]
                    ],
                    alignment_threshold=audio_protocol.alignment_threshold,
                    timestamp_tolerance_seconds=audio_protocol.timestamp_tolerance_seconds,
                    allow_untimed_reference=True,
                )
            )
        else:
            row.update(
                {
                    "sample_type": "speech",
                    "first_attempt_success": False if row["executed"] else None,
                    "first_attempt_executed": row["executed"],
                    "metric_status": "unavailable" if row["executed"] else "incomplete",
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
            sum(window["success"] is False for window in row["windows"])
            / row["scheduled_window_count"]
            if _windows_complete(row)
            else None
        )
        row["unexecuted_attempt_count"] = row["unexecuted_window_count"]
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
            "cold_model_load_seconds": evidence["cold_model_load_seconds"],
            **{
                name: evidence[name]
                for name in ("peak_process_tree_ram_mb", "peak_vram_mb")
            },
        }
    )
    statuses = _metric_records(metrics, videos, evidence)
    for name, record in statuses.items():
        if name in {
            "cold_model_load_seconds",
            "peak_process_tree_ram_mb",
            "peak_vram_mb",
        }:
            continue
        for field, value in record["support"].items():
            metrics[f"{name}.{field}"] = float(value)
    return {
        **evidence,
        "videos": videos,
        "metrics": metrics,
        "metric_statuses": statuses,
        "intervals": intervals,
        "parameters": {
            **evidence["parameters"],
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": protocol.frozen_asr_repetitions,
            "stitching": dict(protocol.stitching),
            "window_length_seconds": protocol.window_length_seconds,
            "overlap_seconds": protocol.overlap_seconds,
            "vram_measurement_method": evidence["vram_measurement_method"],
        },
    }


def _metric_records(metrics, videos, evidence):
    """Describe the actual cohort/unit of each independently available point."""
    windows = [window for row in videos for window in row["windows"]]
    incomplete = bool(evidence["execution_error"]) or any(
        not _windows_complete(row) for row in videos
    )
    records = {}
    lifecycle = {"cold_model_load_seconds", "peak_process_tree_ram_mb", "peak_vram_mb"}
    speed = {
        "real_time_factor",
        "p50_warm_audio_processing_latency_seconds",
        "p95_warm_audio_processing_latency_seconds",
    }
    for name in METRIC_DIRECTIONS:
        value = metrics[name]
        if name in lifecycle:
            applicable = not (
                name == "peak_vram_mb"
                and evidence["vram_measurement_method"] == "not-applicable"
            )
            support = {
                "scheduled_count": 1,
                "eligible_count": int(applicable),
                "contributing_count": int(value is not None),
                "sample_count": int(value is not None),
                "independent_source_count": None,
                "contributing_independent_source_count": None,
                "observation_unit": "worker_lifecycle",
            }
            status = (
                "scored"
                if value is not None
                else "unavailable"
                if applicable
                else "inapplicable"
            )
            reason = (
                None
                if value is not None
                else "required_measurement_unavailable"
                if applicable
                else "cpu_profile"
            )
        else:
            statuses = [row["metric_statuses"].get(name, {}) for row in videos]
            if name in speed:
                units = [(row, row["executed"], row["success"]) for row in videos]
            elif name == "attempt_failure_rate":
                units = [
                    (row, window["executed"], window["success"])
                    for row in videos
                    for window in row["windows"]
                ]
            elif name == "nonspeech_false_transcription_rate":
                units = [
                    (row, unit["executed"], unit["success"])
                    for row in videos
                    for unit in row["nonspeech_units"]
                ]
            else:
                units = [
                    (row, row["executed"], row["success"])
                    for row, state in zip(videos, statuses, strict=True)
                    if state.get("status") != "inapplicable"
                ]
            # Failure-rate observations include recorded failed requests; other
            # conditional points use only completed eligible results.
            contributors = [
                row
                for row, executed, success in units
                if executed
                and (
                    name == "attempt_failure_rate"
                    or (
                        success
                        and (
                            name in speed
                            or name == "nonspeech_false_transcription_rate"
                            or row["metric_values"].get(name) is not None
                        )
                    )
                )
            ]
            contributing_sources = {row["source_group_id"] for row in contributors}
            support = {
                "scheduled_count": len(units)
                if name
                in {"attempt_failure_rate", "nonspeech_false_transcription_rate"}
                else len(videos),
                "eligible_count": len(units),
                "contributing_count": len(contributors),
                "failed_count": sum(
                    executed and not success for _, executed, success in units
                ),
                "unavailable_count": 0
                if name == "attempt_failure_rate"
                else sum(
                    executed
                    and (
                        not success
                        or (
                            name not in speed
                            and name != "nonspeech_false_transcription_rate"
                            and row["metric_values"].get(name) is None
                        )
                    )
                    for row, executed, success in units
                ),
                "incomplete_count": sum(not executed for _, executed, _ in units),
                "independent_source_count": len(
                    {row["source_group_id"] for row in videos}
                ),
                "contributing_independent_source_count": len(contributing_sources),
                "sample_count": len(contributing_sources) if value is not None else 0,
                "scheduled_window_count": len(windows),
                "executed_window_count": sum(window["executed"] for window in windows),
                "failed_window_count": sum(
                    window["success"] is False for window in windows
                ),
                "unexecuted_window_count": sum(
                    not window["executed"] for window in windows
                ),
            }
            for field in (
                "reference_word_count",
                "reference_character_count",
                "reference_timed_segment_count",
            ):
                support["planned_" + field] = sum(
                    row.get("planned_" + field, 0) for row in videos
                )
                support["contributing_" + field] = sum(
                    row.get(field, 0) for row in videos if row["success"]
                )
            if value is not None:
                status, reason = "scored", None
            elif not units:
                status, reason = "inapplicable", "no_reference_task"
            elif support["incomplete_count"] or any(
                state.get("status") == "incomplete" for state in statuses
            ):
                status, reason = "incomplete", "required_work_not_executed"
            elif any(
                state.get("reason") == "no_aligned_boundaries" for state in statuses
            ):
                status, reason = "unavailable", "no_aligned_boundaries"
            else:
                status, reason = "unavailable", "no_completed_eligible_outputs"
        records[name] = {
            "value": value,
            "status": status,
            "reason": reason,
            "support": support,
            "comparison_status": "incomplete" if incomplete else "complete",
            "execution_error_phase": evidence["execution_error"]["phase"]
            if evidence["execution_error"]
            else None,
        }
    return records


def _transcribe_video(
    runtime, item, directory, protocol, audio_protocol, device, commands
):
    directory.mkdir(parents=True)
    started = time.perf_counter()
    duration = float(item["duration_seconds"])
    segments, windows = [], []
    error = None
    canonical_checksum = canonical_duration = None
    planned = _window_records(item, protocol)
    try:
        decoded = directory / "full.wav"
        commands.append(
            {
                "phase": "decode",
                "sample_id": str(item["id"]),
                "command": _decode(item, decoded, audio_protocol),
            }
        )
        canonical_duration = canonical_wav_duration(
            decoded, **_pcm_options(audio_protocol)
        )
        canonical_checksum = sha256_file(decoded)
        if (
            abs(canonical_duration - duration)
            > protocol.manifest_duration_tolerance_seconds
        ):
            raise ValueError(
                "Decoded video audio duration differs from validated inputs"
            )
        if (
            item.get("canonical_sha256") is not None
            and canonical_checksum != item["canonical_sha256"]
        ):
            raise ValueError(
                "Decoded video audio checksum differs from sealed validation"
            )
        for record in planned:
            index, start = record["index"], record["start"]
            window_duration = record["end"] - start
            input_duration = min(window_duration, canonical_duration - start)
            window_started = time.perf_counter()
            output = None
            failure = None
            executed = False
            try:
                window = _slice_wav(
                    decoded,
                    directory / f"{index}.wav",
                    start=start,
                    duration=input_duration,
                    **_pcm_options(audio_protocol),
                )
                seed_deterministically(protocol.meta.seed)
                executed = True
                output = runtime.transcribe(window)
                predictions = validate_prediction(
                    output.text,
                    output.segments,
                    input_duration,
                    audio_protocol.timestamp_tolerance_seconds,
                )
                synchronize(device)
                shifted = [
                    {
                        "text": segment["text"],
                        "start": float(segment["start"]) + start,
                        "end": float(segment["end"]) + start,
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
                    "input_duration_seconds": input_duration,
                    "executed": executed,
                    "success": failure is None if executed else None,
                    "error": failure,
                    "not_executed_reason": failure if not executed else None,
                    "latency_seconds": time.perf_counter() - window_started
                    if executed
                    else None,
                    "preprocessing_elapsed_seconds": time.perf_counter()
                    - window_started
                    if not executed
                    else None,
                    "transcript": output.text if output is not None else None,
                    "segments": list(output.segments) if output is not None else None,
                    "warnings": list(output.warnings) if output is not None else [],
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
        command = getattr(exc, "cmd", None)
        if command is not None:
            commands.append(
                {
                    "phase": "decode",
                    "sample_id": str(item["id"]),
                    "command": list(command),
                    "error": error,
                }
            )
    windows.extend(
        {**record, "not_executed_reason": error or "missing_window"}
        for record in planned[len(windows) :]
    )
    return {
        "sample_id": str(item["id"]),
        "duration_seconds": duration,
        "success": error is None,
        "executed": True,
        "first_attempt_executed": True,
        "first_attempt_success": error is None,
        "error": error,
        "windows": windows,
        "scheduled_window_count": len(planned),
        "executed_window_count": sum(window["executed"] for window in windows),
        "failed_window_count": sum(window["success"] is False for window in windows),
        "unexecuted_window_count": sum(not window["executed"] for window in windows),
        "transcript": " ".join(str(segment["text"]) for segment in segments),
        "segments": segments,
        "latency_seconds": time.perf_counter() - started,
        "canonical_sha256": canonical_checksum,
        "canonical_duration_seconds": canonical_duration,
    }


def _window_records(item, protocol):
    duration = float(item["duration_seconds"])
    return [
        {
            "index": index,
            "start": start,
            "end": min(duration, start + protocol.window_length_seconds),
            "executed": False,
            "success": None,
            "error": None,
            "latency_seconds": None,
            "transcript": None,
            "segments": None,
            "warnings": [],
            "not_executed_reason": "not_executed",
        }
        for index, start in enumerate(
            _window_starts(
                duration, protocol.window_length_seconds, protocol.overlap_seconds
            )
        )
    ]


def _unexecuted_video(item, protocol, reason):
    windows = _window_records(item, protocol)
    return {
        "sample_id": str(item["id"]),
        "duration_seconds": float(item["duration_seconds"]),
        "executed": False,
        "success": False,
        "first_attempt_success": None,
        "first_attempt_executed": False,
        "error": reason,
        "windows": [{**window, "not_executed_reason": reason} for window in windows],
        "scheduled_window_count": len(windows),
        "executed_window_count": 0,
        "failed_window_count": 0,
        "unexecuted_window_count": len(windows),
        "transcript": "",
        "segments": [],
        "latency_seconds": None,
        "canonical_sha256": None,
        "canonical_duration_seconds": None,
    }


def _windows_complete(row):
    return len(row["windows"]) == row["scheduled_window_count"] and all(
        window.get("executed") is True and window.get("success") is not None
        for window in row["windows"]
    )


def _statistics(rows):
    timings = [
        {
            "sample_id": str(index),
            "sample_type": "speech",
            "success": row["success"],
            "executed": row.get("executed", True),
            "duration_seconds": row["duration_seconds"],
            "latency_seconds": row["latency_seconds"],
        }
        for index, row in enumerate(rows)
    ]
    result = aggregate_rows(
        [{**row, "sample_id": str(index)} for index, row in enumerate(rows)], timings
    )
    result.pop("transcript_repeatability_success_rate")
    for percentile in (50, 95):
        result.pop(f"p{percentile}_warm_clip_latency_seconds")
    windows = [window for row in rows for window in row["windows"]]
    scheduled = sum(row["scheduled_window_count"] for row in rows)
    result["attempt_failure_rate"] = (
        sum(window["success"] is False for window in windows) / scheduled
        if scheduled and all(_windows_complete(row) for row in rows)
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
                "unit_type": "whole_video",
                "sample_id": str(item["id"]),
                "reviewed": True,
                "labels": list(item.get("conditions", [])),
                "nonspeech_kind": item.get("nonspeech_kind"),
                "executed": row["executed"],
                "success": row["success"] if row["executed"] else None,
                "false_transcription": bool(normalize_transcript(row["transcript"]))
                if row["success"]
                else None,
            }
        ]
    if protocol.nonspeech_unit_type == "asr_window":
        reviewed = set(item.get("nonspeech_window_indices", []))
        return [
            {
                "unit_type": "asr_window",
                "sample_id": str(item["id"]),
                "reviewed": True,
                "labels": list(item.get("conditions", [])),
                "nonspeech_kind": item.get("nonspeech_kind"),
                "index": window["index"],
                "start": window["start"],
                "end": window["end"],
                "executed": window.get("executed", True),
                "success": window["success"],
                "false_transcription": bool(
                    normalize_transcript(window["transcript"] or "")
                )
                if window["success"]
                else None,
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
    if len(frames) != round(duration * sample_rate_hz) * channels * sample_width_bytes:
        raise ValueError("Required audio window has missing or truncated PCM frames")
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(frames)
    return destination


def _stitch_segments(
    existing, current, maximum_overlap, *, overlap_start=None, overlap_end=None
):
    left, right = _timestamp_units(existing), _timestamp_units(current)
    limit = min(len(left), len(right), maximum_overlap)
    width = next(
        (
            width
            for width in range(limit, 0, -1)
            if [unit["text"] for unit in left[-width:]]
            == [unit["text"] for unit in right[:width]]
            and (
                overlap_start is None
                or all(
                    unit["end"] > overlap_start and unit["start"] < overlap_end
                    for unit in [*left[-width:], *right[:width]]
                )
            )
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
