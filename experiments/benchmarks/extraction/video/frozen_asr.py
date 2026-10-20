"""Versioned frozen-ASR artifacts: exact inputs, model identity, and complete outcomes."""

from __future__ import annotations

import json
import math
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.process import WorkerExecutionError, run_json_worker
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.extraction.audio.adapters import profiles
from experiments.benchmarks.extraction.audio.evaluate import (
    normalize_transcript,
    validate_prediction,
)
from experiments.benchmarks.extraction.audio.protocol import (
    protocol_from_worker as audio_protocol_from_worker,
)
from experiments.benchmarks.extraction.video.frozen_asr_worker import (
    METRIC_DIRECTIONS,
    _stitch_segments,
    _window_starts,
)
from experiments.benchmarks.extraction.video.protocol import protocol_from_worker
from experiments.benchmarks.preparation.models import load_selected_model_lock


def execution_identity():
    """Bind reusable audio to its executing/scoring code and installed dependencies."""
    extraction = PROJECT_ROOT / "experiments/benchmarks/extraction"
    files = [
        *sorted((extraction / "audio").glob("*.py")),
        *sorted((extraction / "video").glob("frozen_asr*.py")),
        extraction / "video/protocol.py",
        extraction / "video/validation.py",
        extraction / "scoring.py",
        extraction / "media.py",
        *(
            PROJECT_ROOT / "experiments/benchmarks/common" / name
            for name in (
                "metrics.py",
                "process.py",
                "resources.py",
                "protocol.py",
                "provenance.py",
            )
        ),
        PROJECT_ROOT / "src/edumind/common/artifacts.py",
        PROJECT_ROOT / "src/edumind/common/gpu.py",
        PROJECT_ROOT / "src/edumind/common/model_placement.py",
        PROJECT_ROOT / "src/edumind/extraction/extractors/audio.py",
        PROJECT_ROOT / "experiments/benchmarks/preparation/models.py",
        PROJECT_ROOT / "pyproject.toml",
        *sorted((PROJECT_ROOT / "requirements").glob("*.lock")),
    ]
    return {
        "python_version": sys.version,
        "implementation_and_locks": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in files
        },
        "package_versions": package_versions(
            (
                "torch",
                "torchaudio",
                "transformers",
                "nemo_toolkit",
                "moss-transcribe-diarize",
                "soundfile",
                "jiwer",
                "rapidfuzz",
                "numpy",
            )
        ),
    }


def model_identity(candidate, decision_path, audio_protocol):
    declared = profiles(audio_protocol)
    if candidate not in declared:
        raise ValueError(f"Unknown selected ASR profile: {candidate}")
    lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(declared[candidate].model,),
    )
    entry = lock[declared[candidate].model]
    return lock, {
        "audio_candidate": candidate,
        "model_path": entry["model_path"],
        "model_revision": entry["revision"],
        "selection_revision": entry["selection_revision"],
        "model_cache_manifest_sha256": entry["model_cache_manifest_sha256"],
        "submodels": entry.get("submodels", []),
        "model_lock_fingerprint": stable_hash(lock),
        "model_decision_fingerprint": stable_hash(
            {"candidate": candidate, "sha256": sha256_file(decision_path)}
        ),
    }


def create_frozen_asr_artifact(
    output_path,
    *,
    items,
    manifest_checksum,
    protocol,
    audio_protocol,
    audio_candidate,
    audio_decision_path,
    device,
    ffmpeg_version,
    profile_name,
    producer_run_id=None,
):
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Frozen ASR artifacts are immutable; choose a new output path")
    lock, identity = model_identity(
        audio_candidate, audio_decision_path, audio_protocol
    )
    executing = execution_identity()
    try:
        result = run_json_worker(
            Path(__file__).with_name("frozen_asr_worker.py"),
            {
                "candidate": audio_candidate,
                "model_lock": lock,
                "items": list(items),
                "device": device,
                "profile": profile_name,
                "protocol": protocol.meta.worker_payload(),
                "audio_protocol": audio_protocol.meta.worker_payload(),
            },
            device=device,
            prefix="edumind-frozen-asr-",
            error_label="Frozen video ASR worker",
            require_vram_measurement=device == "cuda",
            telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
            poll_interval_seconds=protocol.preflight.poll_interval_seconds,
            timeout_seconds=protocol.preflight.worker_timeout_seconds,
        )
    except WorkerExecutionError as exc:
        # Keep the transport's raw worker_result, windows and resource scopes.
        evidence = {
            **exc.resource_evidence,
            "frozen_asr_identity": identity,
            "manifest_checksum": manifest_checksum,
            "protocol_checksum": protocol.meta.checksum,
            "audio_protocol_checksum": audio_protocol.meta.checksum,
        }
        raise type(exc)(str(exc), evidence) from exc
    if execution_identity() != executing:
        raise WorkerExecutionError(
            "Frozen ASR code/software changed during preparation", result
        )
    if any(
        name in result
        for name in (
            "schema_version",
            "artifact_type",
            "run_id",
            "manifest_checksum",
            "protocol_checksum",
            "audio_protocol_checksum",
            "protocol",
            "audio_protocol",
            "device",
            "profile",
            "execution_identity",
            *identity,
        )
    ):
        raise WorkerExecutionError(
            "Frozen ASR worker returned conflicting provenance", result
        )
    artifact = {
        "schema_version": 2,
        "artifact_type": "FrozenASRArtifact",
        "run_id": producer_run_id or "local-" + uuid.uuid4().hex,
        "manifest_checksum": manifest_checksum,
        "protocol_checksum": protocol.meta.checksum,
        "protocol_version": protocol.meta.version,
        "audio_protocol_checksum": audio_protocol.meta.checksum,
        "audio_protocol_version": audio_protocol.meta.version,
        "device": device,
        "profile": profile_name,
        **identity,
        "protocol": protocol.meta.worker_payload(),
        "audio_protocol": audio_protocol.meta.worker_payload(),
        "ffmpeg_version": ffmpeg_version,
        "execution_identity": executing,
        **result,
    }
    artifact["artifact_content_sha256"] = stable_hash(artifact)
    atomic_write_json(output_path, artifact)
    return output_path


def load_frozen_asr_artifact(
    path,
    *,
    manifest_checksum,
    protocol_checksum,
    audio_protocol_checksum,
    timestamp_tolerance_seconds,
    sample_ids,
    expected_identity=None,
    expected_device=None,
    expected_profile=None,
    expected_checksum=None,
    expected_durations=None,
    expected_ffmpeg_version=None,
    expected_canonical_checksums=None,
):
    path = Path(path)
    checksum = sha256_file(path)
    if expected_checksum is not None and checksum != expected_checksum:
        raise ValueError("Frozen ASR artifact content checksum mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 2
        or payload.get("artifact_type") != "FrozenASRArtifact"
    ):
        raise ValueError(
            "Frozen ASR requires schema_version 2; regenerate historical artifacts"
        )
    if payload.get("artifact_content_sha256") != stable_hash(
        {
            name: value
            for name, value in payload.items()
            if name != "artifact_content_sha256"
        }
    ):
        raise ValueError("Frozen ASR artifact content checksum mismatch")
    for name, expected in {
        "manifest_checksum": manifest_checksum,
        "protocol_checksum": protocol_checksum,
        "audio_protocol_checksum": audio_protocol_checksum,
        **dict(expected_identity or {}),
    }.items():
        if payload.get(name) != expected:
            raise ValueError(f"Frozen ASR artifact {name.replace('_', ' ')} mismatch")
    for name, expected in (("device", expected_device), ("profile", expected_profile)):
        if expected is not None and payload.get(name) != expected:
            raise ValueError(f"Frozen ASR artifact {name} mismatch")
    video_protocol = protocol_from_worker(payload.get("protocol"))
    audio_protocol = audio_protocol_from_worker(payload.get("audio_protocol"))
    if (
        video_protocol.meta.checksum,
        audio_protocol.meta.checksum,
        video_protocol.meta.version,
        audio_protocol.meta.version,
    ) != (
        protocol_checksum,
        audio_protocol_checksum,
        payload.get("protocol_version"),
        payload.get("audio_protocol_version"),
    ):
        raise ValueError("Frozen ASR embedded protocol identity mismatch")
    if (
        expected_ffmpeg_version is not None
        and payload.get("ffmpeg_version") != expected_ffmpeg_version
    ):
        raise ValueError("Frozen ASR FFmpeg version mismatch")
    required = {
        "run_id",
        "model_decision_fingerprint",
        "ffmpeg_version",
        "ffmpeg_commands",
        "metrics",
        "intervals",
        "parameters",
        "audio_candidate",
        "model_path",
        "model_revision",
        "selection_revision",
        "model_cache_manifest_sha256",
        "resource_samples",
        "gpu_identity",
        "artifact_content_sha256",
        "execution_error",
        "placement_before",
        "placement_after",
        "metric_statuses",
        "resource_measurement_scope",
        "submodels",
        "model_lock_fingerprint",
    }
    if required - payload.keys():
        raise ValueError(
            "Frozen ASR artifact lacks required provenance or measurements"
        )
    for name in (
        "run_id",
        "model_decision_fingerprint",
        "ffmpeg_version",
        "audio_candidate",
        "model_path",
        "model_revision",
        "selection_revision",
        "model_cache_manifest_sha256",
    ):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"Frozen ASR artifact has invalid {name} provenance")
    if payload.get("execution_identity") != execution_identity():
        raise ValueError(
            "Frozen ASR execution/software identity mismatch; regenerate the artifact"
        )
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping) or set(METRIC_DIRECTIONS) - metrics.keys():
        raise ValueError("Frozen ASR artifact lacks required aggregate metrics")
    if any(
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        )
        for value in metrics.values()
    ):
        raise ValueError("Frozen ASR artifact contains non-finite metrics")
    videos = payload.get("videos")
    if not isinstance(videos, list) or any(
        not isinstance(row, Mapping) for row in videos
    ):
        raise ValueError("Frozen ASR artifact lacks per-video outcomes")
    observed = [row.get("sample_id") for row in videos]
    if (
        any(not isinstance(value, str) or not value for value in observed)
        or set(observed) != set(sample_ids)
        or len(observed) != len(sample_ids)
        or len(set(observed)) != len(observed)
    ):
        raise ValueError("Frozen ASR artifact video IDs do not match the manifest")
    settings = payload["protocol"]["resolved"]["asr"]
    if timestamp_tolerance_seconds != audio_protocol.timestamp_tolerance_seconds:
        raise ValueError(
            "Frozen ASR timestamp tolerance differs from the composed protocol"
        )
    for row in videos:
        if {
            "executed",
            "first_attempt_executed",
            "first_attempt_success",
            "executed_window_count",
            "failed_window_count",
            "unexecuted_window_count",
            "nonspeech_units",
            "source_group_id",
            "canonical_sha256",
            "canonical_duration_seconds",
        } - row.keys():
            raise ValueError(
                "Frozen ASR video lacks complete execution/support records"
            )
        duration, latency = row.get("duration_seconds"), row.get("latency_seconds")
        if (
            not _finite(duration)
            or duration <= 0
            or (latency is not None and (not _finite(latency) or latency < 0))
            or (latency is None and row.get("executed"))
            or type(row.get("success")) is not bool
            or type(row.get("executed")) is not bool
            or row["first_attempt_executed"] is not row["executed"]
            or row["first_attempt_success"]
            is not (row["success"] if row["executed"] else None)
        ):
            raise ValueError(
                "Frozen ASR artifact contains invalid duration, latency, or status"
            )
        if expected_durations is not None and duration != expected_durations.get(
            row["sample_id"]
        ):
            raise ValueError("Frozen ASR video duration differs from validated inputs")
        windows = row.get("windows")
        expected = _window_starts(
            duration,
            float(settings["window_length_seconds"]),
            float(settings["overlap_seconds"]),
        )
        if type(row.get("scheduled_window_count")) is not int or row.get(
            "scheduled_window_count"
        ) != len(expected):
            raise ValueError("Frozen ASR scheduled window count differs from protocol")
        if (
            not isinstance(windows, list)
            or len(windows) > len(expected)
            or any(
                not isinstance(window, Mapping) or type(window.get("index")) is not int
                for window in windows
            )
            or [window["index"] for window in windows] != list(range(len(windows)))
        ):
            raise ValueError("Frozen ASR artifact has invalid window inventory")
        stitched = []
        for index, window in enumerate(windows):
            if {
                "executed",
                "error",
                "not_executed_reason",
                "transcript",
                "segments",
                "warnings",
            } - window.keys():
                raise ValueError("Frozen ASR window lacks complete execution records")
            if (
                not _finite(window.get("start"))
                or not _finite(window.get("end"))
                or window["start"] != expected[index]
                or window["end"]
                != min(
                    duration, expected[index] + float(settings["window_length_seconds"])
                )
                or type(window.get("executed")) is not bool
                or (
                    type(window.get("success")) is not bool
                    if window.get("executed")
                    else window.get("success") is not None
                )
            ):
                raise ValueError(
                    "Frozen ASR window boundaries or status differ from protocol"
                )
            value = window.get("latency_seconds")
            if (
                (not _finite(value) or value < 0)
                if window.get("executed")
                else (value is not None and (not _finite(value) or value < 0))
            ):
                raise ValueError("Frozen ASR window latency is invalid")
            if (
                window.get("executed")
                and not window["success"]
                and not window.get("error")
            ):
                raise ValueError(
                    "Failed frozen ASR window must retain a failure reason"
                )
            if not window.get("executed") and not window.get("not_executed_reason"):
                raise ValueError("Unexecuted frozen ASR window must retain a reason")
            if window["success"]:
                predictions = validate_prediction(
                    window["transcript"],
                    window["segments"],
                    window["end"] - window["start"],
                    timestamp_tolerance_seconds,
                )
                shifted = [
                    {
                        "text": segment["text"],
                        "start": segment["start"] + window["start"],
                        "end": segment["end"] + window["start"],
                    }
                    for segment in predictions
                ]
                stitched = _stitch_segments(
                    stitched,
                    shifted,
                    int(settings["maximum_overlap_units"]),
                    overlap_start=window["start"],
                    overlap_end=window["start"] + float(settings["overlap_seconds"]),
                )
        if row["success"]:
            if len(windows) != len(expected) or any(
                not window["success"] for window in windows
            ):
                raise ValueError("Partial frozen ASR output cannot be marked complete")
            validate_prediction(
                row["transcript"],
                row["segments"],
                duration,
                timestamp_tolerance_seconds,
            )
        elif not row.get("error"):
            raise ValueError("Failed frozen ASR video must retain a failure reason")
        produced = any(window["success"] for window in windows)
        if row.get("segments") != (stitched if produced else None) or row.get(
            "transcript"
        ) != (" ".join(segment["text"] for segment in stitched) if produced else None):
            raise ValueError(
                "Frozen ASR transcript/segments differ from deterministic window stitching"
            )
        for field, count in {
            "executed_window_count": sum(window["executed"] for window in windows),
            "failed_window_count": sum(
                window["success"] is False for window in windows
            ),
            "unexecuted_window_count": sum(
                not window["executed"] for window in windows
            ),
        }.items():
            if type(row.get(field)) is not int or row[field] != count:
                raise ValueError(
                    "Frozen ASR window support counts contradict recorded outcomes"
                )
        units = row["nonspeech_units"]
        if not isinstance(units, list) or any(
            not isinstance(unit, Mapping) for unit in units
        ):
            raise ValueError("Frozen ASR reviewed nonspeech inventory is invalid")
        seen_units = set()
        for unit in units:
            if (
                unit.get("unit_type") != video_protocol.nonspeech_unit_type
                or unit.get("reviewed") is not True
                or unit.get("sample_id") != row["sample_id"]
            ):
                raise ValueError(
                    "Frozen ASR nonspeech units contradict frozen review labels"
                )
            if unit["unit_type"] == "whole_video":
                outcome = row
                key = "whole_video"
            else:
                index = unit.get("index")
                if type(index) is not int or not 0 <= index < len(windows):
                    raise ValueError(
                        "Frozen ASR reviewed nonspeech unit is not an actual ASR window"
                    )
                outcome, key = windows[index], index
            expected_success = outcome["success"] if outcome["executed"] else None
            if (
                key in seen_units
                or unit.get("executed") is not outcome["executed"]
                or unit.get("success") is not expected_success
            ):
                raise ValueError(
                    "Frozen ASR nonspeech outcomes contradict recorded execution"
                )
            seen_units.add(key)
            expected_event = (
                bool(normalize_transcript(outcome["transcript"]))
                if outcome["success"]
                else None
            )
            if unit.get("false_transcription") is not expected_event:
                raise ValueError(
                    "Frozen ASR nonspeech event contradicts its completed output"
                )
    incomplete = payload.get("execution_error") or any(
        len(row["windows"]) != row["scheduled_window_count"]
        or any(not window.get("executed") for window in row["windows"])
        for row in videos
    )
    if incomplete:
        raise WorkerExecutionError(
            "Frozen ASR preparation is incomplete; retained outputs cannot be reused",
            payload,
        )
    if expected_canonical_checksums is not None and not isinstance(
        expected_canonical_checksums, Mapping
    ):
        raise ValueError(
            "Frozen ASR expected canonical checksums must be a sample-ID mapping"
        )
    for row in videos:
        canonical = row["canonical_sha256"]
        canonical_duration = row["canonical_duration_seconds"]
        if (
            not _sha256(canonical)
            or not _finite(canonical_duration)
            or canonical_duration <= 0
            or abs(canonical_duration - row["duration_seconds"])
            > video_protocol.manifest_duration_tolerance_seconds
        ):
            raise ValueError(
                "Frozen ASR decoded video lacks a valid complete canonical checksum/duration"
            )
        if expected_canonical_checksums is not None:
            expected = expected_canonical_checksums.get(row["sample_id"])
            if not _sha256(expected) or canonical != expected:
                raise ValueError(
                    f"Frozen ASR canonical checksum differs from sealed inputs for {row['sample_id']}"
                )
    parameters = payload["parameters"]
    if not isinstance(parameters, Mapping) or parameters.get("device") != payload.get(
        "device"
    ):
        raise ValueError("Frozen ASR runtime parameters contradict its device")
    if payload.get("device") not in {"cpu", "cuda"}:
        raise ValueError("Frozen ASR device must be CPU or CUDA")
    if payload.get("device") == "cpu" and metrics["peak_vram_mb"] is not None:
        raise ValueError("Frozen ASR CPU VRAM must be null")
    _validate_runtime(payload, video_protocol, audio_protocol)
    if checksum != sha256_file(path):
        raise ValueError("Frozen ASR artifact content changed during validation")
    return dict(payload), checksum


def _validate_runtime(payload, video_protocol, audio_protocol):
    """Reject artifacts without full lifecycle, placement, and software evidence."""
    candidate = payload["audio_candidate"]
    declared = profiles(audio_protocol)
    if candidate not in declared:
        raise ValueError("Frozen ASR selected candidate is outside its audio protocol")
    parameters = payload["parameters"]
    profile = declared[candidate]
    execution = video_protocol.profile(payload["profile"])
    if payload["device"] not in (execution.devices or (execution.device,)):
        raise ValueError("Frozen ASR device violates its frozen execution profile")
    expected = {
        **{
            name: value
            for name, value in profile.decoding.items()
            if name not in {"model_id", "cpu_dtype", "cuda_dtype"}
        },
        "candidate": candidate,
        "model": profile.model,
        "backend": profile.backend,
        "device": payload["device"],
        "dtype": audio_protocol.dtype(candidate, payload["device"]),
        "decoder": profile.decoder,
        "timestamp_method": profile.timestamp_method,
        "language": profile.decoding["language"],
        "batch_size": audio_protocol.batch_size,
        "sample_rate_hz": audio_protocol.audio["sample_rate_hz"],
        "seed": video_protocol.meta.seed,
        "warmups": execution.warmups,
        "repetitions": video_protocol.frozen_asr_repetitions,
        "stitching": dict(video_protocol.stitching),
        "window_length_seconds": video_protocol.window_length_seconds,
        "overlap_seconds": video_protocol.overlap_seconds,
        **{
            name: payload[name]
            for name in (
                "model_path",
                "model_revision",
                "selection_revision",
                "model_cache_manifest_sha256",
            )
        },
    }
    if profile.backend in {"transformers", "moss"}:
        expected["num_beams"] = 1
    elif candidate == "canary-180m":
        expected["decoding_strategy"] = "beam"
    if any(parameters.get(name) != value for name, value in expected.items()):
        raise ValueError(
            "Frozen ASR runtime parameters contradict its model/composed protocol identity"
        )
    if not parameters.get("runtime_version") or not isinstance(
        parameters.get("package_versions"), Mapping
    ):
        raise ValueError("Frozen ASR runtime software provenance is missing")
    runtime_distribution = {
        "transformers": "transformers",
        "nemo": "nemo_toolkit",
        "moss": "moss-transcribe-diarize",
    }[profile.backend]
    if parameters["runtime_version"] != payload["execution_identity"][
        "package_versions"
    ].get(runtime_distribution):
        raise ValueError("Frozen ASR runtime version contradicts its software identity")
    if set(payload["execution_identity"]["package_versions"]) - {"numpy"} != set(
        parameters["package_versions"]
    ):
        raise ValueError("Frozen ASR installed dependency versions are incomplete")
    for name, value in parameters["package_versions"].items():
        if payload["execution_identity"]["package_versions"].get(name) != value:
            raise ValueError(
                "Frozen ASR installed runtime versions contradict its software identity"
            )
    metrics = payload["metrics"]
    if (
        metrics["cold_model_load_seconds"] is None
        or metrics["peak_process_tree_ram_mb"] is None
    ):
        raise ValueError("Frozen ASR readiness/RAM measurement is unavailable")
    if payload["device"] == "cuda" and (
        metrics["peak_vram_mb"] is None
        or parameters.get("vram_measurement_method") != "nvml-device-total"
        or any(
            not isinstance(payload[name], Mapping)
            or payload[name].get("status") != "qualified"
            for name in ("placement_before", "placement_after")
        )
    ):
        raise ValueError(
            "Frozen ASR CUDA placement/telemetry is unavailable or violates no-offload"
        )
    if (
        not isinstance(payload["metric_statuses"], Mapping)
        or set(METRIC_DIRECTIONS) - payload["metric_statuses"].keys()
    ):
        raise ValueError("Frozen ASR aggregate metric statuses are missing")
    for name in METRIC_DIRECTIONS:
        record = payload["metric_statuses"][name]
        if (
            not isinstance(record, Mapping)
            or record.get("value") != metrics[name]
            or record.get("status")
            not in {"scored", "inapplicable", "unavailable", "incomplete"}
            or not isinstance(record.get("support"), Mapping)
        ):
            raise ValueError(
                "Frozen ASR metric status contradicts recorded measurements"
            )
    if (
        payload["resource_measurement_scope"] != "full_lifecycle"
        or not isinstance(payload["resource_samples"], list)
        or not payload["resource_samples"]
    ):
        raise ValueError("Frozen ASR completed lifecycle resource evidence is missing")
    commands = payload["ffmpeg_commands"]
    if not isinstance(commands, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("command"), list)
        or not row["command"]
        or any(not isinstance(arg, str) for arg in row["command"])
        for row in commands
    ):
        raise ValueError("Frozen ASR FFmpeg command evidence is invalid")
    decoded = [row.get("sample_id") for row in commands if row.get("phase") == "decode"]
    if sorted(decoded) != sorted(row["sample_id"] for row in payload["videos"]):
        raise ValueError("Frozen ASR must record exactly one decode per video")


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
