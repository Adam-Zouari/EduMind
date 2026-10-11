"""Versioned frozen-ASR artifacts: exact inputs, model identity, and complete outcomes."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.extraction.audio.adapters import profiles
from experiments.benchmarks.extraction.audio.evaluate import validate_prediction
from experiments.benchmarks.extraction.audio.protocol import (
    protocol_from_worker as audio_protocol_from_worker,
)
from experiments.benchmarks.extraction.video.frozen_asr_worker import (
    METRIC_DIRECTIONS,
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
        extraction / "scoring.py",
        extraction / "media.py",
        *(
            PROJECT_ROOT / "experiments/benchmarks/common" / name
            for name in ("metrics.py", "process.py", "resources.py", "protocol.py")
        ),
        PROJECT_ROOT / "src/edumind/common/artifacts.py",
        PROJECT_ROOT / "src/edumind/common/model_placement.py",
        *sorted((PROJECT_ROOT / "requirements").glob("*.lock")),
    ]
    return {
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
    lock, identity = model_identity(
        audio_candidate, audio_decision_path, audio_protocol
    )
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
        vram_limit_mb=audio_protocol.authoritative_peak_vram_mb
        if device == "cuda"
        else None,
        telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocol.preflight.poll_interval_seconds,
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
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
        "execution_identity": execution_identity(),
        **result,
    }
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
):
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
    }
    if required - payload.keys():
        raise ValueError(
            "Frozen ASR artifact lacks required provenance or measurements"
        )
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
    if sorted(observed) != sorted(sample_ids) or len(set(observed)) != len(observed):
        raise ValueError("Frozen ASR artifact video IDs do not match the manifest")
    settings = payload["protocol"]["resolved"]["asr"]
    for row in videos:
        duration, latency = (
            float(row["duration_seconds"]),
            float(row["latency_seconds"]),
        )
        if (
            not math.isfinite(duration)
            or duration <= 0
            or not math.isfinite(latency)
            or latency < 0
            or type(row.get("success")) is not bool
        ):
            raise ValueError(
                "Frozen ASR artifact contains invalid duration, latency, or status"
            )
        if (
            expected_durations is not None
            and duration != expected_durations[row["sample_id"]]
        ):
            raise ValueError("Frozen ASR video duration differs from validated inputs")
        windows = row.get("windows")
        expected = _window_starts(
            duration,
            float(settings["window_length_seconds"]),
            float(settings["overlap_seconds"]),
        )
        if row.get("scheduled_window_count") != len(expected):
            raise ValueError("Frozen ASR scheduled window count differs from protocol")
        if (
            not isinstance(windows, list)
            or len(windows) > len(expected)
            or [window["index"] for window in windows] != list(range(len(windows)))
        ):
            raise ValueError("Frozen ASR artifact has invalid window inventory")
        for index, window in enumerate(windows):
            if (
                window["start"] != expected[index]
                or window["end"]
                != min(
                    duration, expected[index] + float(settings["window_length_seconds"])
                )
                or type(window.get("success")) is not bool
            ):
                raise ValueError(
                    "Frozen ASR window boundaries or status differ from protocol"
                )
            value = window.get("latency_seconds")
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Frozen ASR window latency is invalid")
            if window["success"]:
                validate_prediction(
                    window["transcript"],
                    window["segments"],
                    window["end"] - window["start"],
                    timestamp_tolerance_seconds,
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
    return dict(payload), checksum
