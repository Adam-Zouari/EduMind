"""Creation and validation of the phase-level frozen ASR artifact."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.extraction.audio.adapters import ASR_PROFILES
from experiments.benchmarks.extraction.process import run_json_worker
from experiments.benchmarks.extraction.video.protocol import VideoProtocolLock
from experiments.benchmarks.preparation.models import load_selected_model_lock


def create_frozen_asr_artifact(
    output_path: Path,
    *,
    items: Sequence[Mapping[str, object]],
    manifest_checksum: str,
    protocol: VideoProtocolLock,
    audio_candidate: str,
    audio_decision_path: Path,
    device: str,
    ffmpeg_version: str,
) -> Path:
    if audio_candidate not in ASR_PROFILES:
        raise ValueError(f"Unknown selected ASR profile: {audio_candidate}")
    lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(ASR_PROFILES[audio_candidate].model,),
    )
    payload = {
        "candidate": audio_candidate,
        "model_lock": lock,
        "items": list(items),
        "device": device,
        "warmups": 2,
        "window_length_seconds": protocol.window_length_seconds,
        "overlap_seconds": protocol.overlap_seconds,
        "maximum_overlap_tokens": int(protocol.stitching["maximum_overlap_tokens"]),
    }
    result = run_json_worker(
        Path(__file__).with_name("frozen_asr_worker.py"),
        payload,
        device=device,
        prefix="edumind-frozen-asr-",
        error_label="Frozen video ASR worker",
    )
    entry = lock[ASR_PROFILES[audio_candidate].model]
    artifact = {
        "schema_version": 1,
        "artifact_type": "FrozenASRArtifact",
        "run_id": f"video-asr-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "created_unix_seconds": time.time(),
        "manifest_checksum": manifest_checksum,
        "protocol_checksum": protocol.checksum,
        "model_decision_fingerprint": stable_hash(
            {
                "path": str(audio_decision_path.resolve()),
                "sha256": sha256_file(audio_decision_path),
                "candidate": audio_candidate,
            }
        ),
        "audio_candidate": audio_candidate,
        "model_path": entry.get("model_path", ""),
        "model_revision": entry.get("revision", ""),
        "selection_revision": entry.get("selection_revision", ""),
        "model_cache_manifest_sha256": entry.get("model_cache_manifest_sha256", ""),
        "submodels": entry.get("submodels", []),
        "protocol": {
            "window_length_seconds": protocol.window_length_seconds,
            "overlap_seconds": protocol.overlap_seconds,
            "stitching": dict(protocol.stitching),
        },
        "ffmpeg_version": ffmpeg_version,
        **result,
    }
    atomic_write_json(output_path, artifact)
    return output_path


def load_frozen_asr_artifact(
    path: Path,
    *,
    manifest_checksum: str,
    protocol_checksum: str,
    sample_ids: Sequence[str],
) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("artifact_type") != "FrozenASRArtifact"
    ):
        raise ValueError(f"{path} is not a FrozenASRArtifact schema_version 1")
    if payload.get("manifest_checksum") != manifest_checksum:
        raise ValueError("Frozen ASR artifact manifest checksum mismatch")
    if payload.get("protocol_checksum") != protocol_checksum:
        raise ValueError("Frozen ASR artifact protocol checksum mismatch")
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise ValueError("Frozen ASR artifact lacks per-video outputs")
    observed = [str(item.get("sample_id", "")) for item in videos if isinstance(item, Mapping)]
    if sorted(observed) != sorted(sample_ids) or len(observed) != len(set(observed)):
        raise ValueError("Frozen ASR artifact video IDs do not match the manifest")
    required = {
        "run_id",
        "model_decision_fingerprint",
        "ffmpeg_version",
        "ffmpeg_commands",
        "metrics",
        "parameters",
        "audio_candidate",
        "model_path",
        "model_revision",
        "selection_revision",
        "model_cache_manifest_sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("Frozen ASR artifact lacks fields: " + ", ".join(missing))
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or any(
        name not in metrics
        for name in (
            "word_error_rate",
            "real_time_factor",
            "total_latency_seconds",
            "cold_model_load_seconds",
            "peak_process_tree_ram_mb",
            "peak_vram_mb",
        )
    ):
        raise ValueError("Frozen ASR artifact lacks required aggregate metrics")
    for video in videos:
        if not isinstance(video, Mapping):
            raise ValueError("Frozen ASR artifact contains a malformed video row")
        transcript = str(video.get("transcript", "")).strip()
        segments = video.get("segments")
        windows = video.get("windows")
        if not isinstance(segments, list) or not isinstance(windows, list) or not windows:
            raise ValueError("Frozen ASR artifact lacks timestamp segments or window records")
        if transcript and not segments:
            raise ValueError("Frozen ASR lexical transcript lacks timestamp segments")
        if not transcript and segments:
            raise ValueError("Frozen ASR empty transcript has timestamp segments")
        latency = float(video.get("latency_seconds", -1))
        real_time_factor = float(video.get("real_time_factor", -1))
        if latency < 0 or real_time_factor < 0:
            raise ValueError("Frozen ASR artifact lacks per-video latency or RTF")
        previous = -1.0
        duration = float(video.get("duration_seconds", -1))
        for segment in segments:
            if not isinstance(segment, Mapping) or not str(segment.get("text", "")).strip():
                raise ValueError("Frozen ASR artifact contains a malformed timestamp segment")
            start, end = float(segment.get("start", -1)), float(segment.get("end", -1))
            if start < previous or start < 0 or end <= start or end > duration + 0.1:
                raise ValueError("Frozen ASR artifact contains invalid timestamp boundaries")
            previous = start
    return dict(payload), sha256_file(path)
