"""Video-specific GPU qualification orchestration."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.preflight import (
    current_qualification_fingerprint,
    model_lock_fingerprints,
    run_preflight,
    stress_input_identity,
)
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.extraction.audio.adapters import profiles as audio_profiles
from experiments.benchmarks.extraction.document.profiles import (
    lock_paths,
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.runner import (
    validate_prepared_components,
)
from experiments.benchmarks.extraction.video.candidates import (
    fixed_candidates,
    hybrid_candidates,
    scene_candidates,
)
from experiments.benchmarks.preparation.models import load_selected_model_lock


def run_video_preflight(
    items,
    *,
    manifest,
    manifest_path,
    protocol,
    audio_protocol,
    document_protocol,
    audio_candidate,
    audio_decision,
    image_candidate,
    image_decision,
    no_mlflow,
):
    stress = max(items, key=lambda item: float(item["duration_seconds"]))
    declared, fingerprint, context = video_qualification_identity(
        protocol,
        audio_protocol,
        document_protocol,
        audio_candidate,
        image_candidate,
        manifest,
        stress,
    )
    audio_profile = audio_profiles(audio_protocol)[audio_candidate]
    audio_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(audio_profile.model,),
    )
    image_profile = parse_document_profile(image_candidate)
    image_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(image_profile.lock_candidate,),
    )
    image_entry = image_lock[image_profile.lock_candidate]
    validate_prepared_components(image_candidate, image_entry, document_protocol)
    image_options = {
        **document_protocol.parser_options(image_profile.runtime_engine),
        **image_profile.options,
        **lock_paths(image_entry),
    }

    def probe(component: str):
        kind, candidate = component.split("|", 1)
        if kind == "frozen-asr":
            result = _run_frozen_asr_probe(
                candidate,
                stress,
                audio_lock,
                protocol,
                audio_protocol,
            )
        else:
            result = _run_visual_probe(
                candidate,
                stress,
                image_profile,
                image_candidate,
                image_entry,
                image_options,
                protocol,
                document_protocol,
            )
        supervision = result.pop("_worker_supervision", {})
        if isinstance(supervision, dict):
            result.update(supervision)
        return result

    return run_preflight(
        benchmark="video",
        candidates=declared,
        fingerprint=fingerprint,
        context={
            **context,
            "stress_manifest": str(manifest_path),
            "stress_manifest_checksum": manifest.fingerprint,
            "stress_sample_id": str(stress["id"]),
            "audio_decision": str(audio_decision),
            "image_decision": str(image_decision),
        },
        probe=probe,
        required_groups={
            "frozen-asr": (f"frozen-asr|{audio_candidate}",),
            "visual": tuple(
                candidate
                for candidate in declared
                if candidate.startswith("visual|")
            ),
        },
        decision_files={"audio": audio_decision, "document": image_decision},
        no_mlflow=no_mlflow,
    )


def video_qualification_identity(
    protocol,
    audio_protocol,
    document_protocol,
    audio_candidate,
    image_candidate,
    manifest,
    stress,
):
    threshold = protocol.selected_scene_threshold or protocol.smoke_scene_threshold
    visual_candidates = tuple(
        dict.fromkeys(
            (
                *fixed_candidates(protocol),
                *scene_candidates(protocol),
                *hybrid_candidates(protocol, threshold),
            )
        )
    )
    declared = (
        f"frozen-asr|{audio_candidate}",
        *(f"visual|{candidate}" for candidate in visual_candidates),
    )
    audio_profile = audio_profiles(audio_protocol)[audio_candidate]
    image_profile = parse_document_profile(image_candidate)
    lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(audio_profile.model, image_profile.lock_candidate),
    )
    execution = protocol.profile("development")
    fingerprint, context = current_qualification_fingerprint(
        benchmark="video",
        candidates=declared,
        protocols={
            "video": protocol.meta,
            "audio": audio_protocol.meta,
            "document": document_protocol.meta,
        },
        revisions=model_lock_fingerprints(lock),
        execution={
            "device": execution.device,
            "dtype": execution.dtype,
            "batch_size": execution.batch_size,
            "asr_vram_limit_mb": audio_protocol.authoritative_peak_vram_mb,
            "visual_resource_policy": "backend-specific-reporting",
            "audio_candidate": audio_candidate,
            "image_candidate": image_candidate,
            "preflight": asdict(protocol.preflight),
        },
        input_envelope={
            **stress_input_identity(manifest, (stress,)),
            "window_length_seconds": protocol.window_length_seconds,
            "overlap_seconds": protocol.overlap_seconds,
            "fixed_intervals": protocol.fixed_intervals,
            "scene_thresholds": protocol.scene_thresholds,
            "hybrid_gaps": protocol.hybrid_gaps,
            "tested_video_duration_seconds": float(stress["duration_seconds"]),
        },
    )
    return declared, fingerprint, context


def _run_frozen_asr_probe(
    candidate, stress, audio_lock, protocol, audio_protocol
):
    return run_json_worker(
        Path(__file__).with_name("frozen_asr_worker.py"),
        {
            "candidate": candidate,
            "model_lock": audio_lock,
            "items": [stress],
            "device": "cuda",
            "warmups": protocol.preflight.warmups,
            "window_length_seconds": protocol.window_length_seconds,
            "overlap_seconds": protocol.overlap_seconds,
            "maximum_overlap_tokens": int(
                protocol.stitching["maximum_overlap_tokens"]
            ),
            "protocol": protocol.meta.worker_payload(),
            "audio_protocol": audio_protocol.meta.worker_payload(),
            "mode": "preflight",
        },
        device="cuda",
        prefix="edumind-video-asr-preflight-",
        error_label=f"video frozen-ASR preflight {candidate}",
        vram_limit_mb=audio_protocol.authoritative_peak_vram_mb,
        telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocol.preflight.poll_interval_seconds,
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
    )


def _run_visual_probe(
    candidate,
    stress,
    image_profile,
    image_candidate,
    image_entry,
    image_options,
    protocol,
    document_protocol,
):
    temporary_root = Path(os.environ.get("TEMP", tempfile.gettempdir()))
    return run_json_worker(
        Path(__file__).with_name("visual_worker.py"),
        {
            "candidate": candidate,
            "items": [stress],
            "image_engine": image_profile.runtime_engine,
            "image_candidate": image_candidate,
            "image_revision": str(image_entry.get("revision", "")),
            "image_options": image_options,
            "device": "cuda",
            "warmups": protocol.preflight.warmups,
            "repetitions": protocol.preflight.repetitions,
            "bootstrap_resamples": protocol.preflight.bootstrap_resamples,
            "seed": protocol.meta.seed,
            "protocol": protocol.meta.worker_payload(),
            "document_protocol": document_protocol.meta.worker_payload(),
            "mode": "preflight",
        },
        device="cuda",
        prefix="edumind-video-visual-preflight-",
        error_label=f"video visual preflight {candidate}",
        temporary_root=temporary_root,
        require_vram_measurement=True,
        telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocol.preflight.poll_interval_seconds,
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
    )
