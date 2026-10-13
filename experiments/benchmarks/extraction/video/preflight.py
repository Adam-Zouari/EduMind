"""Video-specific GPU qualification orchestration."""

from __future__ import annotations

import json
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
from experiments.benchmarks.extraction.media import ffmpeg_version
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
    data_validation_report,
):
    validation = json.loads(Path(data_validation_report).read_text(encoding="utf-8"))
    stress = qualification_inputs(manifest, validation)
    declared, fingerprint, context = video_qualification_identity(
        protocol,
        audio_protocol,
        document_protocol,
        audio_candidate,
        image_candidate,
        manifest,
        stress,
        validation,
    )
    stress = [
        {
            **item,
            "source_path": str((PROJECT_ROOT / item["source_path"]).resolve()),
            "canonical_sha256": validation["metadata"]["samples"][str(item["id"])][
                "canonical_sha256"
            ],
        }
        for item in stress
    ]
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
            result["supervisor_resource_evidence"] = supervision
        return result

    return run_preflight(
        benchmark="video",
        candidates=declared,
        fingerprint=fingerprint,
        context={
            **context,
            "stress_manifest": str(manifest_path),
            "stress_manifest_checksum": manifest.fingerprint,
            "stress_sample_ids": [str(item["id"]) for item in stress],
            "audio_decision": str(audio_decision),
            "image_decision": str(image_decision),
            "data_validation_report": str(data_validation_report),
        },
        probe=probe,
        required_groups={
            "frozen-asr": (f"frozen-asr|{audio_candidate}",),
            "visual": tuple(
                candidate for candidate in declared if candidate.startswith("visual|")
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
    validation,
):
    threshold = protocol.selected_scene_threshold
    visual_candidates = (
        *fixed_candidates(protocol),
        *scene_candidates(protocol),
        *(hybrid_candidates(protocol, threshold) if threshold is not None else ()),
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
            "visual_resource_policy": "backend-specific-reporting",
            "audio_candidate": audio_candidate,
            "image_candidate": image_candidate,
            "audio_dtype": audio_protocol.dtype(audio_candidate, execution.device),
            "ffmpeg_version": ffmpeg_version(),
            "preflight": asdict(protocol.preflight),
        },
        input_envelope={
            **stress_input_identity(manifest, stress),
            "window_length_seconds": protocol.window_length_seconds,
            "overlap_seconds": protocol.overlap_seconds,
            "fixed_intervals": protocol.fixed_intervals,
            "scene_thresholds": protocol.scene_thresholds,
            "hybrid_gaps": protocol.hybrid_gaps,
            "tested_video_envelopes": [
                video_envelope(item, validation) for item in stress
            ],
        },
    )
    return declared, fingerprint, context


def video_envelope(item, validation):
    metadata = validation["metadata"]["samples"][str(item["id"])]
    return {
        "duration_seconds": max(
            float(item["duration_seconds"]),
            float(metadata["canonical_duration_seconds"]),
            float(metadata["actual_duration_seconds"]),
        ),
        "width": metadata["width"],
        "height": metadata["height"],
    }


def qualification_inputs(manifest, validation):
    """Keep reviewed inputs not dominated in duration and frame dimensions."""
    items = sorted(
        (item for item in manifest.samples if item.get("kind") == "video"),
        key=lambda item: str(item["id"]),
    )
    selected = []
    for item in items:
        envelope = video_envelope(item, validation)
        if any(
            all(tested[key] >= envelope[key] for key in envelope)
            for _, tested in selected
        ):
            continue
        selected = [
            (other, tested)
            for other, tested in selected
            if not all(envelope[key] >= tested[key] for key in envelope)
        ]
        selected.append((item, envelope))
    if not selected:
        raise ValueError("Video preflight requires reviewed stress videos")
    return tuple(item for item, _ in selected)


def validate_input_envelope(items, validation, qualification):
    tested = qualification["context"]["input_envelope"]["tested_video_envelopes"]
    for item in items:
        envelope = video_envelope(item, validation)
        if not any(
            all(stress[key] >= envelope[key] for key in envelope) for stress in tested
        ):
            raise ValueError(
                f"Video {item['id']} exceeds the qualified input envelope; review larger development stress inputs and rerun preflight"
            )


def _run_frozen_asr_probe(candidate, stress, audio_lock, protocol, audio_protocol):
    return run_json_worker(
        Path(__file__).with_name("frozen_asr_worker.py"),
        {
            "candidate": candidate,
            "model_lock": audio_lock,
            "items": list(stress),
            "device": "cuda",
            "protocol": protocol.meta.worker_payload(),
            "audio_protocol": audio_protocol.meta.worker_payload(),
            "mode": "preflight",
        },
        device="cuda",
        prefix="edumind-video-asr-preflight-",
        error_label=f"video frozen-ASR preflight {candidate}",
        require_vram_measurement=True,
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
    return run_json_worker(
        Path(__file__).with_name("visual_worker.py"),
        {
            "candidate": candidate,
            "items": list(stress),
            "image_engine": image_profile.runtime_engine,
            "image_candidate": image_candidate,
            "image_revision": str(image_entry.get("revision", "")),
            "image_options": image_options,
            "device": "cuda",
            "protocol": protocol.meta.worker_payload(),
            "document_protocol": document_protocol.meta.worker_payload(),
            "mode": "preflight",
        },
        device="cuda",
        prefix="edumind-video-visual-preflight-",
        error_label=f"video visual preflight {candidate}",
        require_vram_measurement=True,
        telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocol.preflight.poll_interval_seconds,
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
    )
