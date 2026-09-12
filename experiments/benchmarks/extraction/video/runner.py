"""Dedicated orchestration for frozen-ASR and visual-only video benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.audio.adapters import ASR_PROFILES
from experiments.benchmarks.extraction.document.profiles import (
    lock_paths,
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.runner import validate_prepared_components
from experiments.benchmarks.extraction.media import ffmpeg_version, media_duration
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.extraction.video.candidates import (
    FIXED_CANDIDATES,
    SCENE_CANDIDATES,
    all_candidates,
    hybrid_candidates,
    parse_candidate,
)
from experiments.benchmarks.extraction.video.frozen_asr import (
    create_frozen_asr_artifact,
    load_frozen_asr_artifact,
)
from experiments.benchmarks.extraction.video.metrics import METRIC_DIRECTIONS
from experiments.benchmarks.extraction.video.protocol import load_protocol_lock
from experiments.benchmarks.preparation.models import load_selected_model_lock

PROFILE_STAGE = {
    "smoke": "video-smoke",
    "standard": "video-development",
    "full": "video-validation",
    "locked": "video-locked-test",
}
EXPECTED_COUNTS = {"standard": 18, "full": 6, "locked": 6}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark visual video extraction")
    parser.add_argument(
        "--profile", choices=("smoke", "standard", "full", "locked"), default="smoke"
    )
    parser.add_argument(
        "--phase", choices=("frozen-asr", "fixed", "scene", "hybrid", "all"), default="all"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--frozen-asr", type=Path)
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--scene-selection", type=Path)
    parser.add_argument("--document-selection", type=Path)
    parser.add_argument("--audio-selection", type=Path)
    parser.add_argument("--image-candidate", help="Smoke-only selected document profile")
    parser.add_argument("--audio-candidate", help="Smoke-only selected ASR profile")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()

    manifest_path = (arguments.manifest or _manifest(arguments.profile)).resolve()
    manifest = load_manifest(manifest_path)
    expected_split = {
        "smoke": "smoke",
        "standard": "development",
        "full": "validation",
        "locked": "locked-test",
    }[arguments.profile]
    if manifest.split != expected_split:
        raise ValueError(
            f"Video {arguments.profile} requires split {expected_split}, "
            f"received {manifest.split}"
        )
    items = [dict(item) for item in manifest.samples if item.get("kind") == "video"]
    _validate_manifest(items, arguments.profile)
    protocol_path = arguments.protocol_lock or (
        PROJECT_ROOT / "data/benchmarks/extraction/video-protocol-smoke.json"
        if arguments.profile == "smoke"
        else None
    )
    if protocol_path is None:
        raise ValueError("Authoritative video execution requires --protocol-lock")
    protocol = load_protocol_lock(
        protocol_path,
        manifest_checksum=manifest.fingerprint,
        profile=arguments.profile,
    )
    ffmpeg_identity = ffmpeg_version()
    if arguments.phase == "frozen-asr":
        if arguments.frozen_asr is None:
            raise ValueError("The frozen-asr phase requires --frozen-asr OUTPUT_JSON")
        audio_candidate, decision_path = _selected_audio(arguments)
        artifact_path = create_frozen_asr_artifact(
            arguments.frozen_asr,
            items=items,
            manifest_checksum=manifest.fingerprint,
            protocol=protocol,
            audio_candidate=audio_candidate,
            audio_decision_path=decision_path,
            device=arguments.device,
            ffmpeg_version=ffmpeg_identity,
        )
        asr_result = _record_frozen_asr(
            artifact_path,
            profile=arguments.profile,
            manifest_path=manifest_path,
            manifest_name=manifest.name,
            manifest_checksum=manifest.fingerprint,
            protocol_path=protocol.path,
            protocol_checksum=protocol.checksum,
            audio_decision=decision_path,
            no_mlflow=arguments.no_mlflow,
        )
        print(
            json.dumps(
                {
                    "frozen_asr": str(arguments.frozen_asr.resolve()),
                    "run_id": asr_result.run_id,
                    "complete": asr_result.complete,
                    "artifacts": str(asr_result.artifact_directory),
                },
                indent=2,
            )
        )
        return 0 if asr_result.complete else 2

    if arguments.frozen_asr is None:
        raise ValueError("Visual video phases require --frozen-asr ARTIFACT_JSON")
    frozen, frozen_checksum = load_frozen_asr_artifact(
        arguments.frozen_asr,
        manifest_checksum=manifest.fingerprint,
        protocol_checksum=protocol.checksum,
        sample_ids=[str(item["id"]) for item in items],
    )
    image_candidate, image_decision = _selected_image(arguments)
    candidates, candidate_decisions = _candidates(arguments, protocol.hybrid_scene_threshold)
    result = run_visual_benchmark(
        arguments.profile,
        candidates,
        items=items,
        manifest_path=manifest_path,
        manifest_name=manifest.name,
        manifest_checksum=manifest.fingerprint,
        protocol_path=protocol.path,
        protocol_checksum=protocol.checksum,
        occurrence_matching=protocol.occurrence_matching,
        frozen_asr_path=arguments.frozen_asr.resolve(),
        frozen_asr=frozen,
        frozen_asr_checksum=frozen_checksum,
        image_candidate=image_candidate,
        image_decision=image_decision,
        decision_files=candidate_decisions,
        device=arguments.device,
        ffmpeg_version=ffmpeg_identity,
        no_mlflow=arguments.no_mlflow,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "complete": result.complete,
                "artifacts": str(result.artifact_directory),
            },
            indent=2,
        )
    )
    return 0 if result.complete else 2


def run_visual_benchmark(
    profile,
    candidates,
    *,
    items,
    manifest_path,
    manifest_name,
    manifest_checksum,
    protocol_path,
    protocol_checksum,
    occurrence_matching,
    frozen_asr_path,
    frozen_asr,
    frozen_asr_checksum,
    image_candidate,
    image_decision,
    decision_files,
    device,
    ffmpeg_version,
    no_mlflow,
):
    document_profile = parse_document_profile(image_candidate)
    engine = document_profile.runtime_engine
    image_options = dict(document_profile.options)
    lock_name = document_profile.lock_candidate
    lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(lock_name,),
    )
    entry = lock[lock_name]
    validate_prepared_components(image_candidate, entry)
    image_options.update(lock_paths(entry))
    repetitions = 1 if profile == "smoke" else 3
    resamples = 0 if profile == "smoke" else 10_000
    stage = PROFILE_STAGE[profile]
    strategies = {parse_candidate(candidate).strategy for candidate in candidates}
    if profile == "standard" and len(strategies) == 1:
        stage = f"{stage}-{next(iter(strategies))}"
    plan = BenchmarkPlan(
        "extraction",
        stage,
        profile,
        manifest_name,
        tuple(candidates),
        repetitions=repetitions,
        bootstrap_resamples=resamples,
        settings={
            "device": device,
            "protocol_checksum": protocol_checksum,
            "frozen_asr_run_id": frozen_asr["run_id"],
            "frozen_asr_checksum": frozen_asr_checksum,
            "image_candidate": image_candidate,
            "ffmpeg_version": ffmpeg_version,
        },
    )

    def evaluate(candidate: str):
        output = _run_visual_worker(
            candidate,
            items,
            image_engine=engine,
            image_revision=str(entry.get("revision", "")),
            image_options=image_options,
            occurrence_matching=occurrence_matching,
            device=device,
            warmups=plan.warmups,
            repetitions=plan.repetitions,
            bootstrap_resamples=plan.bootstrap_resamples,
            seed=plan.seed,
        )
        rows = output["samples"]
        samples = [
            SampleResult(
                str(row["sample_id"]),
                {
                    name: float(row[name])
                    for name in (
                        "visual_content_precision",
                        "visual_content_recall",
                        "visual_content_f1",
                        "mean_visual_first_detection_delay_seconds",
                        "timed_visual_occurrence_coverage",
                        "duplicate_visual_text_rate",
                    )
                    if row.get(name) is not None
                },
                float(row["quality_latency_seconds"]),
                {
                    "selected_frame_count": row["selected_frame_count"],
                    "frozen_asr_run_id": frozen_asr["run_id"],
                },
            )
            for row in rows
        ]
        parameters = {
            **output["parameters"],
            "image_candidate": image_candidate,
            "image_model_path": entry.get("model_path", ""),
            "image_model_cache_manifest_sha256": entry.get(
                "model_cache_manifest_sha256", ""
            ),
            "image_paddle_cache_manifest_sha256": entry.get(
                "paddle_cache_manifest_sha256", ""
            ),
            "image_prepared_components": entry.get("prepared_components", []),
            "image_system_components": entry.get("system_components", {}),
            "manifest_checksum": manifest_checksum,
            "protocol_checksum": protocol_checksum,
            "frozen_asr_run_id": frozen_asr["run_id"],
            "frozen_asr_checksum": frozen_asr_checksum,
            "ffmpeg_version": ffmpeg_version,
        }
        return (
            samples,
            output["operational"],
            output["metrics"],
            parameters,
            output["intervals"],
            {
                "samples": rows,
                "timings": output["timings"],
                "ffmpeg_commands": output["ffmpeg_commands"],
            },
        )

    all_directions = METRIC_DIRECTIONS
    decisions = dict(decision_files)
    if image_decision:
        decisions["document"] = image_decision
    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest_checksum,
        directions=all_directions,
        primary_metric=(
            "visual_content_f1",
            "mean_visual_first_detection_delay_seconds",
            "timed_visual_occurrence_coverage",
        ),
        required_metrics=tuple(all_directions),
        paired_metrics=(),
        revisions={
            "image_parser": str(entry.get("selection_revision", entry.get("revision", ""))),
            "ffmpeg": ffmpeg_version,
            "frozen_asr": frozen_asr_checksum,
        },
        decision_files=decisions,
        input_artifacts={
            "manifest": manifest_path,
            "video_protocol_lock": protocol_path,
            "frozen_asr": frozen_asr_path,
        },
        no_mlflow=no_mlflow,
        monitor_resources=False,
        operational_prefix="",
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
        nullable_metrics=(
            "mean_visual_first_detection_delay_seconds",
            "duplicate_visual_text_rate",
        ),
    )


def _run_visual_worker(candidate, items, **settings):
    temporary_root = Path(os.environ.get("TEMP", tempfile.gettempdir()))
    return run_json_worker(
        Path(__file__).with_name("visual_worker.py"),
        {"candidate": candidate, "items": items, **settings},
        device=str(settings["device"]),
        prefix="edumind-video-candidate-",
        error_label="Visual video worker",
        temporary_root=temporary_root,
    )


def _record_frozen_asr(
    artifact_path,
    *,
    profile,
    manifest_path,
    manifest_name,
    manifest_checksum,
    protocol_path,
    protocol_checksum,
    audio_decision,
    no_mlflow,
):
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_checksum = sha256_file(artifact_path)
    candidate = str(artifact["audio_candidate"])
    videos = artifact["videos"]
    metrics = artifact["metrics"]
    plan = BenchmarkPlan(
        "extraction",
        f"video-input-asr-{PROFILE_STAGE[profile].removeprefix('video-')}",
        profile,
        manifest_name,
        (candidate,),
        repetitions=1,
        bootstrap_resamples=0 if profile == "smoke" else 10_000,
        settings={
            "protocol_checksum": protocol_checksum,
            "frozen_asr_checksum": artifact_checksum,
        },
    )
    directions_map = {
        "word_error_rate": "min",
        "real_time_factor": "min",
        "total_latency_seconds": "min",
        "cold_model_load_seconds": "min",
        "peak_process_tree_ram_mb": "min",
        "peak_vram_mb": "min",
    }

    def evaluate(_candidate):
        sample_results = []
        for row in videos:
            reference_count = int(row["reference_word_count"])
            errors = sum(
                int(row[name])
                for name in ("word_substitutions", "word_deletions", "word_insertions")
            )
            sample_results.append(
                SampleResult(
                    str(row["sample_id"]),
                    ({"word_error_rate": errors / reference_count} if reference_count else {}),
                    float(row["latency_seconds"]),
                    {"window_count": len(row["windows"])},
                )
            )
        intervals = _frozen_asr_intervals(
            videos, resamples=plan.bootstrap_resamples, seed=plan.seed
        )
        operational = {
            name: float(metrics[name])
            for name in (
                "real_time_factor",
                "total_latency_seconds",
                "cold_model_load_seconds",
                "peak_process_tree_ram_mb",
                "peak_vram_mb",
            )
        }
        return (
            sample_results,
            operational,
            {"word_error_rate": metrics["word_error_rate"]},
            {
                **artifact["parameters"],
                "model_revision": artifact["model_revision"],
                "selection_revision": artifact["selection_revision"],
                "model_path": artifact["model_path"],
                "model_cache_manifest_sha256": artifact[
                    "model_cache_manifest_sha256"
                ],
                "submodels": artifact.get("submodels", []),
                "protocol_checksum": protocol_checksum,
                "frozen_asr_checksum": artifact_checksum,
            },
            intervals,
            {"samples": videos, "ffmpeg_commands": artifact["ffmpeg_commands"]},
        )

    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest_checksum,
        directions=directions_map,
        primary_metric=("word_error_rate", "real_time_factor"),
        required_metrics=tuple(directions_map),
        paired_metrics=(),
        revisions={candidate: str(artifact["selection_revision"]), "ffmpeg": artifact["ffmpeg_version"]},
        decision_files={"audio": audio_decision},
        input_artifacts={
            "manifest": manifest_path,
            "video_protocol_lock": protocol_path,
            "frozen_asr": artifact_path,
        },
        no_mlflow=no_mlflow,
        monitor_resources=False,
        operational_prefix="",
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
        nullable_metrics=("word_error_rate",),
    )


def _frozen_asr_intervals(videos, *, resamples, seed):
    if not resamples or len(videos) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sample = [videos[index] for index in rng.integers(0, len(videos), len(videos))]
        references = sum(int(row["reference_word_count"]) for row in sample)
        if not references:
            continue
        errors = sum(
            int(row[name])
            for row in sample
            for name in ("word_substitutions", "word_deletions", "word_insertions")
        )
        draws.append(errors / references)
    if not draws:
        return {}
    return {
        "word_error_rate": {
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
            "confidence": 0.95,
            "resamples": len(draws),
        }
    }


def _candidates(arguments, protocol_threshold: float):
    if arguments.profile in {"full", "locked"}:
        if arguments.shortlist is None:
            raise ValueError(f"Video {arguments.profile} requires --shortlist")
        decision = load_engineer_decision(
            arguments.shortlist,
            exact=1 if arguments.profile == "locked" else None,
            maximum=1 if arguments.profile == "locked" else 3,
        )
        _require_decision_stage(
            decision,
            "video-development" if arguments.profile == "full" else "video-validation",
        )
        for candidate in decision.selected_candidates:
            parse_candidate(candidate)
        return decision.selected_candidates, {"shortlist": arguments.shortlist}
    if arguments.shortlist is not None:
        raise ValueError("Video smoke/standard candidate grids do not accept --shortlist")
    if arguments.phase == "fixed":
        return FIXED_CANDIDATES, {}
    if arguments.phase == "scene":
        return SCENE_CANDIDATES, {}
    threshold = protocol_threshold
    decisions = {}
    if arguments.phase in {"hybrid", "all"} and arguments.profile != "smoke":
        if arguments.scene_selection is None:
            raise ValueError("Authoritative hybrid video runs require --scene-selection")
        scene_decision = load_engineer_decision(arguments.scene_selection, exact=1)
        _require_decision_stage(scene_decision, "video-development-scene")
        selected = scene_decision.selected_candidates[0]
        parsed = parse_candidate(selected)
        if parsed.strategy != "scene":
            raise ValueError("--scene-selection must select one video-scene candidate")
        threshold = float(parsed.scene_threshold)
        if abs(threshold - protocol_threshold) > 1e-12:
            raise ValueError(
                "Selected scene threshold does not match hybrid_scene_threshold in the protocol lock"
            )
        decisions["scene"] = arguments.scene_selection
    if arguments.phase == "hybrid":
        return hybrid_candidates(threshold), decisions
    return all_candidates(threshold), decisions


def _selected_audio(arguments):
    if arguments.profile == "smoke" and arguments.audio_candidate:
        if arguments.audio_selection:
            raise ValueError("Choose either --audio-candidate or --audio-selection")
        if arguments.audio_candidate not in ASR_PROFILES:
            raise ValueError("Unknown smoke ASR candidate")
        # The synthetic decision fingerprint remains an explicit local input.
        path = PROJECT_ROOT / "experiments/benchmarks/extraction/audio/candidates.yaml"
        return arguments.audio_candidate, path
    if arguments.audio_selection is None:
        raise ValueError("Frozen ASR creation requires --audio-selection")
    decision = load_engineer_decision(arguments.audio_selection, exact=1)
    if arguments.profile != "smoke":
        _require_decision_stage(decision, "audio-validation")
    return decision.selected_candidates[0], arguments.audio_selection


def _selected_image(arguments):
    if arguments.profile == "smoke" and arguments.image_candidate:
        if arguments.document_selection:
            raise ValueError("Choose either --image-candidate or --document-selection")
        return arguments.image_candidate, None
    if arguments.document_selection is None:
        raise ValueError("Visual video execution requires --document-selection")
    decision = load_engineer_decision(arguments.document_selection, exact=1)
    if arguments.profile != "smoke":
        _require_decision_stage(decision, "document-architecture-validation-image")
    return decision.selected_candidates[0], arguments.document_selection


def _validate_manifest(items, profile: str) -> None:
    if not items:
        raise ValueError("Video manifest contains no video samples")
    if profile != "smoke" and len(items) != EXPECTED_COUNTS[profile]:
        raise ValueError(
            f"Video {profile} requires exactly {EXPECTED_COUNTS[profile]} samples"
        )
    for item in items:
        missing = [
            name
            for name in (
                "id",
                "source_path",
                "asset_sha256",
                "duration_seconds",
                "reference_transcript",
                "reference_visual_text",
                "visual_occurrences",
            )
            if name not in item
        ]
        if profile != "smoke":
            missing.extend(
                name
                for name in ("source_license", "source_revision", "document_family")
                if not item.get(name)
            )
        if missing:
            raise ValueError(
                f"Video sample {item.get('id')} lacks fields: {', '.join(sorted(set(missing)))}"
            )
        source = (PROJECT_ROOT / str(item["source_path"])).resolve()
        if not source.is_file() or sha256_file(source) != str(item["asset_sha256"]):
            raise ValueError(f"Video sample {item.get('id')} has an invalid asset")
        item["source_path"] = str(source)
        duration = float(item["duration_seconds"])
        if duration <= 0:
            raise ValueError(f"Video sample {item.get('id')} has invalid duration")
        observed_duration = media_duration(source)
        if abs(observed_duration - duration) > 0.1:
            raise ValueError(
                f"Video sample {item.get('id')} duration differs from the asset by more than 0.1s"
            )
        if not isinstance(item["visual_occurrences"], Sequence) or isinstance(
            item["visual_occurrences"], (str, bytes)
        ) or not item["visual_occurrences"]:
            raise ValueError(f"Video sample {item.get('id')} has malformed occurrences")
        visual_text = item["reference_visual_text"]
        if isinstance(visual_text, str):
            has_visual_text = bool(visual_text.strip())
        elif isinstance(visual_text, Sequence):
            has_visual_text = bool(visual_text) and all(
                isinstance(value, str) and value.strip() for value in visual_text
            )
        else:
            has_visual_text = False
        if not has_visual_text:
            raise ValueError(f"Video sample {item.get('id')} has no verified visual text")
        for occurrence in item["visual_occurrences"]:
            if not isinstance(occurrence, dict) or not str(occurrence.get("text", "")).strip():
                raise ValueError(f"Video sample {item.get('id')} has a malformed occurrence")
            start = float(occurrence.get("start", -1))
            end = float(occurrence.get("end", -1))
            if start < 0 or end < start or end > duration + 1e-6:
                raise ValueError(f"Video sample {item.get('id')} has invalid occurrence bounds")


def _manifest(profile: str) -> Path:
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
    split = {"standard": "development", "full": "validation", "locked": "locked-test"}[
        profile
    ]
    return PROJECT_ROOT / f"data/benchmarks/extraction/video-{split}.json"
def _require_decision_stage(decision, expected_stage: str) -> None:
    summary = json.loads(decision.source_summary.read_text(encoding="utf-8"))
    observed = str(summary.get("plan", {}).get("stage", ""))
    if observed != expected_stage:
        raise ValueError(
            f"{decision.source_summary} must select from {expected_stage}, received {observed}"
        )
