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
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.audio.adapters import profiles as audio_profiles
from experiments.benchmarks.extraction.audio.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_AUDIO_PROTOCOL_PATH,
    AudioProtocol,
    load_protocol as load_audio_protocol,
)
from experiments.benchmarks.extraction.document.profiles import (
    lock_paths,
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_DOCUMENT_PROTOCOL_PATH,
    DocumentProtocol,
    load_protocol as load_document_protocol,
)
from experiments.benchmarks.extraction.document.runner import validate_prepared_components
from experiments.benchmarks.extraction.media import ffmpeg_version, media_duration
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.extraction.video.candidates import (
    all_candidates,
    fixed_candidates,
    hybrid_candidates,
    parse_candidate,
    scene_candidates,
)
from experiments.benchmarks.extraction.video.frozen_asr import (
    create_frozen_asr_artifact,
    load_frozen_asr_artifact,
)
from experiments.benchmarks.extraction.video.metrics import METRIC_DIRECTIONS
from experiments.benchmarks.extraction.video.protocol import (
    DEFAULT_PROTOCOL_PATH,
    VideoProtocol,
    load_protocol,
)
from experiments.benchmarks.preparation.models import load_selected_model_lock

PROFILE_STAGE = {
    "smoke": "video-smoke",
    "development": "video-development",
    "validation": "video-validation",
    "locked": "video-locked-test",
}
def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark visual video extraction")
    parser.add_argument(
        "--profile",
        choices=("smoke", "development", "validation", "locked"),
        default="smoke",
    )
    parser.add_argument(
        "--phase", choices=("frozen-asr", "fixed", "scene", "hybrid", "all"), default="all"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--audio-protocol", type=Path, default=DEFAULT_AUDIO_PROTOCOL_PATH
    )
    parser.add_argument(
        "--document-protocol", type=Path, default=DEFAULT_DOCUMENT_PROTOCOL_PATH
    )
    parser.add_argument("--frozen-asr", type=Path)
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--document-selection", type=Path)
    parser.add_argument("--audio-selection", type=Path)
    parser.add_argument("--image-candidate", help="Smoke-only selected document profile")
    parser.add_argument("--audio-candidate", help="Smoke-only selected ASR profile")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()

    manifest_path = (arguments.manifest or _manifest(arguments.profile)).resolve()
    manifest = load_manifest(manifest_path)
    expected_split = {
        "smoke": "smoke",
        "development": "development",
        "validation": "validation",
        "locked": "locked-test",
    }[arguments.profile]
    require_manifest_split(manifest, arguments.profile, expected_split)
    items = [dict(item) for item in manifest.samples if item.get("kind") == "video"]
    protocol = load_protocol(arguments.protocol)
    audio_protocol = load_audio_protocol(arguments.audio_protocol)
    execution = (
        audio_protocol.profile(arguments.profile)
        if arguments.phase == "frozen-asr"
        else protocol.profile(arguments.profile)
    )
    device = arguments.device or execution.device
    if execution.hardware_required and device != execution.device:
        parser.error(
            f"{arguments.phase} {arguments.profile} requires --device {execution.device}"
        )
    _validate_manifest(items, arguments.profile, protocol)
    ffmpeg_identity = ffmpeg_version()
    if arguments.phase == "frozen-asr":
        if arguments.frozen_asr is None:
            raise ValueError("The frozen-asr phase requires --frozen-asr OUTPUT_JSON")
        audio_candidate, decision_path = _selected_audio(arguments, audio_protocol)
        artifact_path = create_frozen_asr_artifact(
            arguments.frozen_asr,
            items=items,
            manifest_checksum=manifest.fingerprint,
            protocol=protocol,
            audio_protocol=audio_protocol,
            audio_candidate=audio_candidate,
            audio_decision_path=decision_path,
            device=device,
            ffmpeg_version=ffmpeg_identity,
            profile_name=arguments.profile,
        )
        asr_result = _record_frozen_asr(
            artifact_path,
            profile=arguments.profile,
            manifest_path=manifest_path,
            manifest_name=manifest.name,
            manifest_checksum=manifest.fingerprint,
            protocol=protocol,
            audio_protocol=audio_protocol,
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
        protocol_checksum=protocol.meta.checksum,
        audio_protocol_checksum=audio_protocol.meta.checksum,
        timestamp_tolerance_seconds=protocol.manifest_duration_tolerance_seconds,
        sample_ids=[str(item["id"]) for item in items],
    )
    image_candidate, image_decision = _selected_image(arguments)
    document_protocol = load_document_protocol(arguments.document_protocol)
    candidates, candidate_decisions = _candidates(arguments, protocol)
    result = run_visual_benchmark(
        arguments.profile,
        candidates,
        items=items,
        manifest_path=manifest_path,
        manifest_name=manifest.name,
        manifest_checksum=manifest.fingerprint,
        protocol=protocol,
        audio_protocol=audio_protocol,
        document_protocol=document_protocol,
        frozen_asr_path=arguments.frozen_asr.resolve(),
        frozen_asr=frozen,
        frozen_asr_checksum=frozen_checksum,
        image_candidate=image_candidate,
        image_decision=image_decision,
        decision_files=candidate_decisions,
        device=device,
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
    protocol: VideoProtocol,
    audio_protocol: AudioProtocol,
    document_protocol: DocumentProtocol,
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
    execution = protocol.profile(profile)
    document_profile = parse_document_profile(image_candidate)
    document_protocol.validate_candidate_factors(document_profile.factors)
    engine = document_profile.runtime_engine
    if device not in document_protocol.backend_devices[engine]:
        raise ValueError(
            f"{engine} cannot run on {device}; allowed devices: "
            + ", ".join(document_protocol.backend_devices[engine])
        )
    image_options = {
        **document_protocol.parser_options(engine),
        **document_profile.options,
    }
    lock_name = document_profile.lock_candidate
    lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(lock_name,),
    )
    entry = lock[lock_name]
    validate_prepared_components(image_candidate, entry, document_protocol)
    image_options.update(lock_paths(entry))
    stage = PROFILE_STAGE[profile]
    strategies = {
        parse_candidate(candidate, protocol).strategy for candidate in candidates
    }
    if profile == "development" and len(strategies) == 1:
        stage = f"{stage}-{next(iter(strategies))}"
    plan = BenchmarkPlan(
        "extraction",
        stage,
        profile,
        manifest_name,
        tuple(candidates),
        seed=protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
        settings={
            "device": device,
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
            image_candidate=image_candidate,
            image_revision=str(entry.get("revision", "")),
            image_options=image_options,
            device=device,
            warmups=plan.warmups,
            repetitions=plan.repetitions,
            bootstrap_resamples=plan.bootstrap_resamples,
            seed=plan.seed,
            protocol=protocol.meta.worker_payload(),
            document_protocol=document_protocol.meta.worker_payload(),
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
            "protocol_checksum": protocol.meta.checksum,
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
            "frozen_asr": frozen_asr_path,
        },
        protocols={
            "video": protocol.meta,
            "audio": audio_protocol.meta,
            "document": document_protocol.meta,
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
    protocol: VideoProtocol,
    audio_protocol: AudioProtocol,
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
        seed=protocol.meta.seed,
        repetitions=1,
        bootstrap_resamples=protocol.profile(profile).bootstrap_resamples,
        warmups=protocol.profile(profile).warmups,
        settings={"frozen_asr_checksum": artifact_checksum},
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
            videos,
            resamples=plan.bootstrap_resamples,
            seed=plan.seed,
            confidence=protocol.confidence_level,
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
                "protocol_checksum": protocol.meta.checksum,
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
            "frozen_asr": artifact_path,
        },
        protocols={"video": protocol.meta, "audio": audio_protocol.meta},
        no_mlflow=no_mlflow,
        monitor_resources=False,
        operational_prefix="",
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
        nullable_metrics=("word_error_rate",),
    )


def _frozen_asr_intervals(videos, *, resamples, seed, confidence):
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
    alpha = (1.0 - confidence) / 2.0
    return {
        "word_error_rate": {
            "lower": float(np.quantile(draws, alpha)),
            "upper": float(np.quantile(draws, 1.0 - alpha)),
            "confidence": confidence,
            "resamples": len(draws),
        }
    }


def _candidates(arguments, protocol: VideoProtocol):
    if arguments.profile in {"validation", "locked"}:
        if arguments.shortlist is None:
            raise ValueError(f"Video {arguments.profile} requires --shortlist")
        decision = load_engineer_decision(
            arguments.shortlist,
            exact=1 if arguments.profile == "locked" else None,
            maximum=(
                1
                if arguments.profile == "locked"
                else protocol.maximum_finalists
            ),
            expected_source=(
                "extraction",
                "video-development"
                if arguments.profile == "validation"
                else "video-validation",
                "development" if arguments.profile == "validation" else "validation",
            ),
        )
        for candidate in decision.selected_candidates:
            parse_candidate(candidate, protocol)
        return decision.selected_candidates, {"shortlist": arguments.shortlist}
    if arguments.shortlist is not None:
        raise ValueError(
            "Video smoke/development candidate grids do not accept --shortlist"
        )
    if arguments.phase == "fixed":
        return fixed_candidates(protocol), {}
    if arguments.phase == "scene":
        return scene_candidates(protocol), {}
    threshold = protocol.hybrid_threshold(arguments.profile)
    if arguments.phase == "hybrid":
        return hybrid_candidates(protocol, threshold), {}
    return all_candidates(protocol, threshold), {}


def _selected_audio(arguments, protocol: AudioProtocol):
    candidates = audio_profiles(protocol)
    if arguments.profile == "smoke" and arguments.audio_candidate:
        if arguments.audio_selection:
            raise ValueError("Choose either --audio-candidate or --audio-selection")
        if arguments.audio_candidate not in candidates:
            raise ValueError("Unknown smoke ASR candidate")
        # The synthetic decision fingerprint remains an explicit local input.
        path = PROJECT_ROOT / "experiments/benchmarks/extraction/audio/candidates.yaml"
        return arguments.audio_candidate, path
    if arguments.audio_selection is None:
        raise ValueError("Frozen ASR creation requires --audio-selection")
    decision = load_engineer_decision(
        arguments.audio_selection,
        exact=1,
        expected_source=("extraction", "audio-validation", "validation")
        if arguments.profile != "smoke"
        else None,
    )
    return decision.selected_candidates[0], arguments.audio_selection


def _selected_image(arguments):
    if arguments.profile == "smoke" and arguments.image_candidate:
        if arguments.document_selection:
            raise ValueError("Choose either --image-candidate or --document-selection")
        return arguments.image_candidate, None
    if arguments.document_selection is None:
        raise ValueError("Visual video execution requires --document-selection")
    decision = load_engineer_decision(
        arguments.document_selection,
        exact=1,
        expected_source=(
            "extraction",
            "document-architecture-validation-image",
            "validation",
        )
        if arguments.profile != "smoke"
        else None,
    )
    return decision.selected_candidates[0], arguments.document_selection


def _validate_manifest(items, profile: str, protocol: VideoProtocol) -> None:
    if not items:
        raise ValueError("Video manifest contains no video samples")
    expected = protocol.video_counts[profile]
    if len(items) != expected:
        raise ValueError(
            f"Video {profile} requires exactly {expected} samples"
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
        if (
            abs(observed_duration - duration)
            > protocol.manifest_duration_tolerance_seconds
        ):
            raise ValueError(
                f"Video sample {item.get('id')} duration differs from the asset by more than "
                f"{protocol.manifest_duration_tolerance_seconds}s"
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
    split = "locked-test" if profile == "locked" else profile
    return PROJECT_ROOT / f"data/benchmarks/extraction/video-{split}.json"
