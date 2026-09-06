"""Direct ASR benchmark orchestration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import load_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.datasets import assert_no_split_leakage, load_manifest
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.audio.adapters import ASR_PROFILES
from experiments.benchmarks.extraction.audio.evaluate import (
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    normalize_transcript,
)
from experiments.benchmarks.preparation.models import load_selected_model_lock

SPEECH_COUNTS = {"standard": 54, "full": 18, "locked": 18}
PROFILE_STAGE = {
    "smoke": "audio-smoke",
    "standard": "audio-development",
    "full": "audio-validation",
    "locked": "audio-locked-test",
}
RELIABILITY_KINDS = {
    "silence",
    "music_without_lyrics",
    "background_noise",
    "environmental_sound",
}
REQUIRED_SPEECH_CONDITIONS = {"clean", "noisy", "accented", "multi_speaker"}


def main(directory: Path) -> int:
    parser = argparse.ArgumentParser(description="Benchmark English audio extraction")
    parser.add_argument(
        "--profile", choices=("smoke", "standard", "full", "locked"), default="smoke"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reliability-manifest", type=Path)
    parser.add_argument(
        "--shortlist", type=Path, help="engineer decision selecting finalists or one ASR"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    candidates = _candidates(directory / "candidates.yaml", arguments.profile, arguments.shortlist)
    result = run(
        arguments.profile,
        candidates,
        manifest_path=arguments.manifest,
        reliability_path=arguments.reliability_manifest,
        device=arguments.device,
        no_mlflow=arguments.no_mlflow,
        decision_file=arguments.shortlist,
    )
    print(
        json.dumps(
            {"run_id": result.run_id, "complete": result.complete, "artifacts": str(result.artifact_directory)},
            indent=2,
        )
    )
    return 0 if result.complete else 2


def run(
    profile: str,
    candidates: tuple[str, ...],
    *,
    manifest_path: Path | None,
    reliability_path: Path | None,
    device: str,
    no_mlflow: bool,
    decision_file: Path | None,
):
    speech_path = (manifest_path or _speech_manifest(profile)).resolve()
    controls_path = (reliability_path or _reliability_manifest(profile)).resolve()
    speech_manifest = load_manifest(speech_path)
    reliability_manifest = load_manifest(controls_path)
    speech = [item for item in speech_manifest.samples if item.get("kind") == "audio"]
    split = {"standard": "development", "full": "validation", "locked": "locked-test"}.get(
        profile, "smoke"
    )
    _validate_reliability_split_isolation(reliability_manifest.samples)
    if speech_manifest.split != split:
        raise ValueError(
            f"ASR {profile} requires the {split} speech split, received {speech_manifest.split}"
        )
    controls = [
        item
        for item in reliability_manifest.samples
        if item.get("kind") == "audio_reliability" and item.get("split") == split
    ]
    _validate_manifest_rows(speech, controls, profile)
    _validate_candidates(candidates)
    if profile != "smoke":
        assert_no_split_leakage(_audio_split_manifests(speech_path, split))
    required_models = tuple(ASR_PROFILES[candidate].model for candidate in candidates)
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=required_models,
    )

    temporary_root = PROJECT_ROOT / "artifacts/benchmarks/asr-canonical"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        canonical_directory = Path(raw_directory)
        canonical_speech = _canonicalize(speech, canonical_directory / "speech")
        canonical_controls = _canonicalize(controls, canonical_directory / "reliability")
        ffmpeg_version = _ffmpeg_version()
        repetitions = 1 if profile == "smoke" else 3
        resamples = 0 if profile == "smoke" else 10_000
        plan = BenchmarkPlan(
            "extraction",
            PROFILE_STAGE[profile],
            profile,
            speech_manifest.name,
            candidates,
            repetitions=repetitions,
            bootstrap_resamples=resamples,
            settings={
                "device": device,
                "sample_rate_hz": 16_000,
                "channels": 1,
                "encoding": "PCM signed 16-bit little-endian",
                "maximum_duration_seconds": 30,
                "ffmpeg_version": ffmpeg_version,
                "ffmpeg_commands": {
                    str(item["id"]): item["ffmpeg_command"]
                    for item in [*canonical_speech, *canonical_controls]
                },
                "speech_manifest_checksum": speech_manifest.fingerprint,
                "reliability_manifest_checksum": reliability_manifest.fingerprint,
                "canonical_checksums": {
                    str(item["id"]): item["canonical_sha256"]
                    for item in [*canonical_speech, *canonical_controls]
                },
            },
        )

        def evaluate(candidate: str):
            output = _run_worker(
                candidate,
                model_lock,
                canonical_speech,
                canonical_controls,
                device=device,
                warmups=plan.warmups,
                repetitions=plan.repetitions,
                bootstrap_resamples=plan.bootstrap_resamples,
                seed=plan.seed,
                directory=canonical_directory,
            )
            sample_results = [
                SampleResult(
                    str(row["sample_id"]),
                    {},
                    float(row["quality_latency_seconds"]),
                    {
                        "sample_type": row["sample_type"],
                        "condition": row["condition"],
                    },
                )
                for row in output["samples"]
            ]
            metrics = dict(output["metrics"])
            lock_entry = model_lock[ASR_PROFILES[candidate].model]
            parameters = {
                **output["parameters"],
                "model_revision": lock_entry.get("revision", ""),
                "selection_revision": lock_entry.get("selection_revision", ""),
                "model_path": lock_entry.get("model_path", ""),
                "data_split": split,
                "ffmpeg_version": ffmpeg_version,
                "canonical_audio": "mono 16 kHz PCM WAV",
                "maximum_duration_seconds": 30,
                "speech_manifest_checksum": speech_manifest.fingerprint,
                "reliability_manifest_checksum": reliability_manifest.fingerprint,
            }
            if candidate == "qwen3-asr-1.7b-aligned":
                parameters["aligner_revision"] = next(
                    str(item.get("revision", ""))
                    for item in lock_entry.get("submodels", [])
                    if item.get("role") == "forced-aligner"
                )
                parameters["aligner_model_path"] = next(
                    str(item.get("model_path", ""))
                    for item in lock_entry.get("submodels", [])
                    if item.get("role") == "forced-aligner"
                )
            operational = {
                name: float(metrics.pop(name))
                for name in (
                    "real_time_factor",
                    "p50_warm_clip_latency_seconds",
                    "p95_warm_clip_latency_seconds",
                    "cold_model_load_seconds",
                    "peak_process_tree_ram_mb",
                    "peak_vram_mb",
                )
            }
            return (
                sample_results,
                operational,
                metrics,
                parameters,
                output["intervals"],
                {"samples": output["samples"], "timings": output["timings"]},
            )

        revisions = {
            candidate: str(model_lock[ASR_PROFILES[candidate].model].get("selection_revision", ""))
            for candidate in candidates
        }
        return run_benchmark(
            plan,
            evaluate,
            dataset_checksum=stable_hash(
                {
                    "speech": speech_manifest.fingerprint,
                    "reliability": reliability_manifest.fingerprint,
                    "canonical": plan.settings["canonical_checksums"],
                }
            ),
            directions=METRIC_DIRECTIONS,
            primary_metric=PRIMARY_METRICS,
            required_metrics=tuple(METRIC_DIRECTIONS),
            paired_metrics=(),
            revisions=revisions,
            decision_files={"shortlist": decision_file} if decision_file else None,
            input_artifacts={"speech": speech_path, "reliability": controls_path},
            no_mlflow=no_mlflow,
            monitor_resources=False,
            operational_prefix="",
            paired_comparisons=False,
            candidate_artifact_name="candidate.json",
            nullable_metrics=("timestamp_boundary_mae_seconds",),
        )


def _run_worker(
    candidate,
    model_lock,
    speech,
    controls,
    *,
    device,
    warmups,
    repetitions,
    bootstrap_resamples,
    seed,
    directory,
):
    safe = "".join(character if character.isalnum() else "-" for character in candidate)
    payload_path = directory / f"{safe}-input.json"
    output_path = directory / f"{safe}-output.json"
    atomic_write_json(
        payload_path,
        {
            "candidate": candidate,
            "model_lock": model_lock,
            "speech": speech,
            "reliability": controls,
            "device": device,
            "warmups": warmups,
            "repetitions": repetitions,
            "bootstrap_resamples": bootstrap_resamples,
            "seed": seed,
        },
    )
    environment = _worker_environment(device)
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("worker.py")), str(payload_path), str(output_path)],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "no worker output").strip()
            raise RuntimeError(f"ASR worker failed: {detail[-4000:]}")
        if not output_path.is_file():
            raise RuntimeError("ASR worker completed without a result file")
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        payload_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def _canonicalize(samples: Sequence[Mapping[str, object]], directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for index, raw in enumerate(samples):
        item = dict(raw)
        source = PROJECT_ROOT / str(item["source_path"])
        expected = str(item.get("asset_sha256", ""))
        if not source.is_file() or not expected or sha256_file(source) != expected:
            raise ValueError(f"Missing or invalid audio asset for {item.get('id')}: {source}")
        destination = directory / f"{index:04d}.wav"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        subprocess.run(command, check=True)
        duration = _wav_duration(destination)
        if duration > 30.0 + 1e-6:
            raise ValueError(f"Audio sample {item['id']} exceeds the 30-second limit")
        if abs(duration - float(item["duration_seconds"])) > 0.1:
            raise ValueError(
                f"Audio sample {item['id']} duration differs from its manifest by more than 0.1s"
            )
        item.update(
            {
                "canonical_path": str(destination.resolve()),
                "canonical_sha256": sha256_file(destination),
                "ffmpeg_command": command,
                "duration_seconds": duration,
            }
        )
        result.append(item)
    return result


def _validate_manifest_rows(speech, controls, profile: str) -> None:
    required_count = SPEECH_COUNTS.get(profile)
    if required_count is not None and len(speech) != required_count:
        raise ValueError(f"ASR {profile} requires exactly {required_count} speech clips")
    if profile == "smoke" and len(speech) < 2:
        raise ValueError("ASR smoke requires at least two speech clips")
    authoritative = profile != "smoke"
    observed_conditions: set[str] = set()
    expected_split = {
        "standard": "development",
        "full": "validation",
        "locked": "locked-test",
    }.get(profile, "smoke")
    for item in speech:
        missing = [
            field
            for field in (
                "id",
                "source_path",
                "asset_sha256",
                "reference",
                "duration_seconds",
                "reference_segments",
            )
            if not item.get(field)
        ]
        if authoritative:
            missing.extend(
                field
                for field in (
                    "source_license",
                    "source_revision",
                    "split",
                    "document_family",
                    "condition",
                )
                if not item.get(field)
            )
        if missing:
            raise ValueError(f"ASR speech sample {item.get('id')} lacks: {', '.join(missing)}")
        if authoritative and item["split"] != expected_split:
            raise ValueError(
                f"ASR speech sample {item['id']} belongs to {item['split']}, not {expected_split}"
            )
        if authoritative:
            observed_conditions.add(str(item["condition"]))
        duration = float(item["duration_seconds"])
        if duration <= 0 or duration > 30:
            raise ValueError(f"ASR speech sample {item['id']} must be between 0 and 30 seconds")
        _validate_reference_segments(item, duration)
    if authoritative:
        missing_conditions = REQUIRED_SPEECH_CONDITIONS - observed_conditions
        if missing_conditions:
            raise ValueError(
                "ASR speech split lacks required conditions: "
                + ", ".join(sorted(missing_conditions))
            )
    if not controls:
        raise ValueError("ASR benchmark requires nonspeech reliability controls")
    kinds = {str(item.get("nonspeech_kind")) for item in controls}
    required_kinds = RELIABILITY_KINDS if authoritative else {"silence", "background_noise"}
    if not required_kinds <= kinds:
        raise ValueError(
            "ASR reliability controls lack: " + ", ".join(sorted(required_kinds - kinds))
        )
    for item in controls:
        missing = [
            field
            for field in ("id", "source_path", "asset_sha256", "nonspeech_kind", "split")
            if not item.get(field)
        ]
        if authoritative:
            missing.extend(
                field
                for field in ("source_license", "source_revision")
                if not item.get(field)
            )
        if missing:
            raise ValueError(
                f"Nonspeech control {item.get('id')} lacks: {', '.join(missing)}"
            )
        if item.get("reference") not in {"", None}:
            raise ValueError(f"Nonspeech control {item.get('id')} must have an empty reference")
        duration = float(item.get("duration_seconds", 0))
        if duration <= 0 or duration > 30:
            raise ValueError(f"Nonspeech control {item.get('id')} has invalid duration")


def _validate_reliability_split_isolation(samples: Sequence[Mapping[str, object]]) -> None:
    seen_ids: dict[str, str] = {}
    seen_assets: dict[str, tuple[str, str]] = {}
    for item in samples:
        if item.get("kind") != "audio_reliability":
            continue
        sample_id = str(item.get("id", ""))
        split = str(item.get("split", ""))
        checksum = str(item.get("asset_sha256", ""))
        if not sample_id or not split or not checksum:
            raise ValueError(
                f"Nonspeech control {item.get('id')} lacks ID, split, or asset checksum"
            )
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate ASR reliability sample ID {sample_id!r} occurs in "
                f"{seen_ids[sample_id]} and {split}"
            )
        previous = seen_assets.get(checksum)
        if previous is not None:
            previous_id, previous_split = previous
            raise ValueError(
                "Duplicate ASR reliability asset checksum "
                f"{checksum!r} occurs in {previous_id}/{previous_split} and "
                f"{sample_id}/{split}"
            )
        seen_ids[sample_id] = split
        seen_assets[checksum] = (sample_id, split)


def _validate_reference_segments(item: Mapping[str, object], duration: float) -> None:
    segments = item.get("reference_segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        raise ValueError(f"ASR speech sample {item['id']} lacks timed reference segments")
    previous_end = 0.0
    segment_texts: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or not str(segment.get("text", "")).strip():
            raise ValueError(f"ASR speech sample {item['id']} has a malformed reference segment")
        segment_texts.append(str(segment["text"]))
        start, end = float(segment.get("start", -1)), float(segment.get("end", -1))
        if start < previous_end or end <= start or end > duration + 1e-6:
            raise ValueError(f"ASR speech sample {item['id']} has invalid segment boundaries")
        previous_end = end
    if normalize_transcript(" ".join(segment_texts)) != normalize_transcript(
        str(item.get("reference", ""))
    ):
        raise ValueError(
            f"ASR speech sample {item['id']} reference does not match its timed segments"
        )


def _validate_candidates(candidates) -> None:
    if len(set(candidates)) != len(candidates):
        raise ValueError("ASR candidate list contains duplicates")
    unknown = sorted(set(candidates) - set(ASR_PROFILES))
    if unknown:
        raise ValueError("Unknown ASR candidates: " + ", ".join(unknown))


def _audio_split_manifests(current_path: Path, current_split: str):
    names = {
        "development": "audio-development.json",
        "validation": "audio-validation.json",
        "locked-test": "audio-locked-test.json",
    }
    paths = {
        split: current_path if split == current_split else current_path.parent / name
        for split, name in names.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "Authoritative ASR runs require all three frozen split manifests; missing: "
            + ", ".join(missing)
        )
    return tuple(load_manifest(paths[split]) for split in names)


def _candidates(path: Path, profile: str, shortlist: Path | None) -> tuple[str, ...]:
    if profile in {"smoke", "standard"}:
        if shortlist is not None:
            raise ValueError(f"ASR {profile} runs the complete configured candidate list")
        return load_candidates(path, profile)
    if shortlist is None:
        raise ValueError(f"ASR {profile} requires --shortlist DECISION_JSON")
    return load_engineer_decision(shortlist, exact=1 if profile == "locked" else None).selected_candidates


def _speech_manifest(profile: str) -> Path:
    name = {
        "smoke": "smoke.json",
        "standard": "audio-development.json",
        "full": "audio-validation.json",
        "locked": "audio-locked-test.json",
    }[profile]
    return PROJECT_ROOT / "data/benchmarks/extraction" / name


def _reliability_manifest(profile: str) -> Path:
    name = "audio-reliability-smoke.json" if profile == "smoke" else "audio-reliability.json"
    return PROJECT_ROOT / "data/benchmarks/extraction" / name


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getframerate() != 16_000 or audio.getsampwidth() != 2:
            raise ValueError(f"FFmpeg did not produce canonical mono 16 kHz PCM audio: {path}")
        return audio.getnframes() / audio.getframerate()


def _ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    )
    return completed.stdout.splitlines()[0].strip()


def _worker_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NVIDIA_VISIBLE_DEVICES"] = "none"
    return environment
