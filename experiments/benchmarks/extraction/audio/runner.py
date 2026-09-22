"""Direct ASR benchmark orchestration."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from edumind.common.artifacts import sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.datasets import (
    assert_no_split_leakage,
    load_manifest,
    require_manifest_split,
)
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.audio.evaluate import (
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    normalize_transcript,
)
from experiments.benchmarks.extraction.audio.protocol import (
    DEFAULT_PROTOCOL_PATH,
    AudioProtocol,
    load_protocol,
)
from experiments.benchmarks.extraction.media import (
    canonical_wav_duration,
    decode_canonical_audio,
    ffmpeg_version,
)
from experiments.benchmarks.preparation.models import load_selected_model_lock

PROFILE_STAGE = {
    "smoke": "audio-smoke",
    "development": "audio-development",
    "validation": "audio-validation",
    "locked": "audio-locked-test",
}


def main(directory: Path) -> int:
    parser = argparse.ArgumentParser(description="Benchmark English audio extraction")
    parser.add_argument(
        "--profile",
        choices=("smoke", "development", "validation", "locked"),
        default="smoke",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reliability-manifest", type=Path)
    parser.add_argument(
        "--shortlist",
        type=Path,
        help="engineer decision selecting finalists or one ASR",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    protocol = load_protocol(arguments.protocol)
    device = arguments.device or protocol.profile(arguments.profile).device
    candidates = _candidates(
        arguments.profile,
        arguments.shortlist,
        protocol,
    )
    result = run(
        arguments.profile,
        candidates,
        manifest_path=arguments.manifest,
        reliability_path=arguments.reliability_manifest,
        device=device,
        no_mlflow=arguments.no_mlflow,
        decision_file=arguments.shortlist,
        protocol_path=arguments.protocol,
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


def run(
    profile: str,
    candidates: tuple[str, ...],
    *,
    manifest_path: Path | None,
    reliability_path: Path | None,
    device: str,
    no_mlflow: bool,
    decision_file: Path | None,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
):
    protocol = load_protocol(protocol_path)
    execution = protocol.profile(profile)
    if execution.hardware_required and device != execution.device:
        raise ValueError(
            f"Authoritative ASR profile requires device {execution.device}"
        )
    speech_path = (manifest_path or _speech_manifest(profile)).resolve()
    controls_path = (reliability_path or _reliability_manifest(profile)).resolve()
    speech_manifest = load_manifest(speech_path)
    reliability_manifest = load_manifest(controls_path)
    speech = [item for item in speech_manifest.samples if item.get("kind") == "audio"]
    split = "locked-test" if profile == "locked" else profile
    _validate_reliability_split_isolation(reliability_manifest.samples)
    require_manifest_split(speech_manifest, profile, split)
    controls = [
        item
        for item in reliability_manifest.samples
        if item.get("kind") == "audio_reliability" and item.get("split") == split
    ]
    _validate_manifest_rows(speech, controls, profile, protocol)
    _validate_candidates(candidates, protocol)
    if profile != "smoke":
        assert_no_split_leakage(_audio_split_manifests(speech_path, split))
    required_models = tuple(
        protocol.candidate(candidate).model_id for candidate in candidates
    )
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=required_models,
    )

    temporary_root = PROJECT_ROOT / "artifacts/benchmarks/asr-canonical"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        canonical_directory = Path(raw_directory)
        canonical_speech = _canonicalize(
            speech, canonical_directory / "speech", protocol
        )
        canonical_controls = _canonicalize(
            controls, canonical_directory / "reliability", protocol
        )
        ffmpeg_identity = ffmpeg_version()
        plan = BenchmarkPlan(
            "extraction",
            PROFILE_STAGE[profile],
            profile,
            speech_manifest.name,
            candidates,
            seed=protocol.meta.seed,
            repetitions=execution.repetitions,
            bootstrap_resamples=execution.bootstrap_resamples,
            warmups=execution.warmups,
            settings={
                "device": device,
                "ffmpeg_version": ffmpeg_identity,
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
                protocol=protocol,
            )
            sample_results = [
                SampleResult(
                    str(row["sample_id"]),
                    {},
                    float(row["quality_latency_seconds"]),
                    {
                        "sample_type": row["sample_type"],
                        "conditions": row["conditions"],
                    },
                )
                for row in output["samples"]
            ]
            metrics = dict(output["metrics"])
            lock_entry = model_lock[protocol.candidate(candidate).model_id]
            parameters = {
                **output["parameters"],
                "model_revision": lock_entry.get("revision", ""),
                "selection_revision": lock_entry.get("selection_revision", ""),
                "model_path": lock_entry.get("model_path", ""),
                "model_cache_manifest_sha256": lock_entry.get(
                    "model_cache_manifest_sha256", ""
                ),
                "data_split": split,
                "ffmpeg_version": ffmpeg_identity,
                "speech_manifest_checksum": speech_manifest.fingerprint,
                "reliability_manifest_checksum": reliability_manifest.fingerprint,
            }
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
            candidate: str(
                model_lock[protocol.candidate(candidate).model_id].get(
                    "selection_revision", ""
                )
            )
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
            protocols={"audio": protocol.meta},
            no_mlflow=no_mlflow,
            monitor_resources=False,
            operational_prefix="",
            paired_comparisons=False,
            candidate_artifact_name="candidate.json",
            nullable_metrics=("timestamp_boundary_mae_seconds",),
            operational_maximums=(
                {"peak_vram_mb": protocol.authoritative_peak_vram_mb}
                if execution.hardware_required
                else None
            ),
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
    protocol,
):
    safe = "".join(character if character.isalnum() else "-" for character in candidate)
    return run_json_worker(
        Path(__file__).with_name("worker.py"),
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
            "protocol": protocol.meta.worker_payload(),
        },
        device=device,
        prefix=f"{safe}-",
        error_label="ASR worker",
        temporary_root=directory,
    )


def _canonicalize(
    samples: Sequence[Mapping[str, object]],
    directory: Path,
    protocol: AudioProtocol,
):
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for index, raw in enumerate(samples):
        item = dict(raw)
        source = PROJECT_ROOT / str(item["source_path"])
        expected = str(item.get("asset_sha256", ""))
        if not source.is_file() or not expected or sha256_file(source) != expected:
            raise ValueError(
                f"Missing or invalid audio asset for {item.get('id')}: {source}"
            )
        destination = directory / f"{index:04d}.wav"
        command = decode_canonical_audio(
            source,
            destination,
            sample_rate_hz=int(protocol.audio["sample_rate_hz"]),
            channels=int(protocol.audio["channels"]),
        )
        duration = canonical_wav_duration(
            destination,
            sample_rate_hz=int(protocol.audio["sample_rate_hz"]),
            channels=int(protocol.audio["channels"]),
            sample_width_bytes=int(protocol.audio["sample_width_bytes"]),
        )
        maximum_duration = float(protocol.audio["maximum_duration_seconds"])
        tolerance = float(protocol.audio["manifest_duration_tolerance_seconds"])
        if duration > maximum_duration + 1e-6:
            raise ValueError(
                f"Audio sample {item['id']} exceeds the {maximum_duration:g}-second limit"
            )
        if abs(duration - float(item["duration_seconds"])) > tolerance:
            raise ValueError(
                f"Audio sample {item['id']} duration differs from its manifest by more than "
                f"{protocol.audio['manifest_duration_tolerance_seconds']}s"
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


def _validate_manifest_rows(
    speech, controls, profile: str, protocol: AudioProtocol
) -> None:
    required_count = protocol.speech_counts[profile]
    if len(speech) != required_count:
        raise ValueError(
            f"ASR {profile} requires exactly {required_count} speech clips"
        )
    authoritative = profile != "smoke"
    observed_conditions: set[str] = set()
    expected_split = {
        "development": "development",
        "validation": "validation",
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
                    "conditions",
                )
                if not item.get(field)
            )
        if missing:
            raise ValueError(
                f"ASR speech sample {item.get('id')} lacks: {', '.join(missing)}"
            )
        if authoritative and item["split"] != expected_split:
            raise ValueError(
                f"ASR speech sample {item['id']} belongs to {item['split']}, not {expected_split}"
            )
        if authoritative:
            raw_conditions = item["conditions"]
            if (
                not isinstance(raw_conditions, list)
                or not raw_conditions
                or not all(isinstance(value, str) and value for value in raw_conditions)
            ):
                raise ValueError(
                    f"ASR speech sample {item['id']} conditions must be a non-empty string list"
                )
            conditions = set(raw_conditions)
            unknown = conditions - protocol.required_conditions
            if unknown:
                raise ValueError(
                    f"ASR speech sample {item['id']} has unknown conditions: "
                    + ", ".join(sorted(unknown))
                )
            for group in protocol.exclusive_condition_groups:
                observed = conditions & group
                if len(observed) != 1:
                    raise ValueError(
                        f"ASR speech sample {item['id']} must contain exactly one of "
                        + ", ".join(sorted(group))
                    )
            observed_conditions.update(conditions)
        duration = float(item["duration_seconds"])
        maximum_duration = float(protocol.audio["maximum_duration_seconds"])
        if duration <= 0 or duration > maximum_duration:
            raise ValueError(
                f"ASR speech sample {item['id']} must be between 0 and "
                f"{maximum_duration:g} seconds"
            )
        _validate_reference_segments(item, duration)
    if authoritative:
        missing_conditions = protocol.required_conditions - observed_conditions
        if missing_conditions:
            raise ValueError(
                "ASR speech split lacks required conditions: "
                + ", ".join(sorted(missing_conditions))
            )
    if not controls:
        raise ValueError("ASR benchmark requires nonspeech reliability controls")
    kinds = {str(item.get("nonspeech_kind")) for item in controls}
    required_kinds = (
        protocol.reliability_categories
        if authoritative
        else protocol.smoke_reliability_categories
    )
    if not required_kinds <= kinds:
        raise ValueError(
            "ASR reliability controls lack: "
            + ", ".join(sorted(required_kinds - kinds))
        )
    for item in controls:
        missing = [
            field
            for field in (
                "id",
                "source_path",
                "asset_sha256",
                "nonspeech_kind",
                "split",
            )
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
            raise ValueError(
                f"Nonspeech control {item.get('id')} must have an empty reference"
            )
        duration = float(item.get("duration_seconds", 0))
        if duration <= 0 or duration > float(
            protocol.audio["maximum_duration_seconds"]
        ):
            raise ValueError(f"Nonspeech control {item.get('id')} has invalid duration")


def _validate_reliability_split_isolation(
    samples: Sequence[Mapping[str, object]],
) -> None:
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
    if (
        not isinstance(segments, Sequence)
        or isinstance(segments, (str, bytes))
        or not segments
    ):
        raise ValueError(
            f"ASR speech sample {item['id']} lacks timed reference segments"
        )
    previous_end = 0.0
    segment_texts: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or not str(segment.get("text", "")).strip():
            raise ValueError(
                f"ASR speech sample {item['id']} has a malformed reference segment"
            )
        segment_texts.append(str(segment["text"]))
        start, end = float(segment.get("start", -1)), float(segment.get("end", -1))
        if start < previous_end or end <= start or end > duration + 1e-6:
            raise ValueError(
                f"ASR speech sample {item['id']} has invalid segment boundaries"
            )
        previous_end = end
    if normalize_transcript(" ".join(segment_texts)) != normalize_transcript(
        str(item.get("reference", ""))
    ):
        raise ValueError(
            f"ASR speech sample {item['id']} reference does not match its timed segments"
        )


def _validate_candidates(candidates, protocol: AudioProtocol) -> None:
    if len(set(candidates)) != len(candidates):
        raise ValueError("ASR candidate list contains duplicates")
    unknown = sorted(set(candidates) - set(protocol.candidates))
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


def _candidates(
    profile: str,
    shortlist: Path | None,
    protocol: AudioProtocol,
) -> tuple[str, ...]:
    if profile in {"smoke", "development"}:
        if shortlist is not None:
            raise ValueError(
                f"ASR {profile} runs the complete configured candidate list"
            )
        return tuple(protocol.candidates)
    if shortlist is None:
        raise ValueError(f"ASR {profile} requires --shortlist DECISION_JSON")
    return load_engineer_decision(
        shortlist,
        exact=1 if profile == "locked" else None,
        maximum=1 if profile == "locked" else protocol.maximum_finalists,
        expected_source=(
            "extraction",
            "audio-development" if profile == "validation" else "audio-validation",
            "development" if profile == "validation" else "validation",
        ),
    ).selected_candidates


def _speech_manifest(profile: str) -> Path:
    name = {
        "smoke": "smoke.json",
        "development": "audio-development.json",
        "validation": "audio-validation.json",
        "locked": "audio-locked-test.json",
    }[profile]
    return PROJECT_ROOT / "data/benchmarks/extraction" / name


def _reliability_manifest(profile: str) -> Path:
    name = (
        "audio-reliability-smoke.json"
        if profile == "smoke"
        else "audio-reliability.json"
    )
    return PROJECT_ROOT / "data/benchmarks/extraction" / name
