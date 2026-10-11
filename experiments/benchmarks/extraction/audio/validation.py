"""ASR annotation, canonical-media, corpus, and control validation."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.data_validation import (
    PROFILES,
    manifest_path,
    verify_report,
    write_report,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.extraction.audio.evaluate import normalize_transcript
from experiments.benchmarks.extraction.audio.protocol import AudioProtocol
from experiments.benchmarks.extraction.media import (
    canonical_wav_duration,
    decode_canonical_audio,
)


def reliability_manifest(profile):
    name = (
        "audio-reliability-smoke.json"
        if profile == "smoke"
        else "audio-reliability.json"
    )
    return PROJECT_ROOT / "data/benchmarks/extraction" / name


def requirements(protocol, profile, controls_path):
    return {
        "normalization": "NFC-prose-v1",
        "audio": dict(protocol.audio),
        "speech_count": protocol.speech_counts[profile],
        "conditions": sorted(protocol.required_conditions)
        if profile != "smoke"
        else [],
        "control_categories": sorted(
            protocol.reliability_categories
            if profile != "smoke"
            else protocol.smoke_reliability_categories
        ),
        "control_path": str(controls_path.resolve()),
    }


def validate(path, profile, protocol, controls_path):
    speech = [
        item for item in load_manifest(path).samples if item.get("kind") == "audio"
    ]
    split = "locked-test" if profile == "locked" else profile
    all_controls = load_manifest(controls_path).samples
    validate_reliability_split_isolation(all_controls)
    controls = [
        item
        for item in all_controls
        if item.get("kind") == "audio_reliability" and item.get("split") == split
    ]
    validate_manifest_rows(speech, controls, profile, protocol)
    with tempfile.TemporaryDirectory(prefix="asr-data-validation-") as raw:
        prepared = canonicalize([*speech, *controls], Path(raw), protocol)
        metadata = {
            str(item["id"]): {
                "actual_duration_seconds": item["duration_seconds"],
                "canonical_sha256": item["canonical_sha256"],
            }
            for item in prepared
        }
    return {
        "samples": metadata,
        "speech_count": len(speech),
        "control_count": len(controls),
    }


def _report(
    prepare, path, profile, protocol, *, controls_path=None, inventory=None, **options
):
    controls = Path(controls_path or reliability_manifest(profile)).resolve()
    inventory = tuple(
        inventory
        or (
            (path,)
            if profile == "smoke"
            else tuple(
                path if name == profile else manifest_path("audio", name)
                for name in PROFILES[1:]
            )
        )
    )
    inventory = tuple(dict.fromkeys((*inventory, controls)))
    arguments = (
        "audio",
        profile,
        path,
        requirements(protocol, profile, controls),
        Path(__file__),
    )
    if prepare:
        return write_report(
            *arguments,
            lambda source, phase: validate(source, phase, protocol, controls),
            inventory=inventory,
            **options,
        )
    return verify_report(*arguments, inventory=inventory, **options)


def prepare_report(path, profile, protocol, **options):
    return _report(True, path, profile, protocol, **options)


def verified_inputs(path, profile, protocol, **options):
    return _report(False, path, profile, protocol, **options)


def canonicalize(
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


def validate_manifest_rows(
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
        if not isinstance(item.get("reference"), str):
            raise ValueError(
                "Speech reference must be explicit text, including reviewed empty text"
            )
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
            if field not in item or item[field] is None
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
        if not math.isfinite(duration) or duration <= 0 or duration > maximum_duration:
            raise ValueError(
                f"ASR speech sample {item['id']} must be between 0 and "
                f"{maximum_duration:g} seconds"
            )
        validate_reference_segments(item, duration)
    if authoritative and not any(
        normalize_transcript(str(item["reference"])) for item in speech
    ):
        raise ValueError(
            "Authoritative ASR speech corpus must contain normalized words and timed segments"
        )
    if authoritative:
        missing_conditions = protocol.required_conditions - observed_conditions
        if missing_conditions:
            raise ValueError(
                "ASR speech split lacks required conditions: "
                + ", ".join(sorted(missing_conditions))
            )
    if controls is None:
        return
    if not controls:
        raise ValueError("ASR benchmark requires nonspeech reliability controls")
    kinds = {str(item.get("nonspeech_kind")) for item in controls}
    if kinds - protocol.reliability_categories:
        raise ValueError("Unknown reviewed nonspeech category")
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
        if (
            "reference" not in item
            or not isinstance(item["reference"], str)
            or normalize_transcript(item["reference"])
        ):
            raise ValueError(
                f"Nonspeech control {item.get('id')} must have an empty reference"
            )
        duration = float(item.get("duration_seconds", 0))
        if (
            not math.isfinite(duration)
            or duration <= 0
            or duration > float(protocol.audio["maximum_duration_seconds"])
        ):
            raise ValueError(f"Nonspeech control {item.get('id')} has invalid duration")
    from experiments.benchmarks.extraction.scoring import source_id

    ids = [str(item["id"]) for item in (*speech, *controls)]
    if len(set(ids)) != len(ids):
        raise ValueError("ASR speech and nonspeech request IDs must be unique together")
    if {source_id(item) for item in speech} & {source_id(item) for item in controls}:
        raise ValueError(
            "ASR separately resampled speech and control sources must be independent"
        )


def validate_reliability_split_isolation(
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


def validate_reference_segments(item: Mapping[str, object], duration: float) -> None:
    segments = item.get("reference_segments")
    if (
        not isinstance(segments, Sequence)
        or isinstance(segments, (str, bytes))
        or (not segments and bool(normalize_transcript(str(item.get("reference", "")))))
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
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < previous_end
            or end <= start
            or end > duration + 1e-6
        ):
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
