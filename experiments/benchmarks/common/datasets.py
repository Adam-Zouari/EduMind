"""Manifest loading, checksums, provenance, and leakage validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .contracts import DatasetManifest


class DatasetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceInterval:
    document_id: str
    start: int
    end: int


@dataclass(frozen=True)
class EvidenceUnit:
    identifier: str
    evidence_type: str
    intervals: tuple[EvidenceInterval, ...]


EVIDENCE_TYPES = frozenset({"text", "table", "formula"})


def load_manifest(path: str | Path, *, verify_checksum: bool = True) -> DatasetManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "name",
        "version",
        "task",
        "split",
        "source",
        "license",
        "revision",
        "checksum",
        "preprocessing_version",
        "split_seed",
        "samples",
    }
    missing = required - payload.keys()
    if missing:
        raise DatasetValidationError(f"Manifest is missing fields: {', '.join(sorted(missing))}")
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise DatasetValidationError("Manifest must contain at least one sample")
    ids = [str(sample.get("id", "")) for sample in samples if isinstance(sample, Mapping)]
    if not all(ids) or len(ids) != len(set(ids)):
        raise DatasetValidationError("Sample IDs must be present and unique")
    manifest = DatasetManifest(
        name=str(payload["name"]),
        version=str(payload["version"]),
        task=str(payload["task"]),
        split=str(payload["split"]),
        source=str(payload["source"]),
        license=str(payload["license"]),
        revision=str(payload["revision"]),
        checksum=str(payload["checksum"]),
        preprocessing_version=str(payload["preprocessing_version"]),
        split_seed=int(payload["split_seed"]),
        samples=tuple(dict(sample) for sample in samples),
    )
    if verify_checksum:
        actual = manifest_content_checksum(manifest.samples)
        if manifest.checksum != actual:
            raise DatasetValidationError(
                f"Manifest checksum mismatch: expected {manifest.checksum}, computed {actual}"
            )
    validate_evidence(manifest)
    return manifest


def manifest_content_checksum(samples: Sequence[Mapping[str, object]]) -> str:
    import hashlib

    encoded = json.dumps(
        list(samples), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence(manifest: DatasetManifest) -> None:
    if manifest.task != "rag":
        return
    documents = {
        str(sample["id"]): str(sample["text"])
        for sample in manifest.samples
        if sample.get("kind") == "document"
    }
    if not documents or any(not text.strip() for text in documents.values()):
        raise DatasetValidationError("RAG manifests require non-empty source documents")
    for sample in manifest.samples:
        if sample.get("kind") != "question":
            continue
        question_id = str(sample["id"])
        if not str(sample.get("question", "")).strip():
            raise DatasetValidationError(f"Question {question_id} has empty text")
        document_id = str(sample.get("document_id", ""))
        if document_id not in documents:
            raise DatasetValidationError(
                f"Question {question_id} refers to an unknown document"
            )
        answerable = sample.get("answerable")
        if not isinstance(answerable, bool):
            raise DatasetValidationError(
                f"Question {question_id} must declare boolean answerable"
            )
        units = evidence_units(sample)
        if answerable and not units:
            raise DatasetValidationError(
                f"Answerable question {question_id} requires verified evidence units"
            )
        if not answerable and units:
            raise DatasetValidationError(
                f"Unanswerable question {question_id} cannot contain evidence units"
            )
        identifiers = [unit.identifier for unit in units]
        if len(identifiers) != len(set(identifiers)):
            raise DatasetValidationError(
                f"Question {question_id} contains duplicate evidence-unit IDs"
            )
        signatures = [
            (
                unit.evidence_type,
                tuple(
                    (interval.document_id, interval.start, interval.end)
                    for interval in unit.intervals
                ),
            )
            for unit in units
        ]
        if len(signatures) != len(set(signatures)):
            raise DatasetValidationError(
                f"Question {question_id} contains duplicate evidence units"
            )
        question_type = str(sample.get("evidence_type", ""))
        if answerable and question_type not in EVIDENCE_TYPES | {"mixed"}:
            raise DatasetValidationError(
                f"Question {question_id} has unsupported evidence_type {question_type!r}"
            )
        unit_types = {unit.evidence_type for unit in units}
        if question_type == "mixed" and len(unit_types) < 2:
            raise DatasetValidationError(
                f"Mixed question {question_id} requires at least two evidence types"
            )
        if question_type in EVIDENCE_TYPES and unit_types - {question_type}:
            raise DatasetValidationError(
                f"Question {question_id} evidence units contradict evidence_type {question_type!r}"
            )
        for unit in units:
            document_ids = {interval.document_id for interval in unit.intervals}
            if len(document_ids) != 1:
                raise DatasetValidationError(
                    f"Evidence unit {unit.identifier} must fit within one source document"
                )
            if document_ids != {document_id}:
                raise DatasetValidationError(
                    f"Evidence unit {unit.identifier} does not belong to question "
                    f"document {document_id}"
                )
            coordinates = [(interval.start, interval.end) for interval in unit.intervals]
            if coordinates != sorted(coordinates) or any(
                left[1] > right[0] for left, right in zip(coordinates, coordinates[1:])
            ):
                raise DatasetValidationError(
                    f"Evidence unit {unit.identifier} intervals must be ordered and non-overlapping"
                )
            for interval in unit.intervals:
                document = documents.get(interval.document_id)
                if (
                    document is None
                    or interval.start < 0
                    or interval.end <= interval.start
                    or interval.end > len(document)
                ):
                    raise DatasetValidationError(
                        f"Invalid evidence offset in question {question_id}"
                    )
            _validate_evidence_text(sample, unit, documents)


def evidence_units(question: Mapping[str, object]) -> tuple[EvidenceUnit, ...]:
    """Parse the frozen RAG evidence-unit representation without fuzzy inference."""

    raw_evidence = question.get("evidence", [])
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        raise DatasetValidationError(
            f"Evidence must be a sequence in question {question.get('id', '')}"
        )
    question_type = str(question.get("evidence_type", ""))
    units: list[EvidenceUnit] = []
    for raw_unit in raw_evidence:
        if not isinstance(raw_unit, Mapping):
            raise DatasetValidationError(
                f"Evidence must be an object in question {question.get('id', '')}"
            )
        identifier = str(raw_unit.get("id", "")).strip()
        if not identifier:
            raise DatasetValidationError(
                f"Every evidence unit requires a stable ID in question {question.get('id', '')}"
            )
        evidence_type = str(raw_unit.get("evidence_type", question_type))
        if evidence_type not in EVIDENCE_TYPES:
            raise DatasetValidationError(
                f"Evidence unit {identifier} has unsupported type {evidence_type!r}"
            )
        raw_intervals = raw_unit.get("intervals")
        if raw_intervals is None:
            raw_intervals = [raw_unit]
        if not isinstance(raw_intervals, Sequence) or isinstance(
            raw_intervals, (str, bytes)
        ):
            raise DatasetValidationError(
                f"Evidence unit {identifier} intervals must be a sequence"
            )
        intervals: list[EvidenceInterval] = []
        for raw_interval in raw_intervals:
            if not isinstance(raw_interval, Mapping):
                raise DatasetValidationError(
                    f"Evidence unit {identifier} contains a malformed interval"
                )
            start = raw_interval.get("start")
            end = raw_interval.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                raise DatasetValidationError(
                    f"Evidence unit {identifier} has non-integer offsets"
                )
            intervals.append(
                EvidenceInterval(str(raw_interval.get("document_id", "")), start, end)
            )
        if not intervals:
            raise DatasetValidationError(
                f"Evidence unit {identifier} requires at least one interval"
            )
        units.append(EvidenceUnit(identifier, evidence_type, tuple(intervals)))
    return tuple(units)


def _validate_evidence_text(
    question: Mapping[str, object],
    unit: EvidenceUnit,
    documents: Mapping[str, str],
) -> None:
    raw_units = question.get("evidence", [])
    raw_unit = next(
        raw
        for raw in raw_units
        if isinstance(raw, Mapping) and str(raw.get("id", "")) == unit.identifier
    )
    raw_intervals = raw_unit.get("intervals")
    source_rows = raw_intervals if raw_intervals is not None else [raw_unit]
    for raw, interval in zip(source_rows, unit.intervals, strict=True):
        if not isinstance(raw, Mapping) or raw.get("text") is None:
            continue
        document = documents[interval.document_id]
        if document[interval.start : interval.end] != str(raw["text"]):
            raise DatasetValidationError(
                f"Evidence text mismatch in question {question.get('id', '')}"
            )


def assert_no_split_leakage(manifests: Sequence[DatasetManifest]) -> None:
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    seen_audio_assets: dict[str, str] = {}
    seen_audio_families: dict[str, tuple[str, str]] = {}
    seen_documents: list[tuple[str, set[tuple[str, ...]]]] = []
    import hashlib

    for manifest in manifests:
        for sample in manifest.samples:
            if sample.get("kind") == "audio":
                label = f"{manifest.name}/{manifest.split}"
                sample_id = str(sample.get("id", ""))
                checksum = str(sample.get("asset_sha256", ""))
                family = str(sample.get("document_family", ""))
                for name, value, seen in (
                    ("sample ID", sample_id, seen_ids),
                    ("asset checksum", checksum, seen_audio_assets),
                ):
                    if not value:
                        raise DatasetValidationError(f"ASR {label} has an empty {name}")
                    if value in seen:
                        raise DatasetValidationError(
                            f"ASR split leakage: {name} {value!r} occurs in "
                            f"{seen[value]} and {label}"
                        )
                    seen[value] = label
                if not family:
                    raise DatasetValidationError(f"ASR {label} has an empty document family")
                previous_family = seen_audio_families.get(family)
                if previous_family is not None and previous_family[0] != manifest.split:
                    raise DatasetValidationError(
                        f"ASR split leakage: document family {family!r} occurs in "
                        f"{previous_family[1]} and {label}"
                    )
                seen_audio_families.setdefault(family, (manifest.split, label))
                continue
            if sample.get("kind") != "document":
                continue
            sample_id = str(sample["id"])
            normalized = " ".join(str(sample.get("text", "")).casefold().split())
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            if sample_id in seen_ids or digest in seen_hashes:
                previous = seen_ids.get(sample_id) or seen_hashes[digest]
                raise DatasetValidationError(
                    f"Document leakage between {previous} and "
                    f"{manifest.name}/{manifest.split}: {sample_id}"
                )
            tokens = normalized.split()
            width = min(5, len(tokens))
            shingles = {
                tuple(tokens[index : index + width])
                for index in range(max(1, len(tokens) - width + 1))
            }
            for previous, previous_shingles in seen_documents:
                union = shingles | previous_shingles
                similarity = len(shingles & previous_shingles) / len(union) if union else 1.0
                if similarity >= 0.85:
                    raise DatasetValidationError(
                        f"Near-duplicate document leakage between {previous} and "
                        f"{manifest.name}/{manifest.split}: {sample_id}"
                    )
            seen_ids[sample_id] = f"{manifest.name}/{manifest.split}"
            seen_hashes[digest] = f"{manifest.name}/{manifest.split}"
            seen_documents.append((f"{manifest.name}/{manifest.split}", shingles))


def seal_manifest(path: str | Path) -> DatasetManifest:
    """Write the samples checksum, then enforce the complete manifest contract."""
    from edumind.common.artifacts import atomic_write_json

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DatasetValidationError("Manifest must contain a non-empty samples list")
    payload["checksum"] = manifest_content_checksum(samples)
    atomic_write_json(manifest_path, payload)
    return load_manifest(manifest_path)


def main() -> int:
    """Small direct utility for sealing and validating hand-authored manifests."""
    import argparse

    parser = argparse.ArgumentParser(description="Seal or validate benchmark manifests")
    parser.add_argument("action", choices=("seal", "validate"))
    parser.add_argument("manifests", nargs="+", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "seal":
        for path in arguments.manifests:
            seal_manifest(path)
    else:
        assert_no_split_leakage([load_manifest(path) for path in arguments.manifests])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
