"""Manifest loading, checksums, provenance, and leakage validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
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

    @property
    def document_id(self) -> str:
        return self.intervals[0].document_id


@dataclass(frozen=True)
class QuestionSources:
    document_ids: tuple[str, ...]
    independent_source_id: str | None

    def metadata(self):
        return {
            "source_document_ids": list(self.document_ids),
            "independent_source_id": self.independent_source_id,
        }


EVIDENCE_TYPES = frozenset({"text", "table", "formula"})


def answerable_questions(
    manifest: DatasetManifest,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("answerable")
    ]


def load_manifest(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    validate_annotations: bool = True,
) -> DatasetManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DatasetValidationError("Manifest must be a JSON object")
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
        raise DatasetValidationError(
            f"Manifest is missing fields: {', '.join(sorted(missing))}"
        )
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise DatasetValidationError("Manifest must contain at least one sample")
    if any(not isinstance(sample, Mapping) for sample in samples):
        raise DatasetValidationError("Every manifest sample must be a JSON object")
    ids = [str(sample.get("id", "")) for sample in samples]
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
    if validate_annotations:
        validate_evidence(manifest)
    return manifest


def require_manifest_split(
    manifest: DatasetManifest,
    profile: str,
    expected: str | Sequence[str],
) -> None:
    """Reject a manifest that does not belong to the requested benchmark profile."""

    allowed = {expected} if isinstance(expected, str) else set(expected)
    if manifest.split not in allowed:
        raise DatasetValidationError(
            f"Profile {profile} requires split {sorted(allowed)}, "
            f"received {manifest.split!r}"
        )


def manifest_content_checksum(samples: Sequence[Mapping[str, object]]) -> str:
    import hashlib

    encoded = json.dumps(
        list(samples), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence(manifest: DatasetManifest) -> dict[str, QuestionSources]:
    if manifest.task != "rag":
        return {}
    if any(
        not isinstance(row.get("id"), str)
        or not row["id"].strip()
        or row.get("kind") not in {"document", "question"}
        or row.get("split", manifest.split) != manifest.split
        for row in manifest.samples
    ) or len({row["id"] for row in manifest.samples}) != len(manifest.samples):
        raise DatasetValidationError(
            "RAG records require unique string IDs, kinds and matching splits"
        )
    documents = {
        sample["id"]: sample.get("text")
        for sample in manifest.samples
        if sample.get("kind") == "document"
    }
    if not documents or any(
        not isinstance(text, str) or not text.strip() for text in documents.values()
    ):
        raise DatasetValidationError("RAG manifests require non-empty source documents")
    by_question = {}
    for sample in manifest.samples:
        if sample.get("kind") != "question":
            continue
        question_id = str(sample["id"])
        if (
            not isinstance(sample.get("question"), str)
            or not sample["question"].strip()
        ):
            raise DatasetValidationError(f"Question {question_id} has empty text")
        if {"document_id", "source_group_id", "independent_source_id"} & sample.keys():
            raise DatasetValidationError(
                f"Question {question_id} must not declare an owner document or source group"
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
        if answerable:
            if "reviewed_document_ids" in sample:
                raise DatasetValidationError(
                    "Answerable question sources come from gold units"
                )
            source_documents = {unit.document_id for unit in units}
        else:
            reviewed = sample.get("reviewed_document_ids")
            if (
                not isinstance(reviewed, list)
                or any(not isinstance(source, str) for source in reviewed)
                or len(set(reviewed)) != len(reviewed)
            ):
                raise DatasetValidationError(
                    f"Question {question_id} must list unique reviewed_document_ids "
                    "(empty only if provenance is unknown)"
                )
            source_documents = set(reviewed)
        if source_documents - documents.keys():
            raise DatasetValidationError(
                f"Question {question_id} refers to an unknown document"
            )
        by_question[question_id] = tuple(sorted(source_documents))
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
            coordinates = [
                (interval.start, interval.end) for interval in unit.intervals
            ]
            if coordinates != sorted(coordinates) or any(
                left[1] > right[0] for left, right in pairwise(coordinates)
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
    return _linked_question_sources(manifest, documents, by_question)


def _linked_question_sources(manifest, documents, by_question):
    """Build connected components before any metric narrows the question inventory."""
    from edumind.common.artifacts import stable_hash

    parent = {document: document for document in documents}

    def root(document):
        while document != parent[document]:
            parent[document] = parent[parent[document]]
            document = parent[document]
        return document

    def link(left, right):
        a, b = sorted((root(left), root(right)))
        parent[b] = a

    labels = {}
    for row in manifest.samples:
        if row["kind"] != "document":
            continue
        families = tuple(
            (kind, row[field])
            for field, kind in (
                ("independent_source_id", "family"),
                ("source_group_id", "family"),
                ("duplicate_group_id", "duplicate"),
            )
            if field in row
        )
        for kind, label in families:
            if not isinstance(label, str) or not label or label != label.strip():
                raise DatasetValidationError(
                    "Reviewed source labels must be nonempty strings"
                )
            link(row["id"], labels.setdefault((kind, label), row["id"]))
    for refs in by_question.values():
        for document in refs[1:]:
            link(refs[0], document)
    groups = {}
    for key, document in labels.items():
        groups.setdefault(root(document), []).append(key)
    group_ids = {
        component: keys[0][1]
        if len(keys) == 1 and keys[0][0] == "family"
        else "linked:" + stable_hash(sorted(keys))
        for component, keys in groups.items()
    }
    if len(set(group_ids.values())) != len(group_ids):
        raise DatasetValidationError("Independent source component IDs collide")
    return {
        question: QuestionSources(refs, group_ids.get(root(refs[0])) if refs else None)
        for question, refs in by_question.items()
    }


def evidence_units(question: Mapping[str, object]) -> tuple[EvidenceUnit, ...]:
    """Parse the frozen RAG evidence-unit representation without fuzzy inference."""

    raw_evidence = question.get("evidence")
    if not isinstance(raw_evidence, Sequence) or isinstance(raw_evidence, (str, bytes)):
        raise DatasetValidationError(
            f"Evidence must be a sequence in question {question.get('id', '')}"
        )
    units: list[EvidenceUnit] = []
    for raw_unit in raw_evidence:
        if not isinstance(raw_unit, Mapping):
            raise DatasetValidationError(
                f"Evidence must be an object in question {question.get('id', '')}"
            )
        identifier = raw_unit.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier != identifier.strip()
        ):
            raise DatasetValidationError(
                f"Every evidence unit requires an unpadded stable ID in question {question.get('id', '')}"
            )
        evidence_type = str(raw_unit.get("evidence_type", ""))
        if evidence_type not in EVIDENCE_TYPES:
            raise DatasetValidationError(
                f"Evidence unit {identifier} has unsupported type {evidence_type!r}"
            )
        document = raw_unit.get("document_id")
        if (
            not isinstance(document, str)
            or not document
            or document != document.strip()
        ):
            raise DatasetValidationError(
                f"Evidence unit {identifier} requires its canonical document_id"
            )
        raw_intervals = raw_unit.get("intervals")
        if not isinstance(raw_intervals, Sequence) or isinstance(
            raw_intervals, (str, bytes)
        ):
            raise DatasetValidationError(
                f"Evidence unit {identifier} intervals must be a sequence"
            )
        intervals: list[EvidenceInterval] = []
        for raw_interval in raw_intervals:
            if not isinstance(raw_interval, Mapping) or set(raw_interval) - {
                "start",
                "end",
                "text",
            }:
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
            intervals.append(EvidenceInterval(document, start, end))
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
    source_rows = raw_unit["intervals"]
    for raw, interval in zip(source_rows, unit.intervals, strict=True):
        if "text" not in raw:
            continue
        document = documents[interval.document_id]
        if document[interval.start : interval.end] != raw["text"]:
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
                    raise DatasetValidationError(
                        f"ASR {label} has an empty document family"
                    )
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
                similarity = (
                    len(shingles & previous_shingles) / len(union) if union else 1.0
                )
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
