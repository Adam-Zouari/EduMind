"""Manifest loading, checksums, provenance, and leakage validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import DatasetManifest


class DatasetValidationError(ValueError):
    pass


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
    for sample in manifest.samples:
        if sample.get("kind") != "question":
            continue
        raw_evidence = sample.get("evidence", [])
        if not isinstance(raw_evidence, (list, tuple)):
            raise DatasetValidationError(f"Evidence must be a sequence in question {sample['id']}")
        for evidence in raw_evidence:
            if not isinstance(evidence, Mapping):
                raise DatasetValidationError(
                    f"Evidence must be an object in question {sample['id']}"
                )
            document = documents.get(str(evidence.get("document_id")))
            start, end = int(evidence.get("start", -1)), int(evidence.get("end", -1))
            if document is None or start < 0 or end < start or end > len(document):
                raise DatasetValidationError(f"Invalid evidence offset in question {sample['id']}")
            expected = evidence.get("text")
            if expected is not None and document[start:end] != expected:
                raise DatasetValidationError(f"Evidence text mismatch in question {sample['id']}")


def assert_no_split_leakage(manifests: list[DatasetManifest]) -> None:
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    seen_documents: list[tuple[str, set[tuple[str, ...]]]] = []
    import hashlib

    for manifest in manifests:
        for sample in manifest.samples:
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
