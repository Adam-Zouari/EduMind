"""Sealed data-validation reports, independent of candidate measurement."""

from __future__ import annotations

import json
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split

REPORT_ROOT = PROJECT_ROOT / "artifacts/benchmarks/data-validation"
PROFILES = ("smoke", "development", "validation", "locked")


def manifest_path(benchmark, profile):
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
    split = "locked-test" if profile == "locked" else profile
    return PROJECT_ROOT / f"data/benchmarks/extraction/{benchmark}-{split}.json"


def input_identity(
    benchmark, profile, path, requirements, validator_path, inventory=None
):
    """Hash actual input bytes; do not decode media or evaluate annotations."""
    path = Path(path).resolve()
    inventory = tuple(
        Path(value).resolve()
        for value in (
            inventory
            or (
                (path,)
                if profile == "smoke"
                else tuple(
                    path if name == profile else manifest_path(benchmark, name)
                    for name in PROFILES[1:]
                )
            )
        )
    )
    if path not in inventory:
        raise ValueError("Validation inventory must contain the invocation's manifest")
    files = {}
    splits = {}
    for source in inventory:
        manifest = load_manifest(source)
        files[str(source)] = sha256_file(source)
        splits[str(source)] = manifest.split
        for item in manifest.samples:
            kinds = (
                {"image", "pdf", "docx"}
                if benchmark == "document"
                else {"audio", "audio_reliability"}
                if benchmark == "audio"
                else {benchmark}
            )
            if item.get("kind") not in kinds:
                continue
            for field in ("source_path", "reference_path"):
                if item.get(field):
                    asset = (PROJECT_ROOT / str(item[field])).resolve()
                    files[str(asset)] = sha256_file(asset)
    extraction = PROJECT_ROOT / "experiments/benchmarks/extraction"
    dependencies = [
        Path(__file__).with_name("datasets.py"),
        Path(__file__).with_name("metrics.py"),
    ]
    dependencies.append(PROJECT_ROOT / "src/edumind/extraction/contracts.py")
    dependencies.extend(
        extraction / name
        for name in (
            (
                "document/metrics.py",
                "document/adapters.py",
                "document/official_metrics.py",
                "document/omnidocbench_worker.py",
            )
            if benchmark == "document"
            else ("audio/evaluate.py", "media.py")
            if benchmark == "audio"
            else (
                "video/metrics.py",
                "video/frozen_asr_worker.py",
                "audio/evaluate.py",
                "audio/validation.py",
                "media.py",
            )
        )
    )
    implementation = {
        str(source): sha256_file(source)
        for source in (Path(__file__), Path(validator_path), *dependencies)
    }
    resolved = {
        "schema_version": 1,
        "validator_version": "extraction-data-v1",
        "benchmark": benchmark,
        "profile": profile,
        "manifest_path": str(path),
        "requirements": requirements,
        "files": files,
        "inventory": splits,
        "implementation": implementation,
    }
    return stable_hash(resolved), resolved


def validate_inventory(benchmark, profile, path, inventory):
    manifest = load_manifest(path)
    require_manifest_split(
        manifest, profile, "locked-test" if profile == "locked" else profile
    )
    if profile == "smoke":
        return
    observed = {}
    splits = set()
    for source in inventory:
        frozen = load_manifest(source)
        if frozen.split in {"development", "validation", "locked-test"}:
            splits.add(frozen.split)
        for item in frozen.samples:
            if benchmark == "document" and item.get("kind") not in {
                "pdf",
                "docx",
                "image",
            }:
                continue
            if benchmark != "document" and item.get("kind") not in (
                {"audio", "audio_reliability"} if benchmark == "audio" else {benchmark}
            ):
                continue
            split = str(item.get("split", frozen.split))
            if (
                frozen.split in {"development", "validation", "locked-test"}
                and split != frozen.split
            ):
                raise ValueError(
                    "Inventory sample split contradicts its manifest split"
                )
            group = item.get("independent_source_id") or item.get("source_group_id")
            if not isinstance(group, str) or not group:
                raise ValueError(
                    "Authoritative samples require reviewed independent source/group IDs"
                )
            keys = ["source:" + group, "asset:" + str(item["asset_sha256"])]
            for path_field, checksum_field in (
                ("source_path", "asset_sha256"),
                ("reference_path", "reference_sha256"),
            ):
                if item.get(path_field) and sha256_file(
                    PROJECT_ROOT / str(item[path_field])
                ) != item.get(checksum_field):
                    raise ValueError("Inventory asset/reference checksum mismatch")
            if item.get("duplicate_group_id"):
                keys.append("duplicate:" + str(item["duplicate_group_id"]))
            for key in keys:
                if key in observed and observed[key] != split:
                    raise ValueError(f"Cross-split source leakage: {key}")
                observed[key] = split
    if splits != {"development", "validation", "locked-test"}:
        raise ValueError(
            "Authoritative validation requires the complete three-split inventory"
        )


def write_report(
    benchmark,
    profile,
    path,
    requirements,
    validator_path,
    validate,
    *,
    inventory=None,
    report_root=REPORT_ROOT,
):
    try:
        fingerprint, identity = input_identity(
            benchmark, profile, path, requirements, validator_path, inventory
        )
    except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
        identity = {
            "schema_version": 1,
            "benchmark": benchmark,
            "profile": profile,
            "manifest_path": str(Path(path).resolve()),
            "requirements": requirements,
            "unsealed": True,
            "input_error": f"{type(exc).__name__}: {exc}",
        }
        fingerprint = stable_hash(identity)
        destination = Path(report_root) / benchmark / (fingerprint + ".json")
        report = {
            **identity,
            "fingerprint": fingerprint,
            "status": "failed",
            "errors": [identity["input_error"]],
        }
        atomic_write_json(destination, report)
        return destination, report
    destination = Path(report_root) / benchmark / (fingerprint + ".json")
    report = {**identity, "fingerprint": fingerprint, "status": "failed", "errors": []}
    try:
        validate_inventory(benchmark, profile, path, identity["inventory"])
        report["metadata"] = validate(Path(path), profile)
        report["status"] = "passed"
    except (ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
        report["errors"] = [f"{type(exc).__name__}: {exc}"]
    report["report_checksum"] = stable_hash(report)
    atomic_write_json(destination, report)
    return destination, report


def verify_report(
    benchmark,
    profile,
    path,
    requirements,
    validator_path,
    *,
    report_root=REPORT_ROOT,
    inventory=None,
):
    fingerprint, identity = input_identity(
        benchmark, profile, path, requirements, validator_path, inventory
    )
    destination = Path(report_root) / benchmark / (fingerprint + ".json")
    if not destination.is_file():
        raise ValueError(
            f"Missing or stale data validation. Run python -m experiments.benchmarks.validate {benchmark} --profile {profile}"
        )
    report = json.loads(destination.read_text(encoding="utf-8"))
    if report.get("report_checksum") != stable_hash(
        {key: value for key, value in report.items() if key != "report_checksum"}
    ):
        raise ValueError("Data-validation report content checksum mismatch")
    if report.get("status") != "passed" or report.get("fingerprint") != fingerprint:
        raise ValueError("Data-validation report did not pass for these inputs")
    if any(report.get(key) != value for key, value in identity.items()):
        raise ValueError("Data-validation report identity mismatch")
    return destination, report
