"""Final reference validation permits genuine retrieval misses at execution."""

from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.rag.generation.validation import (
    default_inventory,
    questions_from_manifest,
)


def manifest_path(profile):
    suffix = {
        "smoke": "generation-smoke",
        "preflight": "rag-selection-dev",
        "locked": "rag-selection-locked-test",
    }[profile]
    return PROJECT_ROOT / f"data/benchmarks/rag/{suffix}.json"


def requirements(protocol, profile):
    return {
        "reference_schema": "final-canonical-references-v1",
        "phase": profile,
        "protocol_checksum": protocol.meta.checksum,
    }


def validate(path, profile):
    manifest = load_manifest(path)
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    if profile == "preflight" and not any(
        row.get("preflight_stress") is True
        for row in manifest.samples
        if row.get("kind") == "question"
    ):
        raise ValueError("Final preflight requires explicitly reviewed stress inputs")
    if profile != "smoke":
        for row in manifest.samples:
            if row.get("kind") == "document" and not all(
                row.get(key)
                for key in (
                    "source_dataset",
                    "source_license",
                    "source_revision",
                )
            ):
                raise ValueError("Final canonical source provenance is incomplete")
        if not any(not q["answerable"] for q in questions):
            raise ValueError(
                "Final evaluation requires reviewed unanswerable questions"
            )
    return {
        "question_count": len(questions),
        "answerable_count": sum(q["answerable"] for q in questions),
    }


def _report(function, path, profile, protocol, options):
    if profile not in {"smoke", "preflight", "locked"}:
        raise ValueError("Final RAG has only smoke/preflight/locked inputs")
    # The preflight input split is development; there is no Final development evaluation.
    split = "development" if profile == "preflight" else profile
    options["inventory"] = options.get("inventory") or default_inventory(path, split)
    args = ("final-rag", split, path, requirements(protocol, profile), Path(__file__))
    if function is write_report:
        return function(*args, lambda source, _: validate(source, profile), **options)
    return function(*args, **options)


def prepare_report(path, profile, protocol, **options):
    return _report(write_report, path, profile, protocol, options)


def verify_inputs(path, profile, protocol, **options):
    return _report(verify_report, path, profile, protocol, options)
