"""Document reference, physical-page, and source-integrity validation."""

from __future__ import annotations

import zipfile
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.common.datasets import load_manifest

from .metrics import load_reference_data, validate_reference


def requirements(protocol, profile):
    return {
        "normalization": "NFC-prose-v1",
        "reference_schema": "document-v2",
        "counts": dict(protocol.sample_counts.get(profile, {})),
    }


def validate(path, profile, protocol):
    authoritative = profile != "smoke"
    items = [
        item
        for item in load_manifest(path).samples
        if item.get("kind") in {"image", "pdf", "docx"}
    ]
    if not items:
        raise ValueError("Document manifest has no document samples")
    metadata = {}
    counts = {kind: 0 for kind in ("image", "pdf", "docx")}
    table_references, formula_references = set(), set()
    for item in items:
        counts[item["kind"]] += 1
        source = PROJECT_ROOT / str(item["source_path"])
        if sha256_file(source) != item.get("asset_sha256"):
            raise ValueError("Document asset checksum mismatch")
        if authoritative and any(
            not item.get(name)
            for name in ("source_license", "source_revision", "document_family")
        ):
            raise ValueError("Document source provenance is incomplete")
        if item.get("reference_path"):
            reference_path = PROJECT_ROOT / str(item["reference_path"])
            if sha256_file(reference_path) != item.get("reference_sha256"):
                raise ValueError("Document reference checksum mismatch")
        payload, reference = load_reference_data(item)
        validate_reference(
            item, authoritative=authoritative, payload=payload, reference=reference
        )
        table_references.update(
            element.html
            for element in reference.elements
            if "tables" in reference.capabilities and element.kind.value == "table"
        )
        formula_references.update(
            element.latex
            for element in reference.elements
            if "formulas" in reference.capabilities and element.kind.value == "formula"
        )
        if item["kind"] == "image":
            from PIL import Image

            with Image.open(source) as image:
                image.verify()
            pages = 1
        elif item["kind"] == "pdf":
            import pypdfium2

            with pypdfium2.PdfDocument(source) as document:
                pages = len(document)
            if pages < 1:
                raise ValueError("PDF has no physical pages")
        else:
            with zipfile.ZipFile(source) as document:
                if (
                    document.testzip() is not None
                    or "word/document.xml" not in document.namelist()
                ):
                    raise ValueError("Invalid DOCX package")
            pages = None
        if "pages" in reference.capabilities and (
            pages is None or set(reference.pages) != set(range(1, pages + 1))
        ):
            raise ValueError(
                "Reference must identify every physical page, including blanks"
            )
        if item["kind"] == "docx" and "layout_boxes" in reference.capabilities:
            raise ValueError("Native DOCX cannot claim invented page geometry")
        metadata[str(item["id"])] = {"physical_page_count": pages}
    for kind, minimum in protocol.sample_counts.get(profile, {}).items():
        if counts[kind] < minimum:
            raise ValueError(f"Document {profile} requires {minimum} {kind} inputs")
    if table_references or formula_references:
        from .official_metrics import score_official_metrics

        tables, formulas = score_official_metrics(
            [(value, value) for value in sorted(table_references)],
            [(value, value) for value in sorted(formula_references)],
            timeout_seconds=protocol.evaluator_timeout_seconds,
        )
        if any(value != (1.0, 1.0) for value in tables) or any(
            value != 1.0 for value in formulas
        ):
            raise ValueError(
                "Document references cannot be reconstructed by the pinned official evaluators"
            )
    return {"samples": metadata, "counts": counts}


def prepare_report(path, profile, protocol, **options):
    return write_report(
        "document",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        lambda source, phase: validate(source, phase, protocol),
        **options,
    )


def verified_inputs(path, profile, protocol, **options):
    return verify_report(
        "document",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        **options,
    )
