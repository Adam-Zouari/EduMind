"""Reviewed generation references and sealed, preparation-only data reports.

Gold claims and verification material are never inferred from accepted answers.
All text is inline in the manifest, so shared report hashing covers its bytes.
"""

from collections import Counter
from hashlib import sha256
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.common.datasets import (
    evidence_units,
    load_manifest,
    validate_evidence,
)
from experiments.benchmarks.common.protocol import (
    boolean,
    integer,
    sequence,
    strict_object,
    string,
)


def _objects(value, label, fields, *, allow_empty=False):
    rows = [strict_object(item, label, fields) for item in sequence(value, label)]
    ids = [string(item["id"], label + ".id") for item in rows]
    if (not ids and not allow_empty) or len(ids) != len(set(ids)):
        raise ValueError(f"{label} requires a nonempty unique ID inventory")
    return rows


def questions_from_manifest(manifest):
    """Validate references and build deterministic evaluator-only coverage maps."""
    if manifest.task != "rag":
        raise ValueError("Generation requires a RAG manifest")
    validate_evidence(manifest)
    documents = {
        row["id"]: row for row in manifest.samples if row.get("kind") == "document"
    }
    questions = []
    for row in manifest.samples:
        if row.get("kind") != "question":
            continue
        answerable = boolean(row.get("answerable"), "answerable")
        question = string(row.get("question"), "question")
        document = documents[row["document_id"]]
        accepted = sequence(row.get("accepted_answers"), "accepted_answers")
        for answer in accepted:
            string(answer, "accepted answer")
        claims = _objects(
            row.get("required_gold_claims"),
            "required_gold_claims",
            {"id", "text"},
            allow_empty=not answerable,
        )
        for claim in claims:
            string(claim["text"], "gold claim text")
        if len({claim["text"].strip().casefold() for claim in claims}) != len(claims):
            raise ValueError(
                "Gold claims must be non-duplicate; atomicity additionally requires human review"
            )
        material = _objects(
            row.get("verification_material"),
            "verification_material",
            {"id", "text", "sha256"},
            allow_empty=not answerable,
        )
        for item in material:
            string(item["text"], "verification text")
            if sha256(item["text"].encode("utf-8")).hexdigest() != item["sha256"]:
                raise ValueError("Verification material checksum mismatch")
        blocks = _objects(
            row.get("supplied_evidence"),
            "supplied_evidence",
            {"id", "text", "intervals"},
        )
        units = evidence_units(row)
        for block in blocks:
            string(block["text"], "evidence text")
            intervals = []
            source_texts = []
            for interval in sequence(block["intervals"], "supplied evidence intervals"):
                interval = strict_object(
                    interval, "interval", {"document_id", "start", "end"}
                )
                source_id = interval["document_id"]
                if source_id != row["document_id"]:
                    raise ValueError(
                        "Supplied context must come from this question's source document"
                    )
                source = documents[source_id]["text"]
                start = integer(interval["start"], "interval.start", minimum=0)
                end = integer(
                    interval["end"],
                    "interval.end",
                    minimum=start + 1,
                    maximum=len(source),
                )
                intervals.append((source_id, start, end))
                source_texts.append(source[start:end])
            if not intervals or block["text"] != "\n".join(source_texts):
                raise ValueError(
                    "Supplied evidence text must exactly match its frozen canonical source intervals"
                )
            block["gold_unit_ids"] = [
                unit.identifier
                for unit in units
                if all(
                    _contains(
                        intervals, required.document_id, required.start, required.end
                    )
                    for required in unit.intervals
                )
            ]
        gold_ids = [unit.identifier for unit in units]
        covered = {unit for block in blocks for unit in block["gold_unit_ids"]}
        if answerable and (
            not accepted or not claims or not gold_ids or not set(gold_ids) <= covered
        ):
            raise ValueError(
                "Answerable generation requires accepted answers, required gold claims, and sufficient supplied evidence"
            )
        if not answerable and (gold_ids or claims or accepted):
            raise ValueError(
                "Unanswerable generation must have explicit empty accepted answers, required claims and gold evidence"
            )
        questions.append(
            {
                "id": row["id"],
                "document_id": row["document_id"],
                "source_id": document.get("independent_source_id")
                or document.get("source_group_id")
                or row["document_id"],
                "question": question,
                "answerable": answerable,
                "evidence_type": row.get("evidence_type", "text"),
                "blocks": blocks,
                "accepted_answers": accepted,
                "gold_claims": claims,
                "gold_unit_ids": gold_ids,
                "verification_material": material,
            }
        )
    if not questions:
        raise ValueError("Generation requires a nonempty complete question set")
    return questions


def _contains(intervals, document, start, end):
    cursor = start
    for source, left, right in sorted(intervals):
        if source == document and left <= cursor:
            cursor = max(cursor, right)
    return cursor >= end


def requirements(protocol, profile):
    return {
        "reference_schema": "generation-source-verified-v1",
        "required_fields": [
            "accepted_answers",
            "required_gold_claims",
            "verification_material",
            "supplied_evidence",
        ],
        "validator_dependencies": {
            "generation/validation.py": sha256_file(Path(__file__))
        },
        "data": dict(protocol.data_requirements) if profile != "smoke" else {},
    }


def validate(path, profile, protocol):
    manifest = load_manifest(path)
    questions = questions_from_manifest(manifest)
    documents = [row for row in manifest.samples if row.get("kind") == "document"]
    counts = Counter(row["evidence_type"] for row in questions if row["answerable"])
    if profile != "smoke":
        for row in documents:
            for field in ("source_dataset", "source_license", "source_revision"):
                string(row.get(field), field)
            string(
                row.get("independent_source_id") or row.get("source_group_id"),
                "independent source ID",
            )
        qasper = sum(row["source_dataset"] == "allenai/qasper" for row in documents)
        if qasper != protocol.data_requirements["qasper_papers"][profile]:
            raise ValueError(
                "QASPER paper allocation does not match the generation protocol"
            )
        if not any(not row["answerable"] for row in questions):
            raise ValueError(
                "Authoritative generation requires reviewed unanswerable questions"
            )
        for name, minimum in protocol.data_requirements[
            "minimum_structured_questions"
        ].items():
            if counts[name] < minimum:
                raise ValueError(
                    f"Required structured evidence slice {name} needs {minimum} questions"
                )
    return {
        "question_count": len(questions),
        "document_count": len(documents),
        "source_count": len({row["source_id"] for row in questions}),
        "answerable_questions": sum(row["answerable"] for row in questions),
        "evidence_type_counts": dict(counts),
    }


def default_inventory(path, profile):
    if profile == "smoke":
        return (Path(path),)
    return tuple(
        Path(path)
        if name == profile
        else PROJECT_ROOT / f"data/benchmarks/rag/{suffix}.json"
        for name, suffix in (
            ("development", "rag-selection-dev"),
            ("validation", "rag-selection-validation"),
            ("locked", "rag-selection-locked-test"),
        )
    )


def prepare_report(path, profile, protocol, **options):
    options["inventory"] = options.get("inventory") or default_inventory(path, profile)
    return write_report(
        "generation",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        lambda source, phase: validate(source, phase, protocol),
        **options,
    )


def verify_inputs(path, profile, protocol, **options):
    options["inventory"] = options.get("inventory") or default_inventory(path, profile)
    return verify_report(
        "generation",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        **options,
    )
