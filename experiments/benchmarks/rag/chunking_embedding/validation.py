"""Preparation-time RAG annotation validation and cheap sealed-report reuse."""

from collections import Counter
from pathlib import Path

from edumind.rag.tokenizers import TiktokenOffsetTokenizer
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.common.datasets import answerable_questions, load_manifest


def requirements(protocol, profile):
    return {
        "tokenizer": protocol.tokenizer,
        "evidence_schema": "complete-source-unit-v2",
        "data": dict(protocol.data_requirements) if profile != "smoke" else {},
    }


def validate(path, profile, protocol):
    manifest = load_manifest(path)
    if manifest.task != "rag":
        raise ValueError("Chunking/embedding requires a RAG manifest")
    documents = [row for row in manifest.samples if row.get("kind") == "document"]
    questions = answerable_questions(manifest)
    if any(
        not isinstance(row.get("id"), str)
        or not row["id"].strip()
        or row.get("kind") not in {"document", "question"}
        for row in manifest.samples
    ):
        raise ValueError(
            "RAG samples require string IDs and explicit document/question kinds"
        )
    if any(not isinstance(row.get("text"), str) for row in documents) or any(
        not isinstance(row.get("question"), str)
        for row in manifest.samples
        if row.get("kind") == "question"
    ):
        raise ValueError("Canonical documents and questions must be text strings")
    if not questions:
        raise ValueError("Chunking/embedding requires answerable questions")
    tokenizer = TiktokenOffsetTokenizer(protocol.tokenizer.removeprefix("tiktoken:"))
    if any(tokenizer.count(str(row["text"])) <= 0 for row in documents):
        raise ValueError("Canonical documents must contain source tokens")
    counts = Counter(str(row["evidence_type"]) for row in questions)
    if profile != "smoke":
        if any(
            not row.get(field)
            for row in documents
            for field in ("source_dataset", "source_license", "source_revision")
        ):
            raise ValueError("Authoritative RAG source provenance is incomplete")
        qasper = sum(row["source_dataset"] == "allenai/qasper" for row in documents)
        if qasper != protocol.data_requirements["qasper_papers"][profile]:
            raise ValueError("QASPER paper allocation does not match the protocol")
        for scope, minimum in protocol.data_requirements[
            "minimum_structured_questions"
        ].items():
            if counts[scope] < minimum:
                raise ValueError(
                    f"Required structured evidence slice {scope} needs {minimum} questions"
                )
    return {
        "document_count": len(documents),
        "answerable_question_count": len(questions),
        "evidence_type_counts": dict(counts),
    }


def prepare_report(path, profile, protocol, **options):
    return write_report(
        "chunking-embedding",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        lambda source, phase: validate(source, phase, protocol),
        **options,
    )


def verify_inputs(path, profile, protocol, **options):
    return verify_report(
        "chunking-embedding",
        profile,
        path,
        requirements(protocol, profile),
        Path(__file__),
        **options,
    )
