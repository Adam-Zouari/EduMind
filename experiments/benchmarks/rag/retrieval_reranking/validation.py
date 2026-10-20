"""RR input validation and sealed-report reuse; never constructs query pools."""

from pathlib import Path

from edumind.common.artifacts import sha256_file
from experiments.benchmarks.common.data_validation import verify_report, write_report
from experiments.benchmarks.rag.chunking_embedding import (
    validation as chunking_validation,
)


def requirements(protocol, profile, *, chunking_protocol):
    chunking = chunking_protocol
    if protocol.evaluation_tokenizer != chunking.tokenizer:
        raise ValueError("RR and selected chunking evaluation tokenizers must match")
    return {
        "evaluation_tokenizer": protocol.evaluation_tokenizer,
        "evidence_coverage_rule": protocol.evidence_coverage_rule,
        "requested_pool_size": protocol.pool_size,
        "quality_cutoffs": list(protocol.quality_cutoffs),
        "chunking": chunking_validation.requirements(chunking, profile),
        "chunking_validator_sha256": sha256_file(Path(chunking_validation.__file__)),
        # Constructed corpus/index/pool checks are runtime gates, not annotations.
        "input_contract": "retrieval-reranking-inputs-v1",
    }


def validate(path, profile, protocol, *, chunking_protocol):
    chunking = chunking_protocol
    requirements(protocol, profile, chunking_protocol=chunking)
    result = chunking_validation.validate(path, profile, chunking)
    if not result["document_count"] or not result["answerable_question_count"]:
        raise ValueError(
            "RR requires a nonempty canonical corpus and answerable allocation"
        )
    return {
        **result,
        "requested_pool_size": protocol.pool_size,
        "quality_cutoffs": list(protocol.quality_cutoffs),
    }


def prepare_report(path, profile, protocol, *, chunking_protocol, **options):
    chunking = chunking_protocol
    return write_report(
        "retrieval-reranking",
        profile,
        path,
        requirements(protocol, profile, chunking_protocol=chunking),
        Path(__file__),
        lambda source, phase: validate(
            source, phase, protocol, chunking_protocol=chunking
        ),
        **options,
    )


def verify_inputs(path, profile, protocol, *, chunking_protocol, **options):
    return verify_report(
        "retrieval-reranking",
        profile,
        path,
        requirements(protocol, profile, chunking_protocol=chunking_protocol),
        Path(__file__),
        **options,
    )
