"""Offline support-policy readiness before one-use held-out execution."""

from experiments.benchmarks.rag.generation.metrics import required_ci_policy_keys
from experiments.benchmarks.rag.generation.validation import questions_from_manifest


def require_locked_ci_readiness(manifest, generation_protocol, retrieval_protocol):
    """Require reviewed policies, not invented minima or sufficient source counts.

    Final permits missing retrieved evidence. This reference-only check neither
    loads a model nor examines generated answers or the eventual CI estimates.
    """
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    missing = [
        "generation." + key
        for key in required_ci_policy_keys(questions)
        if key not in generation_protocol.minimum_ci_sources
    ]
    for field in ("minimum_quality_ci_documents", "minimum_latency_ci_documents"):
        if getattr(retrieval_protocol, field) is None:
            missing.append("retrieval.statistics." + field)
    if missing:
        raise ValueError(
            "ci_support_policy_pending: freeze required source-support policies before locked reservation: "
            + ", ".join(sorted(missing))
        )
