"""Metric contract for the retrieval/reranking benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from experiments.benchmarks.common.contracts import DatasetManifest, SampleResult
from experiments.benchmarks.common.datasets import answerable_questions, evidence_units
from experiments.benchmarks.common.metrics import paired_bootstrap_interval
from experiments.benchmarks.rag.chunking_embedding.metrics import (
    covered_unit_ids,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import RetrievalProtocol

OPERATIONAL_DIRECTIONS = {
    "operational.full_stack_latency_ms_p50": "min",
    "operational.full_stack_latency_ms_p95": "min",
    "operational.first_stage_latency_ms_p50": "min",
    "operational.first_stage_latency_ms_p95": "min",
    "operational.reranker_latency_ms_p50": "min",
    "operational.reranker_latency_ms_p95": "min",
    "operational.cold_initialization_seconds": "min",
    "operational.peak_process_tree_ram_mb": "min",
    "operational.peak_vram_mb": "min",
    "operational.index_build_seconds": "min",
}
BASE_WORKLOAD_DIRECTIONS = {
    "workload.corpus_document_count": "descriptive",
    "workload.answerable_query_count": "descriptive",
    "workload.chunk_count": "descriptive",
    "workload.requested_pool_size": "descriptive",
    "workload.actual_pool_size_mean": "descriptive",
    "workload.actual_pool_size_min": "descriptive",
    "workload.short_pool_query_count": "descriptive",
    "workload.reranker_input_tokens_per_query_mean": "descriptive",
    "workload.reranker_input_tokens_per_query_p95": "descriptive",
}
STORAGE_DIRECTIONS = {
    "storage.index_bytes": "descriptive",
    "storage.required_index_bytes": "descriptive",
    "storage.incremental_index_bytes": "descriptive",
    "storage.reranker_snapshot_bytes": "descriptive",
}
VALIDITY_DIRECTIONS = {
    "validity.ranking_agreement": "gate",
    "validity.expected_query_count": "descriptive",
    "validity.processed_query_count": "descriptive",
    "validity.failed_query_count": "gate",
    "validity.truncated_input_count": "gate",
    "validity.nonfinite_score_count": "gate",
    "validity.pool_collection_match": "gate",
    "validity.exact_pool_permutation": "gate",
}


def directions_for(
    manifest: DatasetManifest,
    protocol: RetrievalProtocol,
) -> tuple[dict[str, str], tuple[str, ...]]:
    questions = answerable_questions(manifest)
    scopes = {"overall", *(str(question["evidence_type"]) for question in questions)}
    alpha_questions = [
        question for question in questions if len(evidence_units(question)) >= 2
    ]
    alpha_scopes = (
        {"overall", *(str(question["evidence_type"]) for question in alpha_questions)}
        if alpha_questions
        else set()
    )
    quality: dict[str, str] = {}
    for scope in sorted(scopes):
        for metric in protocol.quality_metrics:
            if metric in protocol.alpha_ndcg_metrics and scope not in alpha_scopes:
                continue
            quality[f"quality.{scope}.{metric}"] = "max"
    eligibility = {
        f"validity.quality.{scope}.{suffix}": "descriptive"
        for scope in sorted(scopes)
        for suffix in (
            "eligible_question_count",
            "eligible_document_count",
            "alpha_ndcg_eligible_question_count",
            "alpha_ndcg_eligible_document_count",
        )
    }
    sample_counts = {f"{metric}.sample_count": "descriptive" for metric in quality}
    directions = {
        **quality,
        **sample_counts,
        protocol.pool_recall_metric: "max",
        f"{protocol.pool_recall_metric}.sample_count": "descriptive",
        **VALIDITY_DIRECTIONS,
        **eligibility,
        **OPERATIONAL_DIRECTIONS,
        **BASE_WORKLOAD_DIRECTIONS,
        **{
            f"workload.retrieved_tokens_at_{cutoff}_{suffix}": "descriptive"
            for cutoff in protocol.quality_cutoffs
            for suffix in ("mean", "p95")
        },
        **STORAGE_DIRECTIONS,
    }
    universally_required = (
        *quality,
        *sample_counts,
        "validity.ranking_agreement",
        "validity.expected_query_count",
        "validity.processed_query_count",
        "validity.failed_query_count",
        "validity.truncated_input_count",
        "validity.nonfinite_score_count",
        "validity.pool_collection_match",
        "validity.exact_pool_permutation",
        *eligibility,
        "operational.full_stack_latency_ms_p50",
        "operational.full_stack_latency_ms_p95",
        "operational.first_stage_latency_ms_p50",
        "operational.first_stage_latency_ms_p95",
        "operational.cold_initialization_seconds",
        "operational.peak_process_tree_ram_mb",
        "operational.peak_vram_mb",
        "workload.corpus_document_count",
        "workload.answerable_query_count",
        "workload.chunk_count",
        "workload.requested_pool_size",
        "workload.actual_pool_size_mean",
        "workload.actual_pool_size_min",
        "workload.short_pool_query_count",
        *(
            f"workload.retrieved_tokens_at_{cutoff}_{suffix}"
            for cutoff in protocol.quality_cutoffs
            for suffix in ("mean", "p95")
        ),
    )
    return directions, universally_required


def primary_metrics(protocol: RetrievalProtocol) -> tuple[str, ...]:
    return tuple(
        f"quality.overall.{metric}" for metric in protocol.primary_quality_metrics
    )


def pool_evidence_unit_recall(
    question: Mapping[str, object], chunks: Sequence[object]
) -> float:
    units = evidence_units(question)
    recovered = set().union(*(covered_unit_ids(chunk, units) for chunk in chunks))
    return len(recovered) / len(units) if units else 0.0


def retrieved_tokens(chunks: Sequence[object], cutoff: int) -> int:
    return sum(int(chunk.tokens) for chunk in chunks[:cutoff])


def latency_summary(samples: Sequence[SampleResult]) -> dict[str, float]:
    full = np.asarray([sample.latency_seconds * 1000.0 for sample in samples])
    first = np.asarray(
        [float(sample.metadata["first_stage_latency_ms"]) for sample in samples]
    )
    result = {
        "full_stack_latency_ms_p50": float(np.quantile(full, 0.50)),
        "full_stack_latency_ms_p95": float(np.quantile(full, 0.95)),
        "first_stage_latency_ms_p50": float(np.quantile(first, 0.50)),
        "first_stage_latency_ms_p95": float(np.quantile(first, 0.95)),
    }
    reranker = [
        float(sample.metadata["reranker_latency_ms"])
        for sample in samples
        if sample.metadata.get("reranker_latency_ms") is not None
    ]
    if reranker:
        result.update(
            {
                "reranker_latency_ms_p50": float(np.quantile(reranker, 0.50)),
                "reranker_latency_ms_p95": float(np.quantile(reranker, 0.95)),
            }
        )
    return result


def latency_intervals(
    samples: Sequence[SampleResult],
    *,
    resamples: int,
    seed: int,
    minimum_documents: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    if not resamples:
        return {}
    documents = sorted({str(sample.metadata["document_id"]) for sample in samples})
    if len(documents) < minimum_documents:
        return {}
    families = {
        "full_stack": {str(sample.metadata["document_id"]): [] for sample in samples},
        "first_stage": {str(sample.metadata["document_id"]): [] for sample in samples},
    }
    if any(
        sample.metadata.get("reranker_latency_ms") is not None for sample in samples
    ):
        families["reranker"] = {
            str(sample.metadata["document_id"]): [] for sample in samples
        }
    for sample in samples:
        document = str(sample.metadata["document_id"])
        families["full_stack"][document].append(sample.latency_seconds * 1000.0)
        families["first_stage"][document].append(
            float(sample.metadata["first_stage_latency_ms"])
        )
        if "reranker" in families:
            families["reranker"][document].append(
                float(sample.metadata["reranker_latency_ms"])
            )
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    alpha = (1.0 - confidence) / 2.0
    for family, by_document in families.items():
        observed = [value for document in documents for value in by_document[document]]
        draws = {50: [], 95: []}
        for _ in range(resamples):
            selected = rng.integers(0, len(documents), size=len(documents))
            values = [
                value
                for position in selected
                for value in by_document[documents[int(position)]]
            ]
            for percentile, estimates in draws.items():
                estimates.append(float(np.quantile(values, percentile / 100.0)))
        for percentile, estimates in draws.items():
            key = f"operational.{family}_latency_ms_p{percentile}"
            result[key] = {
                "estimate": float(np.quantile(observed, percentile / 100.0)),
                "lower": float(np.quantile(estimates, alpha)),
                "upper": float(np.quantile(estimates, 1.0 - alpha)),
                "confidence": confidence,
            }
    return result


def paired_document_interval(
    baseline: Sequence[SampleResult],
    candidate: Sequence[SampleResult],
    metric: str,
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, float] | None, int, int]:
    baseline_by_id = {sample.sample_id: sample for sample in baseline}
    candidate_by_id = {sample.sample_id: sample for sample in candidate}
    ids = sorted(baseline_by_id.keys() & candidate_by_id.keys())
    ids = [
        identifier
        for identifier in ids
        if metric in baseline_by_id[identifier].metrics
        and metric in candidate_by_id[identifier].metrics
    ]
    by_document: dict[str, list[tuple[float, float]]] = {}
    for identifier in ids:
        left = baseline_by_id[identifier]
        right = candidate_by_id[identifier]
        document = str(left.metadata["document_id"])
        if document != str(right.metadata["document_id"]):
            raise ValueError(f"Document mismatch for paired question {identifier}")
        by_document.setdefault(document, []).append(
            (float(left.metrics[metric]), float(right.metrics[metric]))
        )
    baseline_values = [
        sum(value[0] for value in rows) / len(rows)
        for _, rows in sorted(by_document.items())
    ]
    candidate_values = [
        sum(value[1] for value in rows) / len(rows)
        for _, rows in sorted(by_document.items())
    ]
    if not baseline_values or not resamples or len(baseline_values) < 2:
        return None, len(ids), len(by_document)
    interval = paired_bootstrap_interval(
        candidate_values,
        baseline_values,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )
    return (
        {
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence": interval.confidence,
        },
        len(ids),
        len(by_document),
    )
