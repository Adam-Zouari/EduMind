"""Fresh-process benchmark contract for chunker/embedding pair selection."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.datasets import evidence_units
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.rag.evaluation import (
    InputCompatibilityError,
    build_index,
    dense_rank_with_scores,
)

from .metrics import (
    ALPHA,
    ALPHA_NDCG_METRICS,
    AUDIT_K,
    MAX_QUALITY_K,
    PRIMARY_QUALITY_METRICS,
    QUALITY_CUTOFFS,
    QUALITY_METRICS,
    aggregate_quality,
    eligible_counts,
    latency_intervals,
    prefixed_quality,
    score_question,
)
from .profiles import embedding_spec


QUALITY_DIRECTIONS = {
    f"quality.{scope}.{metric}": "max"
    for scope in ("overall", "text", "table", "formula", "mixed")
    for metric in QUALITY_METRICS
}
OPERATIONAL_DIRECTIONS = {
    "operational.corpus_build_seconds": "min",
    "operational.corpus_build_source_tokens_per_second": "max",
    "operational.query_latency_ms_p50": "min",
    "operational.query_latency_ms_p95": "min",
    "operational.peak_process_tree_ram_mb": "min",
    "operational.peak_vram_mb": "min",
}
PRIMARY_METRICS = tuple(
    f"quality.overall.{metric}" for metric in PRIMARY_QUALITY_METRICS
)
PAIRED_METRICS = (
    *PRIMARY_METRICS,
    *(f"quality.overall.{metric}" for metric in ALPHA_NDCG_METRICS),
)
WORKER = Path(__file__).with_name("worker.py")


def directions_for(manifest: DatasetManifest) -> tuple[dict[str, str], tuple[str, ...]]:
    """Declare only the evidence slices present in this frozen manifest."""

    questions = _answerable_questions(manifest)
    scopes = {"overall", *(str(question["evidence_type"]) for question in questions)}
    alpha_questions = [
        question for question in questions if len(evidence_units(question)) >= 2
    ]
    alpha_scopes = (
        {"overall", *(str(question["evidence_type"]) for question in alpha_questions)}
        if alpha_questions
        else set()
    )
    quality = {}
    for name, direction in QUALITY_DIRECTIONS.items():
        _, scope, metric = name.split(".", 2)
        if scope not in scopes:
            continue
        if metric in ALPHA_NDCG_METRICS and scope not in alpha_scopes:
            continue
        quality[name] = direction
    directions = {**quality, **OPERATIONAL_DIRECTIONS}
    return directions, tuple(directions)


def evaluate_candidate(
    candidate: str,
    manifest: DatasetManifest,
    model_lock: Mapping[str, Mapping[str, object]],
    plan: BenchmarkPlan,
    *,
    device: str,
    dtype: str,
):
    """Evaluate one eligible pair; the caller supplies process isolation."""

    _seed_everything(plan.seed)
    os.environ["EDUMIND_BENCHMARK_EMBEDDING_DEVICE"] = device
    os.environ["EDUMIND_BENCHMARK_EMBEDDING_DTYPE"] = dtype
    chunker_name, embedding_name = _split_candidate(candidate)
    entry = model_lock[embedding_name]
    index = build_index(
        manifest,
        chunker_name,
        embedding_name,
        model_lock,
        with_dense=True,
        with_bm25=False,
    )
    questions = _answerable_questions(manifest)
    random.Random(plan.seed).shuffle(questions)
    if not questions:
        raise RuntimeError("Chunking/embedding requires answerable questions")
    for _ in range(plan.warmups):
        dense_rank_with_scores(index, str(questions[0]["question"]), AUDIT_K)

    samples: list[SampleResult] = []
    query_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    query_latencies: list[float] = []
    failures: list[str] = []
    failed_query_ids: set[str] = set()
    determinism_mismatches = 0
    for question in questions:
        rankings: list[list[tuple[int, float]]] = []
        latencies: list[float] = []
        for repetition in range(plan.repetitions):
            started = time.perf_counter()
            try:
                ranking = dense_rank_with_scores(
                    index, str(question["question"]), AUDIT_K
                )
                latency = time.perf_counter() - started
                rankings.append(ranking)
                latencies.append(latency)
                timing_rows.append(
                    {
                        "question_id": str(question["id"]),
                        "repetition": repetition + 1,
                        "latency_ms": latency * 1000.0,
                        "success": True,
                        "error": None,
                    }
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                latencies.append(latency)
                error = f"{type(exc).__name__}: {exc}"
                failed_query_ids.add(str(question["id"]))
                failures.append(f"{question['id']} repetition {repetition + 1}: {error}")
                timing_rows.append(
                    {
                        "question_id": str(question["id"]),
                        "repetition": repetition + 1,
                        "latency_ms": latency * 1000.0,
                        "success": False,
                        "error": error,
                    }
                )
        if len(rankings) != plan.repetitions:
            continue
        first_ids = [position for position, _ in rankings[0]][:MAX_QUALITY_K]
        deterministic = all(
            [position for position, _ in ranking][:MAX_QUALITY_K] == first_ids
            for ranking in rankings[1:]
        )
        if not deterministic:
            determinism_mismatches += 1
        latency = float(np.median(latencies))
        query_latencies.append(latency)
        selected = [index.chunks[position] for position, _ in rankings[0]]
        score = score_question(question, selected, index.chunks, index.tokenizer)
        evidence_type = str(question["evidence_type"])
        metrics = prefixed_quality(score.metrics, evidence_type)
        samples.append(
            SampleResult(
                str(question["id"]),
                metrics,
                latency,
                {
                    "document_id": str(question["document_id"]),
                    "evidence_type": evidence_type,
                    f"deterministic_top_{MAX_QUALITY_K}": deterministic,
                },
            )
        )
        query_rows.append(
            {
                "question_id": str(question["id"]),
                "document_id": str(question["document_id"]),
                "evidence_type": evidence_type,
                "latency_ms": latency * 1000.0,
                **score.metrics,
            }
        )
        for rank, (position, similarity) in enumerate(rankings[0], start=1):
            chunk = index.chunks[position]
            retrieval_rows.append(
                {
                    "question_id": str(question["id"]),
                    "rank": rank,
                    "chunk_id": chunk.identifier,
                    "document_id": chunk.document_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "cosine_similarity": similarity,
                    "scored": rank <= MAX_QUALITY_K,
                }
            )
        match_rows.extend(
            {
                **row,
                "matching_chunk_ids": json.dumps(row["matching_chunk_ids"]),
            }
            for row in score.matches
        )

    documents = [row for row in manifest.samples if row.get("kind") == "document"]
    source_tokens = sum(index.tokenizer.count(str(row["text"])) for row in documents)
    chunk_tokens = np.asarray([chunk.tokens for chunk in index.chunks], dtype=float)
    vector_bytes = int(index.vectors.nbytes) if index.vectors is not None else 0
    expected_vector_bytes = (
        len(index.chunks) * index.embedding_spec.dimension * np.dtype(np.float32).itemsize
    )
    if vector_bytes != expected_vector_bytes:
        raise RuntimeError(
            f"Embedding matrix byte mismatch: expected {expected_vector_bytes}, "
            f"received {vector_bytes}"
        )
    workload = {
        "workload.source_tokens": float(source_tokens),
        "workload.indexed_token_occurrences": float(chunk_tokens.sum()),
        "workload.document_count": float(len(documents)),
        "workload.answerable_query_count": float(len(questions)),
        "workload.unanswerable_query_count": float(
            sum(
                row.get("kind") == "question" and not row.get("answerable")
                for row in manifest.samples
            )
        ),
        "workload.chunk_count": float(len(index.chunks)),
        "workload.chunk_tokens_mean": float(chunk_tokens.mean()),
        "workload.chunk_tokens_p95": float(np.quantile(chunk_tokens, 0.95)),
        "workload.embedding_dimension": float(index.embedding_spec.dimension),
        "workload.embedding_matrix_bytes": float(vector_bytes),
    }
    validity = {
        "validity.expected_document_count": float(len(documents)),
        "validity.processed_document_count": float(len(documents)),
        "validity.expected_answerable_query_count": float(len(questions)),
        "validity.processed_answerable_query_count": float(len(samples)),
        "validity.truncated_input_count": 0.0,
        "validity.failed_query_count": float(len(failed_query_ids)),
        "validity.nonfinite_vector_count": 0.0,
        "validity.zero_norm_vector_count": 0.0,
        "validity.dimension_mismatch_count": 0.0,
        "validity.determinism_mismatch_count": float(determinism_mismatches),
        **eligible_counts(samples),
    }
    resamples = 0 if plan.profile == "smoke" else plan.bootstrap_resamples
    aggregate, intervals = aggregate_quality(
        samples, resamples=resamples, seed=plan.seed
    )
    intervals.update(latency_intervals(samples, resamples=resamples, seed=plan.seed))
    latency_sample_count = float(
        len({str(sample.metadata["document_id"]) for sample in samples})
    )
    aggregate.update(
        {
            "operational.query_latency_ms_p50.sample_count": latency_sample_count,
            "operational.query_latency_ms_p95.sample_count": latency_sample_count,
        }
    )
    aggregate.update(workload)
    aggregate.update(validity)
    operational = {
        "corpus_build_seconds": index.corpus_build_seconds,
        "corpus_build_source_tokens_per_second": source_tokens
        / max(index.corpus_build_seconds, 1e-9),
        "query_latency_ms_p50": (
            float(np.quantile(query_latencies, 0.50)) * 1000.0
            if query_latencies
            else 0.0
        ),
        "query_latency_ms_p95": (
            float(np.quantile(query_latencies, 0.95)) * 1000.0
            if query_latencies
            else 0.0
        ),
    }
    parameters = _parameters(
        candidate,
        chunker_name,
        embedding_name,
        entry,
        plan,
        device,
        dtype,
        manifest.split,
        manifest.checksum,
        manifest.fingerprint,
        index=index,
    )
    validation_errors = [*failures]
    if determinism_mismatches:
        validation_errors.append(
            f"{determinism_mismatches} query rankings changed across repetitions"
        )
    validation_report = {
        "candidate": candidate,
        "status": "failed" if validation_errors else "passed",
        "manifest_checksum": manifest.checksum,
        "manifest_fingerprint": manifest.fingerprint,
        "preflight": dict(index.preflight),
        "validity": validity,
        "errors": validation_errors,
        "warnings": [],
    }
    artifacts = {
        "chunk_manifest": [
            {
                "chunk_id": chunk.identifier,
                "document_id": chunk.document_id,
                "start": chunk.start,
                "end": chunk.end,
                "evaluation_tokens": chunk.tokens,
                "model_tokens": chunk.model_tokens,
                "chunker": chunker_name,
                "chunking_fingerprint": index.chunking_fingerprint,
                "chunking_contract": json.dumps(
                    index.chunking_contract, sort_keys=True
                ),
            }
            for chunk in index.chunks
        ],
        "query_metrics": query_rows,
        "retrievals": retrieval_rows,
        "evidence_matches": match_rows,
        "timings": timing_rows,
        "validation_report": validation_report,
    }
    if validation_errors:
        raise CandidateExecutionError(
            "; ".join(validation_errors),
            samples=tuple(samples),
            operational=operational,
            metrics={**workload, **validity},
            parameters=parameters,
            artifacts=artifacts,
        )
    return samples, operational, aggregate, parameters, intervals, artifacts


def run_in_fresh_process(
    candidate: str,
    manifest: DatasetManifest,
    model_lock: Mapping[str, Mapping[str, object]],
    plan: BenchmarkPlan,
    *,
    device: str,
    dtype: str,
):
    payload = {
        "candidate": candidate,
        "manifest": asdict(manifest),
        "model_lock": model_lock,
        "plan": asdict(plan),
        "device": device,
        "dtype": dtype,
    }
    result = run_json_worker(
        WORKER,
        payload,
        device=device,
        prefix="edumind-chunking-embedding-",
        error_label=f"chunking/embedding worker {candidate}",
    )
    status = str(result.get("status", ""))
    samples = tuple(_sample_from_payload(row) for row in result.get("samples", []))
    if status == "failed":
        raise CandidateExecutionError(
            str(result["error"]),
            parameters=_mapping(result.get("parameters")),
            artifacts=_mapping(result.get("artifacts")),
            samples=samples,
            metrics=_numeric_mapping(result.get("metrics")),
            intervals=_mapping(result.get("intervals")),
            operational=_numeric_mapping(result.get("operational")),
        )
    if status != "success":
        raise RuntimeError(f"Worker returned unsupported status {status!r}")
    return (
        list(samples),
        _numeric_mapping(result.get("operational")),
        _numeric_mapping(result.get("metrics")),
        _mapping(result.get("parameters")),
        _mapping(result.get("intervals")),
        _mapping(result.get("artifacts")),
    )


def execute_payload(payload: dict[str, object]) -> dict[str, object]:
    """JSON worker entry point."""

    candidate = str(payload["candidate"])
    manifest_payload = _mapping(payload["manifest"])
    manifest = DatasetManifest(
        **{
            **manifest_payload,
            "samples": tuple(
                dict(row) for row in manifest_payload.get("samples", [])
            ),
        }
    )
    plan_payload = _mapping(payload["plan"])
    plan = BenchmarkPlan(
        **{
            **plan_payload,
            "candidates": tuple(plan_payload.get("candidates", [])),
        }
    )
    model_lock = {
        str(name): dict(value)
        for name, value in _mapping(payload["model_lock"]).items()
        if isinstance(value, Mapping)
    }
    device, dtype = str(payload["device"]), str(payload["dtype"])
    try:
        evaluated = evaluate_candidate(
            candidate, manifest, model_lock, plan, device=device, dtype=dtype
        )
    except InputCompatibilityError as exc:
        chunker_name, embedding_name = _split_candidate(candidate)
        parameters = _parameters(
            candidate,
            chunker_name,
            embedding_name,
            model_lock[embedding_name],
            plan,
            device,
            dtype,
            manifest.split,
            manifest.checksum,
            manifest.fingerprint,
            preflight=exc.report,
        )
        report = {
            "candidate": candidate,
            "status": "failed",
            "reason_code": "input_length_exceeded",
            "manifest_checksum": manifest.checksum,
            "manifest_fingerprint": manifest.fingerprint,
            "preflight": exc.report,
            "errors": [str(exc)],
            "warnings": [],
        }
        return {
            "status": "failed",
            "error": str(exc),
            "parameters": parameters,
            "artifacts": {"validation_report": report},
        }
    except CandidateExecutionError as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "samples": [asdict(sample) for sample in exc.samples],
            "operational": exc.operational,
            "metrics": exc.metrics,
            "intervals": exc.intervals,
            "parameters": exc.parameters,
            "artifacts": exc.artifacts,
        }
    except Exception as exc:
        chunker_name, embedding_name = _split_candidate(candidate)
        error = f"{type(exc).__name__}: {exc}"
        parameters = _parameters(
            candidate,
            chunker_name,
            embedding_name,
            model_lock[embedding_name],
            plan,
            device,
            dtype,
            manifest.split,
            manifest.checksum,
            manifest.fingerprint,
        )
        return {
            "status": "failed",
            "error": error,
            "parameters": parameters,
            "artifacts": {
                "validation_report": {
                    "candidate": candidate,
                    "status": "failed",
                    "manifest_checksum": manifest.checksum,
                    "manifest_fingerprint": manifest.fingerprint,
                    "preflight": {},
                    "errors": [error],
                    "warnings": [],
                }
            },
        }
    return {
        "status": "success",
        "samples": [asdict(sample) for sample in evaluated[0]],
        "operational": evaluated[1],
        "metrics": evaluated[2],
        "parameters": evaluated[3],
        "intervals": evaluated[4],
        "artifacts": evaluated[5],
    }


def _parameters(
    candidate: str,
    chunker_name: str,
    embedding_name: str,
    entry: Mapping[str, object],
    plan: BenchmarkPlan,
    device: str,
    dtype: str,
    split: str,
    manifest_checksum: str,
    manifest_fingerprint: str,
    *,
    index=None,
    preflight: Mapping[str, object] | None = None,
) -> dict[str, object]:
    resolved_spec = (
        index.embedding_spec
        if index is not None
        else embedding_spec(
            embedding_name,
            revision=str(entry["revision"]),
            local_path=str(entry["model_path"]),
            document_device=device,
            query_device=device,
        )
    )
    embedding_contract = (
        asdict(index.embedding_spec)
        if index is not None and hasattr(index.embedding_spec, "__dataclass_fields__")
        else dict(vars(index.embedding_spec))
        if index is not None
        else asdict(resolved_spec)
    )
    chunking_contract = (
        dict(index.chunking_contract)
        if index is not None
        else dict((preflight or {}).get("chunking_contract", {}))
        if isinstance((preflight or {}).get("chunking_contract"), Mapping)
        else {}
    )
    return {
        "candidate": candidate,
        "chunker": chunker_name,
        "chunking_fingerprint": (
            index.chunking_fingerprint
            if index is not None
            else chunking_contract.get("fingerprint", "")
        ),
        "chunking_contract": chunking_contract,
        "embedding": embedding_name,
        "embedding_contract": embedding_contract,
        "embedding_batch_size": (
            getattr(index.embedder, "batch_size", 32)
            if index is not None and index.embedder is not None
            else 32
        ),
        "model_revision": entry.get("revision"),
        "model_path": entry.get("model_path"),
        "model_cache_manifest_sha256": entry.get("model_cache_manifest_sha256"),
        "tokenizer_revision": entry.get("revision"),
        "tokenizer_path": entry.get("model_path"),
        "tokenizer_cache_manifest_sha256": entry.get(
            "model_cache_manifest_sha256"
        ),
        "query_prefix": embedding_contract.get("query_prefix", ""),
        "document_prefix": embedding_contract.get("document_prefix", ""),
        "query_prompt_name": embedding_contract.get("query_prompt_name"),
        "document_prompt_name": embedding_contract.get("document_prompt_name"),
        "pooling": embedding_contract.get("pooling"),
        "normalize": embedding_contract.get("normalize"),
        "device": device,
        "model_dtype": dtype,
        "stored_vector_dtype": "float32",
        "search_matrix_normalization": "l2",
        "exact_search": "numpy-cosine-stable-corpus-order-v1",
        "evaluation_tokenizer": "tiktoken:cl100k_base",
        "evidence_coverage_rule": "single-chunk-complete-unit-v1",
        "quality_cutoffs": QUALITY_CUTOFFS,
        "maximum_scored_rank": MAX_QUALITY_K,
        "artifact_top_k": AUDIT_K,
        "alpha": ALPHA,
        "seed": plan.seed,
        "warmups": plan.warmups,
        "repetitions": plan.repetitions,
        "split": split,
        "manifest_checksum": manifest_checksum,
        "manifest_fingerprint": manifest_fingerprint,
        "resolved_query_input": dict(
            (index.preflight if index is not None else preflight or {}).get(
                "resolved_query_input", {}
            )
        ),
        "resolved_document_input": dict(
            (index.preflight if index is not None else preflight or {}).get(
                "resolved_document_input", {}
            )
        ),
        "package_versions": package_versions(
            ("numpy", "sentence-transformers", "tiktoken", "torch", "transformers")
        ),
    }


def _answerable_questions(manifest: DatasetManifest) -> list[Mapping[str, object]]:
    return [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("answerable")
    ]


def _split_candidate(candidate: str) -> tuple[str, str]:
    values = candidate.split("|", 1)
    if len(values) != 2 or not all(values):
        raise ValueError(f"Malformed chunker/embedding candidate: {candidate}")
    return values[0], values[1]


def _sample_from_payload(payload: object) -> SampleResult:
    row = _mapping(payload)
    return SampleResult(
        str(row["sample_id"]),
        _numeric_mapping(row["metrics"]),
        float(row["latency_seconds"]),
        _mapping(row.get("metadata")),
    )


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _numeric_mapping(value: object) -> dict[str, float]:
    return {
        str(name): float(number)
        for name, number in _mapping(value).items()
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass
