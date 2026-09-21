"""Fresh-process benchmark contract for chunker/embedding pair selection."""

from __future__ import annotations

import json
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
from experiments.benchmarks.common.datasets import answerable_questions, evidence_units
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.common.protocol import validate_execution
from experiments.benchmarks.common.process import (
    benchmark_objects_from_payload,
    candidate_execution_payload,
    decode_worker_result,
    run_json_worker,
    seed_deterministically,
    successful_execution_payload,
)
from experiments.benchmarks.rag.evaluation import (
    InputCompatibilityError,
    build_index,
    dense_rank_with_scores,
)

from .metrics import (
    aggregate_quality,
    eligible_counts,
    latency_intervals,
    prefixed_quality,
    score_question,
)
from .profiles import embedding_spec, split_candidate
from .protocol import ChunkingEmbeddingProtocol, protocol_from_settings


OPERATIONAL_DIRECTIONS = {
    "operational.corpus_build_seconds": "min",
    "operational.corpus_build_source_tokens_per_second": "max",
    "operational.query_latency_ms_p50": "min",
    "operational.query_latency_ms_p95": "min",
    "operational.peak_process_tree_ram_mb": "min",
    "operational.peak_vram_mb": "min",
}
WORKER = Path(__file__).with_name("worker.py")


def directions_for(
    manifest: DatasetManifest,
    protocol: ChunkingEmbeddingProtocol,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Declare only the evidence slices present in this frozen manifest."""

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
    quality_directions = {
        f"quality.{scope}.{metric}": "max"
        for scope in ("overall", "text", "table", "formula", "mixed")
        for metric in protocol.quality_metrics
    }
    alpha_metrics = set(protocol.alpha_ndcg_metrics)
    quality = {}
    for name, direction in quality_directions.items():
        _, scope, metric = name.split(".", 2)
        if scope not in scopes:
            continue
        if metric in alpha_metrics and scope not in alpha_scopes:
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

    protocol = protocol_from_settings(plan.settings)
    validate_execution(
        protocol.meta,
        plan.profile,
        seed=plan.seed,
        warmups=plan.warmups,
        repetitions=plan.repetitions,
        bootstrap_resamples=plan.bootstrap_resamples,
        device=device,
        dtype=dtype,
        batch_size=protocol.embedding_batch_size,
    )
    seed_deterministically(plan.seed)
    chunker_name, embedding_name = split_candidate(candidate)
    entry = model_lock[embedding_name]
    index = build_index(
        manifest,
        chunker_name,
        embedding_name,
        model_lock,
        with_dense=True,
        with_bm25=False,
        device=device,
        dtype=dtype,
        chunking_protocol=protocol,
    )
    questions = answerable_questions(manifest)
    random.Random(plan.seed).shuffle(questions)
    if not questions:
        raise RuntimeError("Chunking/embedding requires answerable questions")
    for _ in range(plan.warmups):
        dense_rank_with_scores(
            index, str(questions[0]["question"]), protocol.audit_depth
        )

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
                    index, str(question["question"]), protocol.audit_depth
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
        maximum_quality_rank = max(protocol.quality_cutoffs)
        first_ids = [position for position, _ in rankings[0]][:maximum_quality_rank]
        deterministic = all(
            [position for position, _ in ranking][:maximum_quality_rank] == first_ids
            for ranking in rankings[1:]
        )
        if not deterministic:
            determinism_mismatches += 1
        latency = float(np.median(latencies))
        query_latencies.append(latency)
        selected = [index.chunks[position] for position, _ in rankings[0]]
        score = score_question(
            question,
            selected,
            index.chunks,
            index.tokenizer,
            cutoffs=protocol.quality_cutoffs,
            alpha=protocol.alpha_ndcg_alpha,
        )
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
                    f"deterministic_top_{maximum_quality_rank}": deterministic,
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
                    "scored": rank <= maximum_quality_rank,
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
        **eligible_counts(samples, alpha_metrics=protocol.alpha_ndcg_metrics),
    }
    resamples = 0 if plan.profile == "smoke" else plan.bootstrap_resamples
    aggregate, intervals = aggregate_quality(
        samples,
        resamples=resamples,
        seed=plan.seed,
        confidence=protocol.confidence_level,
    )
    intervals.update(
        latency_intervals(
            samples,
            resamples=resamples,
            seed=plan.seed,
            minimum_documents=protocol.minimum_latency_ci_documents,
            confidence=protocol.confidence_level,
        )
    )
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
        protocol=protocol,
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
    return decode_worker_result(result)


def execute_payload(payload: dict[str, object]) -> dict[str, object]:
    """JSON worker entry point."""

    candidate, manifest, model_lock, plan = benchmark_objects_from_payload(payload)
    device, dtype = str(payload["device"]), str(payload["dtype"])
    protocol = protocol_from_settings(plan.settings)
    try:
        evaluated = evaluate_candidate(
            candidate, manifest, model_lock, plan, device=device, dtype=dtype
        )
    except InputCompatibilityError as exc:
        chunker_name, embedding_name = split_candidate(candidate)
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
            protocol=protocol,
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
        return candidate_execution_payload(exc)
    except Exception as exc:
        chunker_name, embedding_name = split_candidate(candidate)
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
            protocol=protocol,
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
    return successful_execution_payload(evaluated)


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
    protocol: ChunkingEmbeddingProtocol,
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
            getattr(index.embedder, "batch_size", protocol.embedding_batch_size)
            if index is not None and index.embedder is not None
            else protocol.embedding_batch_size
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
        "stored_vector_dtype": protocol.stored_vector_dtype,
        "search_matrix_normalization": "l2",
        "exact_search": "numpy-cosine-stable-corpus-order-v1",
        "evaluation_tokenizer": protocol.tokenizer,
        "evidence_coverage_rule": "single-chunk-complete-unit-v1",
        "quality_cutoffs": protocol.quality_cutoffs,
        "maximum_scored_rank": max(protocol.quality_cutoffs),
        "artifact_top_k": protocol.audit_depth,
        "alpha": protocol.alpha_ndcg_alpha,
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
