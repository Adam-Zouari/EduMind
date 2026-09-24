"""Fresh-process execution for complete retrieval/reranking candidates."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.datasets import answerable_questions
from experiments.benchmarks.common.process import (
    benchmark_objects_from_payload,
    candidate_execution_payload,
    decode_worker_result,
    payload_mapping,
    run_json_worker,
    seed_deterministically,
    successful_execution_payload,
)
from experiments.benchmarks.common.protocol import PreflightSettings
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.rag.chunking_embedding.metrics import (
    aggregate_quality,
    eligible_counts,
    prefixed_quality,
    score_question,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import split_candidate
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    protocol_from_settings as chunking_protocol_from_settings,
)
from experiments.benchmarks.rag.evaluation import (
    ExactIndex,
    InputCompatibilityError,
    build_index,
    dense_rank_with_scores,
)
from experiments.benchmarks.rag.methods import (
    Reranker,
    reciprocal_rank_fusion_with_scores,
)

from .metrics import (
    latency_intervals,
    latency_summary,
    pool_evidence_unit_recall,
    retrieved_tokens,
)
from .profiles import RetrievalCandidate, parse_candidate
from .protocol import RetrievalProtocol, protocol_from_settings

WORKER = Path(__file__).with_name("worker.py")


def evaluate_candidate(
    candidate_name: str,
    manifest: DatasetManifest,
    model_lock: Mapping[str, Mapping[str, object]],
    plan: BenchmarkPlan,
    *,
    device: str,
    dtype: str,
    frozen_pool_collection: Mapping[str, object] | None,
    child_run_id: str | None,
    worker_started_at: float | None = None,
):
    """Evaluate one complete stack and return auditable rows and aggregates."""

    protocol = protocol_from_settings(plan.settings)
    chunking_protocol = chunking_protocol_from_settings(plan.settings)
    protocol.validate_execution(
        plan.profile,
        seed=plan.seed,
        warmups=plan.warmups,
        repetitions=plan.repetitions,
        bootstrap_resamples=plan.bootstrap_resamples,
        device=device,
        dtype=dtype,
    )
    seed_deterministically(plan.seed)
    candidate = parse_candidate(candidate_name)
    chunker_name, embedding_name = split_candidate(
        str(plan.settings.get("chunker_embedding", ""))
    )
    questions = answerable_questions(manifest)
    random.Random(plan.seed).shuffle(questions)
    if not questions:
        raise RuntimeError("Retrieval/reranking requires answerable questions")

    initialization_started = (
        worker_started_at if worker_started_at is not None else time.perf_counter()
    )
    index = build_index(
        manifest,
        chunker_name,
        embedding_name,
        model_lock,
        with_dense=candidate.retriever in {"dense", "rrf"},
        with_bm25=candidate.retriever in {"bm25", "rrf"},
        device=device,
        dtype=dtype,
        chunking_protocol=chunking_protocol,
        retrieval_protocol=protocol,
    )
    if (
        plan.profile == "smoke"
        and len(index.chunks) != protocol.smoke_expected_chunk_count
    ):
        raise RuntimeError(
            "Retrieval smoke fixture must produce exactly "
            f"{protocol.smoke_expected_chunk_count} chunks, got {len(index.chunks)}"
        )
    reranker = _reranker(candidate, model_lock, protocol, device=device, dtype=dtype)
    if reranker is not None:
        reranker.prepare()
    initialization_elapsed = time.perf_counter() - initialization_started
    preflight_seconds = float(
        index.preflight.get("input_preflight_seconds_excluded_from_corpus_build", 0.0)
    )
    cold_initialization_seconds = max(
        0.0,
        initialization_elapsed - index.corpus_build_seconds - preflight_seconds,
    )
    index_identity = _index_identity(index, candidate)
    index_build = _index_build_artifact(index, candidate, index_identity)

    pool_by_question, pool_collection_metadata = _load_pool_collection(
        candidate,
        frozen_pool_collection,
        questions,
        index,
        protocol,
        index_identity=index_identity,
    )
    reranker_input_totals = _reranker_preflight(
        reranker, questions, pool_by_question, index
    )

    for warmup in range(plan.warmups):
        question = questions[warmup % len(questions)]
        live_pool = _first_stage(
            index, str(question["question"]), candidate.retriever, protocol
        )
        if reranker is not None:
            frozen = pool_by_question[str(question["id"])]
            _require_same_pool(live_pool, frozen, str(question["id"]))
            reranker.rank_with_scores(
                str(question["question"]),
                [index.chunks[position].text for position, _ in frozen],
                validate_inputs=False,
            )

    samples: list[SampleResult] = []
    query_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    pool_collection_rows: list[dict[str, object]] = []
    failures: list[str] = []
    failed_query_ids: set[str] = set()
    nonfinite_score_count = 0
    permutation_failures = 0
    query_pool_mismatches = 0
    expected_agreements = len(questions) * max(0, plan.repetitions - 1)
    matching_agreements = 0
    actual_pool_sizes: list[int] = []
    retrieved_by_cutoff = {cutoff: [] for cutoff in protocol.quality_cutoffs}

    for question in questions:
        question_id = str(question["id"])
        repetition_orders: list[list[int] | None] = []
        successful_latencies: list[float] = []
        first_stage_latencies: list[float] = []
        reranker_latencies: list[float] = []
        first_output: list[tuple[int, float]] | None = None
        first_pool: list[tuple[int, float]] | None = None
        for repetition in range(plan.repetitions):
            full_started = time.perf_counter()
            first_stage_ms: float | None = None
            reranker_ms: float | None = None
            try:
                first_started = time.perf_counter()
                live_pool = _first_stage(
                    index,
                    str(question["question"]),
                    candidate.retriever,
                    protocol,
                )
                first_stage_ms = (time.perf_counter() - first_started) * 1000.0
                _require_finite_scores(live_pool)
                quality_pool = live_pool
                if reranker is not None:
                    quality_pool = pool_by_question[question_id]
                    try:
                        _require_same_pool(live_pool, quality_pool, question_id)
                    except ValueError:
                        query_pool_mismatches += 1
                        raise
                    rerank_started = time.perf_counter()
                    local_order = reranker.rank_with_scores(
                        str(question["question"]),
                        [index.chunks[position].text for position, _ in quality_pool],
                        validate_inputs=False,
                    )
                    reranker_ms = (time.perf_counter() - rerank_started) * 1000.0
                    final = [
                        (quality_pool[position][0], score)
                        for position, score in local_order
                    ]
                    try:
                        _require_permutation(final, quality_pool, question_id)
                    except ValueError:
                        permutation_failures += 1
                        raise
                else:
                    final = quality_pool
                _require_finite_scores(final)
                full_seconds = time.perf_counter() - full_started
                order = [position for position, _ in final]
                repetition_orders.append(order)
                successful_latencies.append(full_seconds)
                first_stage_latencies.append(first_stage_ms)
                if reranker_ms is not None:
                    reranker_latencies.append(reranker_ms)
                if first_output is None:
                    first_output = list(final)
                    first_pool = list(quality_pool)
                timing_rows.append(
                    {
                        "question_id": question_id,
                        "repetition": repetition + 1,
                        "full_stack_latency_ms": full_seconds * 1000.0,
                        "first_stage_latency_ms": first_stage_ms,
                        "reranker_latency_ms": reranker_ms,
                        "success": True,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - record per-attempt failures
                full_seconds = time.perf_counter() - full_started
                if "non-finite" in str(exc).casefold():
                    nonfinite_score_count += 1
                repetition_orders.append(None)
                failed_query_ids.add(question_id)
                error = f"{type(exc).__name__}: {exc}"
                failures.append(f"{question_id} repetition {repetition + 1}: {error}")
                timing_rows.append(
                    {
                        "question_id": question_id,
                        "repetition": repetition + 1,
                        "full_stack_latency_ms": full_seconds * 1000.0,
                        "first_stage_latency_ms": first_stage_ms,
                        "reranker_latency_ms": reranker_ms,
                        "success": False,
                        "error": error,
                    }
                )

        designated = repetition_orders[0] if repetition_orders else None
        matching_agreements += sum(
            order is not None and designated is not None and order == designated
            for order in repetition_orders[1:]
        )
        if any(order is None for order in repetition_orders):
            continue
        if first_output is None or first_pool is None:
            continue
        final_chunks = [index.chunks[position] for position, _ in first_output]
        pool_chunks = [index.chunks[position] for position, _ in first_pool]
        score = score_question(
            question,
            final_chunks,
            index.chunks,
            index.tokenizer,
            cutoffs=protocol.quality_cutoffs,
            alpha=protocol.alpha_ndcg_alpha,
        )
        evidence_type = str(question["evidence_type"])
        sample_metrics = prefixed_quality(score.metrics, evidence_type)
        if reranker is None:
            pool_recall = pool_evidence_unit_recall(question, pool_chunks)
            sample_metrics[protocol.pool_recall_metric] = pool_recall
        median_latency = float(np.median(successful_latencies))
        median_first_stage = float(np.median(first_stage_latencies))
        median_reranker = (
            float(np.median(reranker_latencies)) if reranker_latencies else None
        )
        samples.append(
            SampleResult(
                question_id,
                sample_metrics,
                median_latency,
                {
                    "document_id": str(question["document_id"]),
                    "evidence_type": evidence_type,
                    "first_stage_latency_ms": median_first_stage,
                    "reranker_latency_ms": median_reranker,
                },
            )
        )
        actual_pool_sizes.append(len(first_pool))
        for cutoff in protocol.quality_cutoffs:
            retrieved_by_cutoff[cutoff].append(retrieved_tokens(final_chunks, cutoff))
        query_rows.append(
            {
                "question_id": question_id,
                "document_id": str(question["document_id"]),
                "evidence_type": evidence_type,
                "full_stack_latency_ms": median_latency * 1000.0,
                "first_stage_latency_ms": median_first_stage,
                "reranker_latency_ms": median_reranker,
                **score.metrics,
                **(
                    {
                        protocol.pool_recall_metric.removeprefix("diagnostic."): (
                            pool_recall
                        )
                    }
                    if reranker is None
                    else {}
                ),
            }
        )
        pool_score_by_position = dict(first_pool)
        for rank, (position, score_value) in enumerate(first_output, start=1):
            chunk = index.chunks[position]
            ranking_rows.append(
                {
                    "question_id": question_id,
                    "rank": rank,
                    "chunk_id": chunk.identifier,
                    "document_id": chunk.document_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "first_stage_score": pool_score_by_position[position],
                    "final_score": score_value,
                    "scored": rank <= max(protocol.quality_cutoffs),
                }
            )
        match_rows.extend(
            {
                **row,
                "matching_chunk_ids": json.dumps(row["matching_chunk_ids"]),
            }
            for row in score.matches
        )
        if reranker is None:
            for rank, (position, score_value) in enumerate(first_pool, start=1):
                chunk = index.chunks[position]
                pool_collection_rows.append(
                    {
                        "question_id": question_id,
                        "rank": rank,
                        "chunk_id": chunk.identifier,
                        "score": score_value,
                    }
                )

    resamples = plan.bootstrap_resamples
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
    documents = [row for row in manifest.samples if row.get("kind") == "document"]
    validity = {
        "validity.ranking_agreement": (
            matching_agreements / expected_agreements
            if expected_agreements
            else float(not failures)
        ),
        "validity.expected_query_count": float(len(questions)),
        "validity.processed_query_count": float(len(samples)),
        "validity.failed_query_count": float(len(failed_query_ids)),
        "validity.truncated_input_count": 0.0,
        "validity.nonfinite_score_count": float(nonfinite_score_count),
        "validity.pool_collection_match": float(query_pool_mismatches == 0),
        "validity.exact_pool_permutation": float(permutation_failures == 0),
        **eligible_counts(samples, alpha_metrics=protocol.alpha_ndcg_metrics),
    }
    workload = {
        "workload.corpus_document_count": float(len(documents)),
        "workload.answerable_query_count": float(len(questions)),
        "workload.chunk_count": float(len(index.chunks)),
        "workload.requested_pool_size": float(protocol.pool_size),
        "workload.actual_pool_size_mean": _mean(actual_pool_sizes),
        "workload.actual_pool_size_min": float(min(actual_pool_sizes, default=0)),
        "workload.short_pool_query_count": float(
            sum(size < protocol.pool_size for size in actual_pool_sizes)
        ),
        **{
            f"workload.retrieved_tokens_at_{cutoff}_{suffix}": value
            for cutoff, values in retrieved_by_cutoff.items()
            for suffix, value in (
                ("mean", _mean(values)),
                ("p95", _quantile(values, 0.95)),
            )
        },
    }
    if reranker_input_totals:
        workload.update(
            {
                "workload.reranker_input_tokens_per_query_mean": _mean(
                    reranker_input_totals
                ),
                "workload.reranker_input_tokens_per_query_p95": _quantile(
                    reranker_input_totals, 0.95
                ),
            }
        )
    storage = _storage_metrics(index, candidate, reranker)
    aggregate.update(validity)
    aggregate.update(workload)
    aggregate.update(storage)
    operational = {
        **(latency_summary(samples) if samples else {}),
        "cold_initialization_seconds": cold_initialization_seconds,
    }
    if reranker is None:
        operational["index_build_seconds"] = index.corpus_build_seconds

    pool_collection_checksum = (
        stable_hash(pool_collection_rows)
        if reranker is None
        else str(pool_collection_metadata["pool_collection_checksum"])
    )
    parameters = _parameters(
        candidate,
        chunker_name,
        embedding_name,
        model_lock,
        index,
        reranker,
        plan,
        protocol,
        device=device,
        dtype=dtype,
        manifest=manifest,
        pool_collection_checksum=pool_collection_checksum,
        index_identity=index_identity,
        pool_owner_child_run_id=(
            child_run_id
            if reranker is None
            else pool_collection_metadata["owner_child_run_id"]
        ),
    )
    validation_errors = list(failures)
    if validity["validity.ranking_agreement"] != 1.0:
        validation_errors.append(
            f"repeated top-{protocol.pool_size} rankings did not agree exactly"
        )
    if query_pool_mismatches:
        validation_errors.append(
            "live first-stage output did not match the frozen pool collection"
        )
    if permutation_failures:
        validation_errors.append("reranker output was not an exact pool permutation")
    validation_errors.extend(
        _candidate_contract_errors(candidate, aggregate, operational, protocol)
    )
    validation_report = {
        "candidate": candidate.identifier,
        "status": "failed" if validation_errors else "passed",
        "manifest_checksum": manifest.checksum,
        "manifest_fingerprint": manifest.fingerprint,
        "pool_owner": candidate.owner_identifier,
        "pool_owner_child_run_id": parameters["pool_owner_child_run_id"],
        "pool_collection_checksum": pool_collection_checksum,
        **index_identity,
        "preflight": dict(index.preflight),
        "validity": validity,
        "errors": validation_errors,
        "warnings": [],
    }
    artifacts: dict[str, object] = {
        "query_metrics": query_rows,
        "rankings": ranking_rows,
        "evidence_matches": match_rows,
        "timings": timing_rows,
        "validation_report": validation_report,
    }
    if reranker is None:
        artifacts["pool_collection"] = pool_collection_rows
        artifacts["index_build"] = index_build
    if validation_errors:
        raise CandidateExecutionError(
            "; ".join(validation_errors),
            samples=tuple(samples),
            operational=operational,
            metrics=aggregate,
            parameters=parameters,
            intervals=intervals,
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
    frozen_pool_collection: Mapping[str, object] | None,
    child_run_id: str | None,
):
    payload = {
        "candidate": candidate,
        "manifest": asdict(manifest),
        "model_lock": model_lock,
        "plan": asdict(plan),
        "device": device,
        "dtype": dtype,
        "frozen_pool_collection": frozen_pool_collection,
        "child_run_id": child_run_id,
    }
    result = run_json_worker(
        WORKER,
        payload,
        device=device,
        prefix="edumind-retrieval-reranking-",
        error_label=f"retrieval/reranking worker {candidate}",
    )
    return decode_worker_result(result)


def preflight_in_fresh_process(
    candidate: str,
    manifest: DatasetManifest,
    model_lock: Mapping[str, Mapping[str, object]],
    plan: BenchmarkPlan,
    *,
    vram_limit_mb: float,
    preflight: PreflightSettings,
) -> dict[str, object]:
    chunker, _embedding = split_candidate(str(plan.settings["chunker_embedding"]))
    model_backed = parse_candidate(candidate).model_backed or chunker == "semantic"
    result = run_json_worker(
        WORKER,
        {
            "candidate": candidate,
            "manifest": asdict(manifest),
            "model_lock": model_lock,
            "plan": asdict(plan),
            "device": "cuda",
            "dtype": "float16",
            "mode": "preflight",
        },
        device="cuda",
        prefix="edumind-retrieval-reranking-preflight-",
        error_label=f"retrieval/reranking preflight worker {candidate}",
        vram_limit_mb=vram_limit_mb if model_backed else None,
        require_vram_measurement=model_backed,
        telemetry_interval_seconds=preflight.telemetry_interval_seconds,
        poll_interval_seconds=preflight.poll_interval_seconds,
        timeout_seconds=preflight.worker_timeout_seconds,
    )
    supervision = result.pop("_worker_supervision", {})
    if isinstance(supervision, Mapping):
        result.update(supervision)
    if not model_backed:
        result.setdefault("vram_measurement_method", "not-applicable")
    return result


def execute_payload(payload: dict[str, object]) -> dict[str, object]:
    candidate, manifest, model_lock, plan = benchmark_objects_from_payload(payload)
    if payload.get("mode") == "preflight":
        return _preflight_candidate(candidate, manifest, model_lock, plan)
    try:
        evaluated = evaluate_candidate(
            candidate,
            manifest,
            model_lock,
            plan,
            device=str(payload["device"]),
            dtype=str(payload["dtype"]),
            frozen_pool_collection=(
                payload_mapping(payload.get("frozen_pool_collection"))
                if payload.get("frozen_pool_collection") is not None
                else None
            ),
            child_run_id=(
                str(payload["child_run_id"])
                if payload.get("child_run_id") is not None
                else None
            ),
            worker_started_at=(
                float(payload["worker_started_at"])
                if payload.get("worker_started_at") is not None
                else None
            ),
        )
    except CandidateExecutionError as exc:
        return candidate_execution_payload(exc)
    except InputCompatibilityError as exc:
        return _unexpected_failure(candidate, manifest, str(exc), preflight=exc.report)
    except Exception as exc:  # noqa: BLE001 - worker reports candidate failure
        return _unexpected_failure(candidate, manifest, f"{type(exc).__name__}: {exc}")
    return successful_execution_payload(evaluated)


def _preflight_candidate(candidate_name, manifest, model_lock, plan):
    protocol = protocol_from_settings(plan.settings)
    chunking_protocol = chunking_protocol_from_settings(plan.settings)
    seed_deterministically(plan.seed)
    candidate = parse_candidate(candidate_name)
    chunker_name, embedding_name = split_candidate(
        str(plan.settings["chunker_embedding"])
    )
    index = build_index(
        manifest,
        chunker_name,
        embedding_name,
        model_lock,
        with_dense=candidate.retriever in {"dense", "rrf"},
        with_bm25=candidate.retriever in {"bm25", "rrf"},
        device="cuda",
        dtype="float16",
        chunking_protocol=chunking_protocol,
        retrieval_protocol=protocol,
    )
    questions = answerable_questions(manifest)
    if not questions:
        raise RuntimeError("Retrieval preflight requires an answerable question")
    question = questions[0]
    pool = _first_stage(index, str(question["question"]), candidate.retriever, protocol)
    reranker = _reranker(
        candidate, model_lock, protocol, device="cuda", dtype="float16"
    )
    reports = []
    placement_before = []
    placement_after = []
    embedding_before = index.preflight.get("placement")
    if isinstance(embedding_before, Mapping):
        reports.append(embedding_before)
        placement_before.append(dict(embedding_before))
    if index.embedder is not None:
        embedding_after = index.embedder.placement_report("cuda")
        reports.append(embedding_after)
        placement_after.append(embedding_after)
    if reranker is not None:
        reranker.prepare()
        reranker_before = reranker.placement_report()
        reports.append(reranker_before)
        placement_before.append(reranker_before)
        counts = reranker.input_token_counts(
            str(question["question"]),
            [index.chunks[position].text for position, _ in pool],
        )
        if any(count > reranker.maximum_length for count in counts):
            raise InputCompatibilityError(
                f"{reranker.model_name} exceeds its input contract",
                {
                    "reason_code": "input_length_exceeded",
                    "maximum_length": reranker.maximum_length,
                    "input_token_counts": counts,
                },
            )
        reranker.rank_with_scores(
            str(question["question"]),
            [index.chunks[position].text for position, _ in pool],
            validate_inputs=False,
        )
        reranker_after = reranker.placement_report()
        reports.append(reranker_after)
        placement_after.append(reranker_after)
    return {
        "placement": _combined_placement(reports),
        "placement_before": placement_before,
        "placement_after": placement_after,
        "input_validation": dict(index.preflight),
        "stress_question_id": str(question["id"]),
        "stress_pool_size": len(pool),
    }


def _combined_placement(reports):
    if not reports:
        return {
            "status": "qualified",
            "verification": "not-applicable-no-model-weights",
            "observed_devices": [],
        }
    statuses = {str(report.get("status")) for report in reports}
    if "offload_detected" in statuses:
        status = "offload_detected"
    elif "wrong_device" in statuses:
        status = "wrong_device"
    elif "placement_unverifiable" in statuses:
        status = "placement_unverifiable"
    else:
        status = "qualified"
    return {"status": status, "components": [dict(report) for report in reports]}


def _first_stage(
    index: ExactIndex,
    query: str,
    retriever: str,
    protocol: RetrievalProtocol,
) -> list[tuple[int, float]]:
    if retriever == "dense":
        return dense_rank_with_scores(index, query, protocol.pool_size)
    if index.bm25 is None:
        raise RuntimeError("BM25 state is unavailable")
    lexical = index.bm25.rank(query, protocol.pool_size)
    if retriever == "bm25":
        return lexical
    dense = dense_rank_with_scores(index, query, protocol.pool_size)
    return reciprocal_rank_fusion_with_scores(
        [[position for position, _ in dense], [position for position, _ in lexical]],
        protocol.pool_size,
        protocol.rrf_constant,
        weights=protocol.rrf_weights,
        tie_keys={
            position: chunk.identifier for position, chunk in enumerate(index.chunks)
        },
    )


def _reranker(
    candidate: RetrievalCandidate,
    model_lock: Mapping[str, Mapping[str, object]],
    protocol: RetrievalProtocol,
    *,
    device: str,
    dtype: str,
) -> Reranker | None:
    if candidate.reranker_model is None:
        return None
    entry = model_lock[candidate.reranker_model]
    return Reranker(
        candidate.reranker_model,
        str(entry["revision"]),
        str(entry["model_path"]),
        device=device,
        dtype=dtype,
        batch_size=protocol.reranker_batch_size,
        maximum_length=protocol.maximum_tokens(candidate.reranker),
    )


def _load_pool_collection(
    candidate: RetrievalCandidate,
    frozen_pool_collection: Mapping[str, object] | None,
    questions: Sequence[Mapping[str, object]],
    index: ExactIndex,
    protocol: RetrievalProtocol,
    *,
    index_identity: Mapping[str, str],
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, object]]:
    if candidate.reranker == "none":
        if frozen_pool_collection is not None:
            raise ValueError(
                "Pool-owner candidates cannot consume a frozen pool collection"
            )
        return {}, {}
    if frozen_pool_collection is None:
        raise ValueError(
            f"{candidate.identifier} requires its owner's frozen pool collection"
        )
    rows = frozen_pool_collection.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Frozen pool collection rows are missing")
    normalized_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(normalized_rows) != len(rows):
        raise ValueError("Frozen pool collection contains malformed rows")
    checksum = str(frozen_pool_collection.get("pool_collection_checksum", ""))
    if not checksum or stable_hash(normalized_rows) != checksum:
        raise ValueError("Frozen pool collection checksum mismatch")
    if frozen_pool_collection.get("owner_candidate") != candidate.owner_identifier:
        raise ValueError(
            "Frozen pool collection belongs to a different first-stage retriever"
        )
    observed_identity = {
        key: str(frozen_pool_collection[key])
        for key in _INDEX_IDENTITY_FIELDS
        if key in frozen_pool_collection
    }
    if observed_identity.keys() != index_identity.keys():
        missing = sorted(index_identity.keys() - observed_identity.keys())
        unexpected = sorted(observed_identity.keys() - index_identity.keys())
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            "Frozen pool collection index identity is invalid: " + "; ".join(details)
        )
    mismatched = sorted(
        key for key, value in index_identity.items() if observed_identity[key] != value
    )
    if mismatched:
        raise ValueError(
            "Frozen pool collection index identity mismatch: " + ", ".join(mismatched)
        )
    owner_child_run_id = frozen_pool_collection.get("owner_child_run_id")
    if owner_child_run_id is not None and not str(owner_child_run_id):
        raise ValueError("Frozen pool collection has an invalid owner child run ID")
    positions = {
        chunk.identifier: position for position, chunk in enumerate(index.chunks)
    }
    by_question: dict[str, list[tuple[int, float]]] = {}
    for row in normalized_rows:
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id not in positions:
            raise ValueError(
                f"Frozen pool collection references unknown chunk {chunk_id!r}"
            )
        by_question.setdefault(str(row.get("question_id", "")), []).append(
            (positions[chunk_id], float(row["score"]))
        )
    expected = {str(question["id"]) for question in questions}
    if set(by_question) != expected:
        raise ValueError(
            "Frozen pool collection does not account for every answerable question"
        )
    for question_id, values in by_question.items():
        if not values or len(values) > protocol.pool_size:
            raise ValueError(f"Frozen query pool has an invalid size for {question_id}")
        ranks = [
            int(row["rank"])
            for row in normalized_rows
            if str(row.get("question_id", "")) == question_id
        ]
        if ranks != list(range(1, len(values) + 1)):
            raise ValueError(f"Frozen query pool has invalid ranks for {question_id}")
        identifiers = [position for position, _ in values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Frozen query pool duplicates chunks for {question_id}")
    return by_question, {
        "pool_collection_checksum": checksum,
        "owner_child_run_id": owner_child_run_id,
    }


def _reranker_preflight(
    reranker: Reranker | None,
    questions: Sequence[Mapping[str, object]],
    pools: Mapping[str, Sequence[tuple[int, float]]],
    index: ExactIndex,
) -> list[int]:
    if reranker is None:
        return []
    totals: list[int] = []
    for question in questions:
        positions = pools[str(question["id"])]
        counts = reranker.input_token_counts(
            str(question["question"]),
            [index.chunks[position].text for position, _ in positions],
        )
        oversized = [count for count in counts if count > reranker.maximum_length]
        if oversized:
            raise InputCompatibilityError(
                f"{reranker.model_name} has inputs beyond its "
                f"{reranker.maximum_length}-token contract",
                {
                    "status": "failed",
                    "reason_code": "input_length_exceeded",
                    "maximum_length": reranker.maximum_length,
                    "question_id": str(question["id"]),
                    "maximum_input_tokens": max(oversized),
                    "offending_input_count": len(oversized),
                    "truncated_inputs": 0,
                },
            )
        totals.append(sum(counts))
    return totals


def _require_same_pool(
    observed: Sequence[tuple[int, float]],
    expected: Sequence[tuple[int, float]],
    question_id: str,
) -> None:
    if [position for position, _ in observed] != [position for position, _ in expected]:
        raise ValueError(f"First-stage pool mismatch for {question_id}")


def _require_permutation(
    observed: Sequence[tuple[int, float]],
    pool: Sequence[tuple[int, float]],
    question_id: str,
) -> None:
    observed_ids = [position for position, _ in observed]
    pool_ids = [position for position, _ in pool]
    if len(observed_ids) != len(pool_ids) or sorted(observed_ids) != sorted(pool_ids):
        raise ValueError(f"Reranker output is not a pool permutation for {question_id}")


def _require_finite_scores(rows: Sequence[tuple[int, float]]) -> None:
    if any(not np.isfinite(score) for _, score in rows):
        raise ValueError("Ranking contains non-finite scores")


def _storage_metrics(
    index: ExactIndex,
    candidate: RetrievalCandidate,
    reranker: Reranker | None,
) -> dict[str, float]:
    dense_bytes = int(index.vectors.nbytes) if index.vectors is not None else 0
    bm25_bytes = index.bm25.storage_bytes if index.bm25 is not None else 0
    result: dict[str, float] = {}
    if reranker is None:
        if candidate.retriever == "dense":
            result["storage.index_bytes"] = float(dense_bytes)
        elif candidate.retriever == "bm25":
            result["storage.index_bytes"] = float(bm25_bytes)
        else:
            result["storage.required_index_bytes"] = float(dense_bytes + bm25_bytes)
            result["storage.incremental_index_bytes"] = 0.0
    if reranker is not None:
        result["storage.reranker_snapshot_bytes"] = float(reranker.snapshot_bytes)
    return result


def _candidate_contract_errors(
    candidate: RetrievalCandidate,
    metrics: Mapping[str, float],
    operational: Mapping[str, float],
    protocol: RetrievalProtocol,
) -> list[str]:
    required_metrics = {
        "workload.reranker_input_tokens_per_query_mean",
        "workload.reranker_input_tokens_per_query_p95",
        "storage.reranker_snapshot_bytes",
    }
    required_operational = {
        "full_stack_latency_ms_p50",
        "full_stack_latency_ms_p95",
        "first_stage_latency_ms_p50",
        "first_stage_latency_ms_p95",
        "cold_initialization_seconds",
    }
    if candidate.reranker == "none":
        required_metrics = {
            protocol.pool_recall_metric,
            f"{protocol.pool_recall_metric}.sample_count",
        }
        required_operational.add("index_build_seconds")
        required_metrics.add(
            "storage.required_index_bytes"
            if candidate.retriever == "rrf"
            else "storage.index_bytes"
        )
        if candidate.retriever == "rrf":
            required_metrics.add("storage.incremental_index_bytes")
    else:
        required_operational.update(
            {"reranker_latency_ms_p50", "reranker_latency_ms_p95"}
        )
    missing_metrics = sorted(required_metrics - metrics.keys())
    missing_operational = sorted(required_operational - operational.keys())
    errors = []
    if missing_metrics:
        errors.append("missing candidate metrics: " + ", ".join(missing_metrics))
    if missing_operational:
        errors.append("missing operational metrics: " + ", ".join(missing_operational))
    return errors


_INDEX_IDENTITY_FIELDS = frozenset(
    {
        "retriever",
        "chunks_sha256",
        "dense_index_sha256",
        "bm25_index_sha256",
    }
)


def _index_identity(index: ExactIndex, candidate: RetrievalCandidate) -> dict[str, str]:
    dense_sha = (
        hashlib.sha256(index.vectors.tobytes()).hexdigest()
        if index.vectors is not None
        else None
    )
    bm25_sha = (
        hashlib.sha256(index.bm25.serialized).hexdigest()
        if index.bm25 is not None
        else None
    )
    identity: dict[str, str] = {
        "retriever": candidate.retriever,
        "chunks_sha256": stable_hash(
            [{"id": chunk.identifier, "text": chunk.text} for chunk in index.chunks]
        ),
    }
    if dense_sha is not None:
        identity["dense_index_sha256"] = dense_sha
    if bm25_sha is not None:
        identity["bm25_index_sha256"] = bm25_sha
    expected_components = {
        "dense": {"dense_index_sha256"},
        "bm25": {"bm25_index_sha256"},
        "rrf": {"dense_index_sha256", "bm25_index_sha256"},
    }[candidate.retriever]
    observed_components = set(identity) & {
        "dense_index_sha256",
        "bm25_index_sha256",
    }
    if observed_components != expected_components:
        raise RuntimeError(
            f"{candidate.retriever} built unexpected index components: "
            + ", ".join(sorted(observed_components))
        )
    return identity


def _index_build_artifact(
    index: ExactIndex,
    candidate: RetrievalCandidate,
    identity: Mapping[str, str] | None = None,
) -> dict[str, object]:
    identity = dict(identity or _index_identity(index, candidate))
    return {
        "chunking_fingerprint": index.chunking_fingerprint,
        **identity,
        "build_seconds": index.corpus_build_seconds,
        "dense_index_bytes": (
            int(index.vectors.nbytes) if index.vectors is not None else 0
        ),
        "bm25_index_bytes": (index.bm25.storage_bytes if index.bm25 is not None else 0),
        "rrf_incremental_index_bytes": 0 if candidate.retriever == "rrf" else None,
    }


def _parameters(
    candidate: RetrievalCandidate,
    chunker_name: str,
    embedding_name: str,
    model_lock: Mapping[str, Mapping[str, object]],
    index: ExactIndex,
    reranker: Reranker | None,
    plan: BenchmarkPlan,
    protocol: RetrievalProtocol,
    *,
    device: str,
    dtype: str,
    manifest: DatasetManifest,
    pool_collection_checksum: str,
    index_identity: Mapping[str, str],
    pool_owner_child_run_id: str | None,
) -> dict[str, object]:
    embedding_entry = model_lock[embedding_name]
    reranker_entry = (
        model_lock[candidate.reranker_model]
        if candidate.reranker_model is not None
        else None
    )
    parameters: dict[str, object] = {
        "candidate": candidate.identifier,
        "reranker": candidate.reranker,
        "chunker": chunker_name,
        "chunking_fingerprint": index.chunking_fingerprint,
        "embedding": embedding_name,
        "embedding_revision": embedding_entry.get("revision"),
        "embedding_model_cache_manifest_sha256": embedding_entry.get(
            "model_cache_manifest_sha256"
        ),
        "embedding_contract": asdict(index.embedding_spec),
        "embedding_batch_size": protocol.embedding_batch_size,
        "pool_owner": candidate.owner_identifier,
        "pool_owner_child_run_id": pool_owner_child_run_id,
        "pool_collection_checksum": pool_collection_checksum,
        **index_identity,
        "retrieval_protocol_version": protocol.meta.version,
        "retrieval_protocol_checksum": protocol.meta.checksum,
        "requested_pool_size": protocol.pool_size,
        "quality_cutoffs": list(protocol.quality_cutoffs),
        "evaluation_tokenizer": protocol.evaluation_tokenizer,
        "evidence_coverage_rule": protocol.evidence_coverage_rule,
        "cold_initialization_excludes": "offline-index-build,input-preflight",
        "alpha": protocol.alpha_ndcg_alpha,
        "device": device,
        "dtype": dtype,
        "seed": plan.seed,
        "warmups": plan.warmups,
        "repetitions": plan.repetitions,
        "split": manifest.split,
        "manifest_checksum": manifest.checksum,
        "manifest_fingerprint": manifest.fingerprint,
        "package_versions": package_versions(
            (
                "numpy",
                "rank-bm25",
                "sentence-transformers",
                "tiktoken",
                "torch",
                "transformers",
            )
        ),
    }
    if candidate.retriever in {"dense", "rrf"}:
        parameters.update(
            {
                "dense_similarity": protocol.dense_similarity,
                "dense_normalized": protocol.dense_normalized,
                "dense_tie_break": "stable-corpus-order",
            }
        )
    if candidate.retriever in {"bm25", "rrf"}:
        parameters.update(
            {
                "bm25_variant": protocol.bm25_variant,
                "bm25_tokenizer": protocol.bm25_tokenizer,
                "bm25_k1": protocol.bm25_k1,
                "bm25_b": protocol.bm25_b,
                "bm25_epsilon": protocol.bm25_epsilon,
                "bm25_tie_break": "stable-corpus-order",
            }
        )
    if candidate.retriever == "rrf":
        parameters.update(
            {
                "rrf_source_depth": protocol.rrf_source_depth,
                "rrf_dense_weight": protocol.rrf_weights[0],
                "rrf_bm25_weight": protocol.rrf_weights[1],
                "rrf_constant": protocol.rrf_constant,
                "rrf_union": "stable-chunk-id",
                "rrf_limit": protocol.pool_size,
                "rrf_tie_break": ("best-source-rank,combined-source-rank,chunk-id"),
            }
        )
    if reranker is not None and reranker_entry is not None:
        parameters.update(
            {
                "reranker_model": reranker.model_name,
                "reranker_revision": reranker.revision,
                "reranker_model_path": reranker.model_path,
                "reranker_model_cache_manifest_sha256": reranker_entry.get(
                    "model_cache_manifest_sha256"
                ),
                "reranker_tokenizer": reranker.model_name,
                "reranker_tokenizer_revision": reranker.revision,
                "reranker_tokenizer_path": reranker.model_path,
                "reranker_tokenizer_cache_manifest_sha256": reranker_entry.get(
                    "model_cache_manifest_sha256"
                ),
                "reranker_input_template": protocol.reranker_input_template,
                "reranker_reject_truncation": protocol.reject_truncation,
                "reranker_maximum_length": reranker.maximum_length,
                "reranker_batch_size": reranker.batch_size,
                "reranker_score_interpretation": "raw-cross-encoder-logit",
                "reranker_tie_break": "frozen-pool-order",
            }
        )
    return parameters


def _unexpected_failure(
    candidate: str,
    manifest: DatasetManifest,
    error: str,
    *,
    preflight: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": "failed",
        "error": error,
        "artifacts": {
            "validation_report": {
                "candidate": candidate,
                "status": "failed",
                "manifest_checksum": manifest.checksum,
                "manifest_fingerprint": manifest.fingerprint,
                "preflight": dict(preflight or {}),
                "errors": [error],
                "warnings": [],
            }
        },
    }


def _mean(values: Sequence[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: Sequence[int], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0
