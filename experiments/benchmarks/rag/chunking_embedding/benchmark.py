"""Isolated execution and offline scoring for chunker/embedding pairs."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.datasets import answerable_questions
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    benchmark_objects_from_payload,
    candidate_execution_payload,
    decode_worker_result,
    payload_mapping,
    run_json_worker,
    seed_deterministically,
    successful_execution_payload,
)
from experiments.benchmarks.common.protocol import validate_execution
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.rag.evaluation import (
    InputCompatibilityError,
    build_index,
    dense_rank_with_scores,
    validate_ranking,
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


def directions_for(manifest, protocol):
    scopes = {
        "overall",
        *(str(q["evidence_type"]) for q in answerable_questions(manifest)),
    }
    directions = {
        **{
            f"quality.{scope}.{metric}": "max"
            for scope in sorted(scopes)
            for metric in protocol.quality_metrics
        },
        **OPERATIONAL_DIRECTIONS,
    }
    return directions, tuple(directions)


def evaluate_candidate(candidate, manifest, model_lock, plan, *, device, dtype):
    """Stop measurement before scoring; preserve every scheduled first observation."""
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
    execution = protocol.profile(plan.profile)
    if device not in execution.devices or dtype != execution.dtype_for(device):
        raise ValueError("Worker device/dtype does not match the frozen profile")
    chunker, model = split_candidate(candidate)
    entry = model_lock[model]
    questions = answerable_questions(manifest)
    if not questions:
        raise ValueError("Chunking/embedding requires answerable questions")
    random.Random(plan.seed).shuffle(questions)
    attempts = [
        {
            "phase": "query",
            "question_id": str(q["id"]),
            "repetition": rep,
            "status": "unexecuted",
            "latency_ms": None,
            "error": "execution_stopped",
            "ranking": None,
        }
        for q in questions
        for rep in range(1, plan.repetitions + 1)
    ]
    progress, errors = {}, []
    index = None
    monitor = ResourceMonitor(device=device, require_vram=device == "cuda")
    try:
        with monitor:
            seed_deterministically(plan.seed)
            index = build_index(
                manifest,
                chunker,
                model,
                model_lock,
                with_dense=True,
                with_bm25=False,
                device=device,
                dtype=dtype,
                chunking_protocol=protocol,
                progress=progress,
            )
            if (
                device == "cuda"
                and index.preflight.get("placement", {}).get("status") != "qualified"
            ):
                raise RuntimeError("Embedding weights are not verifiably GPU-resident")
            for _ in range(plan.warmups):
                ranking = dense_rank_with_scores(
                    index, str(questions[0]["question"]), protocol.audit_depth
                )
                validate_ranking(index, ranking, protocol.audit_depth)
            by_id = {str(q["id"]): q for q in questions}
            for row in attempts:
                seed_deterministically(plan.seed)
                started = time.perf_counter()
                try:
                    ranking = dense_rank_with_scores(
                        index,
                        str(by_id[row["question_id"]]["question"]),
                        protocol.audit_depth,
                    )
                    validate_ranking(index, ranking, protocol.audit_depth)
                    row.update(status="success", error=None, ranking=ranking)
                except Exception as exc:  # noqa: BLE001 - retain every measured outcome
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    errors.append(str(row["error"]))
                    if isinstance(exc, InputCompatibilityError):
                        progress["input_validation"] = exc.report
                row["latency_ms"] = 1000 * (time.perf_counter() - started)
                if not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
                    row["latency_ms"] = None
                    row["timing_error"] = "latency_measurement_unavailable"
                    errors.append(row["timing_error"])
            if device == "cuda":
                placement = index.embedder.placement_report(device)
                progress["placement_after"] = placement
                if placement.get("status") != "qualified":
                    raise RuntimeError(f"Embedding GPU placement changed: {placement}")
    except Exception as exc:  # noqa: BLE001 - build/warmup/measurement failure
        if "build_started" in progress and index is None:
            progress["build_elapsed_to_failure"] = (
                time.perf_counter() - progress["build_started"]
            )
        errors.append(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, InputCompatibilityError):
            progress["input_validation"] = exc.report
    # This boundary excludes all scoring, aggregation, and bootstrap allocations.
    try:
        operational = monitor.metrics()
        resource_evidence = {**operational, "gpu_identity": dict(monitor.gpu_identity)}
    except RuntimeError as exc:
        errors.append(str(exc))
        resource_evidence = getattr(exc, "resource_evidence", {})
        operational = {
            key: resource_evidence.get(key)
            for key in ("peak_process_tree_ram_mb", "peak_vram_mb")
        }
    build_seconds = index.corpus_build_seconds if index is not None else None
    if build_seconds is not None and (
        not math.isfinite(build_seconds) or build_seconds < 0
    ):
        errors.append("Invalid completed corpus-build timer")
        build_seconds = None
    build_elapsed = progress.get("build_elapsed_to_failure")
    chunks = index.chunks if index is not None else progress.get("chunks", [])
    generation_complete = index is not None or progress.get(
        "generation_complete", False
    )
    documents = [row for row in manifest.samples if row.get("kind") == "document"]
    tokenizer = index.tokenizer if index is not None else None
    if tokenizer is None:
        from edumind.rag.tokenizers import TiktokenOffsetTokenizer

        tokenizer = TiktokenOffsetTokenizer(
            protocol.tokenizer.removeprefix("tiktoken:")
        )
    source_tokens = sum(tokenizer.count(str(row["text"])) for row in documents)
    spec = (
        index.embedding_spec
        if index is not None
        else embedding_spec(
            model,
            revision=str(entry["revision"]),
            local_path=str(entry["model_path"]),
            document_device=device,
            query_device=device,
            config_options=protocol.model_config_options(model),
        )
    )
    lengths = [chunk.tokens for chunk in chunks]
    # Shape, finiteness, norms, dtype and contiguity are validated once in build_index.
    matrix_bytes = index.vectors.nbytes if index is not None else None
    workload = {
        "workload.document_count": float(len(documents)),
        "workload.answerable_query_count": float(len(questions)),
        "workload.unanswerable_query_count": float(
            sum(
                row.get("kind") == "question" and not row.get("answerable")
                for row in manifest.samples
            )
        ),
        "workload.source_tokens": float(source_tokens),
        "workload.chunk_count": float(len(chunks)) if generation_complete else None,
        "workload.indexed_token_occurrences": float(sum(lengths))
        if generation_complete
        else None,
        "workload.chunk_tokens_mean": float(np.mean(lengths))
        if generation_complete and lengths
        else None,
        "workload.chunk_tokens_p95": float(np.percentile(lengths, 95))
        if generation_complete and lengths
        else None,
        "workload.embedding_dimension": float(spec.dimension),
        "workload.embedding_matrix_bytes": float(matrix_bytes)
        if matrix_bytes is not None
        else None,
    }
    samples, retrievals, matches, query_rows = [], [], [], []
    determinism_failures = 0
    for question in questions:
        rows = [row for row in attempts if row["question_id"] == str(question["id"])]
        first = rows[0]
        successful = [row for row in rows if row["status"] == "success"]
        timed = [row for row in successful if row["latency_ms"] is not None]
        latency = (
            float(np.median([row["latency_ms"] for row in timed])) / 1000
            if timed
            else None
        )
        rankings = [
            tuple(i for i, _ in row["ranking"][: max(protocol.quality_cutoffs)])
            for row in successful
        ]
        if rankings and any(ranking != rankings[0] for ranking in rankings[1:]):
            determinism_failures += 1
            errors.append(f"Repeated ranking changed for {question['id']}")
        values = {name: None for name in protocol.quality_metrics}
        diagnostics = {}
        reason = first["error"]
        status = "unavailable" if first["status"] != "success" else "complete"
        if first["status"] == "success":
            selected = [chunks[position] for position, _ in first["ranking"]]
            try:
                scored = score_question(
                    question,
                    selected,
                    chunks,
                    tokenizer,
                    cutoffs=protocol.quality_cutoffs,
                    alpha=protocol.alpha_ndcg_alpha,
                )
                values.update(scored.metrics)
                diagnostics = dict(scored.diagnostics)
                matches.extend(
                    {
                        "question_id": question["id"],
                        **row,
                        "matching_chunk_ids": json.dumps(row["matching_chunk_ids"]),
                    }
                    for row in scored.matches
                )
            except (ValueError, RuntimeError) as exc:
                status, reason = "unavailable", f"scoring_integrity_failed: {exc}"
                errors.append(reason)
            retrievals.extend(
                {
                    "question_id": str(question["id"]),
                    "rank": rank,
                    "chunk_id": chunks[position].identifier,
                    "document_id": chunks[position].document_id,
                    "start": chunks[position].start,
                    "end": chunks[position].end,
                    "cosine_score": score,
                    "scored": rank <= max(protocol.quality_cutoffs),
                }
                for rank, (position, score) in enumerate(first["ranking"], 1)
            )
        metadata = {
            "document_id": str(question["document_id"]),
            "evidence_type": str(question["evidence_type"]),
            "first_attempt_status": first["status"],
            "quality_status": status,
            "quality_reason": reason,
            "successful_attempts": len(successful),
            "timed_successful_attempts": len(timed),
            "scheduled_attempts": len(rows),
            "native_query_tokens": index.preflight.get("query_token_counts", {}).get(
                str(question["question"])
            )
            if index is not None
            else None,
        }
        samples.append(
            SampleResult(
                str(question["id"]),
                prefixed_quality(values, str(question["evidence_type"])),
                latency,
                metadata,
            )
        )
        query_rows.append(
            {
                "question_id": str(question["id"]),
                **metadata,
                **values,
                **diagnostics,
                "latency_ms": latency * 1000 if latency is not None else None,
            }
        )
    support = {}
    resamples = 0 if plan.profile == "smoke" else plan.bootstrap_resamples
    aggregate, intervals = aggregate_quality(
        samples,
        resamples=resamples,
        seed=plan.seed,
        confidence=protocol.confidence_level,
        minimum_documents=protocol.minimum_quality_ci_documents,
        diagnostics=support,
        metric_names=[
            f"quality.{scope}.{name}"
            for scope in ("overall", "text", "table", "formula", "mixed")
            for name in protocol.quality_metrics
        ],
    )
    intervals.update(
        latency_intervals(
            samples,
            resamples=resamples,
            seed=plan.seed,
            confidence=protocol.confidence_level,
            minimum_documents=protocol.minimum_latency_ci_documents,
        )
    )
    medians = [
        sample.latency_seconds * 1000
        for sample in samples
        if sample.latency_seconds is not None
    ]
    operational.update(
        {
            "corpus_build_seconds": build_seconds,
            "corpus_build_source_tokens_per_second": source_tokens / build_seconds
            if build_seconds and math.isfinite(build_seconds)
            else None,
            "query_latency_ms_p50": float(np.percentile(medians, 50))
            if medians
            else None,
            "query_latency_ms_p95": float(np.percentile(medians, 95))
            if medians
            else None,
        }
    )
    validity = {
        "validity.expected_documents": float(len(documents)),
        "validity.processed_documents": float(
            len(documents)
            if index is not None
            else progress.get("processed_documents", 0)
        ),
        "validity.expected_queries": float(len(questions)),
        "validity.processed_queries": float(
            sum(
                row["repetition"] == 1 and row["status"] != "unexecuted"
                for row in attempts
            )
        ),
        "validity.scheduled_attempt_count": float(len(attempts)),
        **{
            f"validity.{status}_attempt_count": float(
                sum(row["status"] == status for row in attempts)
            )
            for status in ("success", "failed", "unexecuted")
        },
        "validity.determinism_failed_queries": float(determinism_failures),
        "validity.truncated_inputs": 0.0,
        **eligible_counts(samples),
        **{
            f"validity.{name}.{count}": float(value)
            for name, details in support.items()
            for count, value in details.items()
            if count.endswith("_count")
        },
    }
    aggregate.update(workload)
    aggregate.update(validity)
    for name in ("query_latency_ms_p50", "query_latency_ms_p95"):
        aggregate[f"operational.{name}.sample_count"] = float(
            len(
                {
                    s.metadata["document_id"]
                    for s in samples
                    if s.latency_seconds is not None
                }
            )
        )
    input_validation = {
        "chunking_contract": index.chunking_contract
        if index is not None
        else progress.get("chunking_contract", {}),
        **(index.preflight if index is not None else {}),
        **progress.get("input_validation", {}),
    }
    if "placement_after" in progress:
        input_validation["placement_after"] = progress["placement_after"]
    parameters = _parameters(
        candidate,
        chunker,
        model,
        entry,
        plan,
        device,
        dtype,
        manifest.split,
        manifest.checksum,
        manifest.fingerprint,
        spec=spec,
        preflight=input_validation,
        protocol=protocol,
    )
    parameters.update(
        {
            "measurement.resource_scope": "partial"
            if any(row["status"] == "unexecuted" for row in attempts)
            else "complete",
            "resource_window": "model-load-through-measured-queries",
            "vram_measurement_method": monitor.vram_measurement_method,
            "gpu_identity": resource_evidence.get("gpu_identity", {}),
            "quality_support": support,
            "chunk_inventory_scope": "complete"
            if generation_complete
            else "partial"
            if chunks
            else "unavailable",
            "metric_statuses": {
                **{
                    f"operational.{name}": {
                        "status": "complete" if value is not None else "unavailable",
                        "reason": None
                        if value is not None
                        else "no_completed_observation",
                    }
                    for name, value in operational.items()
                },
                **{
                    name: {
                        "status": row["status"],
                        "reason": None
                        if row["status"] == "complete"
                        else "no_eligible_questions"
                        if row["status"] == "inapplicable"
                        else "missing_required_first_scores",
                    }
                    for name, row in support.items()
                },
                "operational.peak_vram_mb": {
                    "status": "inapplicable"
                    if device == "cpu"
                    else "complete"
                    if operational["peak_vram_mb"] is not None
                    else "unavailable",
                    "reason": "cpu_execution"
                    if device == "cpu"
                    else "gpu_telemetry_unavailable"
                    if operational["peak_vram_mb"] is None
                    else None,
                },
            },
        }
    )
    artifacts = {
        "chunk_manifest": [
            {
                "chunk_id": c.identifier,
                "document_id": c.document_id,
                "start": c.start,
                "end": c.end,
                "evaluation_tokens": c.tokens,
                "model_tokens": c.model_tokens,
                "scope": parameters["chunk_inventory_scope"],
                "chunking_fingerprint": parameters["chunking_fingerprint"],
            }
            for c in chunks
        ],
        "query_metrics": query_rows,
        "retrievals": retrievals,
        "evidence_matches": matches,
        "timings": [
            {
                "phase": "corpus_build",
                "status": "success" if build_seconds is not None else "failed",
                "duration_seconds": build_seconds,
                "elapsed_to_failure_seconds": build_elapsed,
            },
            *[
                {key: value for key, value in row.items() if key != "ranking"}
                for row in attempts
            ],
        ],
        "resources": [
            {
                **row,
                "scope": parameters["measurement.resource_scope"],
                "ram_status": "complete"
                if row["process_tree_ram_mb"] is not None
                else "unavailable",
                "vram_status": "inapplicable"
                if device == "cpu"
                else "complete"
                if row["vram_mb"] is not None
                else "unavailable",
            }
            for row in monitor.samples()
        ],
        "validation_report": {
            "candidate": candidate,
            "status": "failed" if errors else "passed",
            "manifest_checksum": manifest.checksum,
            "manifest_fingerprint": manifest.fingerprint,
            "reason_code": input_validation.get("reason_code"),
            "preflight": dict(input_validation),
            "errors": errors,
            "warnings": [],
            **validity,
        },
    }
    if errors:
        raise CandidateExecutionError(
            "; ".join(errors),
            samples=tuple(samples),
            metrics=aggregate,
            intervals=intervals,
            operational=operational,
            parameters=parameters,
            artifacts=artifacts,
        )
    return samples, operational, aggregate, parameters, intervals, artifacts


def run_in_fresh_process(candidate, manifest, model_lock, plan, *, device, dtype):
    protocol = protocol_from_settings(plan.settings)
    try:
        result = run_json_worker(
            WORKER,
            _payload(candidate, manifest, model_lock, plan, device, dtype),
            device=device,
            prefix="edumind-chunking-embedding-",
            error_label=f"chunking/embedding worker {candidate}",
            timeout_seconds=protocol.preflight.worker_timeout_seconds,
        )
        try:
            return decode_worker_result(result)
        except CandidateExecutionError:
            raise
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            raise WorkerExecutionError(
                f"Invalid chunking/embedding worker result: {exc}",
                {
                    key: value
                    for key, value in payload_mapping(result.get("operational")).items()
                    if key in {"peak_process_tree_ram_mb", "peak_vram_mb"}
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                },
            ) from exc
    except WorkerExecutionError as exc:
        # A lost worker result does not reveal which requests actually executed.
        scheduled = len(answerable_questions(manifest)) * plan.repetitions
        exc.parameters["measurement.resource_scope"] = "partial"
        exc.artifacts["validation_report"] = {
            "candidate": candidate,
            "status": "failed",
            "reason_code": "missing_worker_result",
            "manifest_checksum": manifest.checksum,
            "scheduled_attempt_count": scheduled,
            "missing_attempt_record_count": scheduled,
            "errors": [str(exc)],
        }
        raise


def _payload(candidate, manifest, model_lock, plan, device, dtype):
    return {
        "candidate": candidate,
        "manifest": asdict(manifest),
        "model_lock": dict(model_lock),
        "plan": asdict(plan),
        "device": device,
        "dtype": dtype,
    }


def preflight_in_fresh_process(
    candidate, manifest, model_lock, plan, *, vram_limit_mb, preflight
):
    result = run_json_worker(
        WORKER,
        {
            **_payload(
                candidate,
                manifest,
                model_lock,
                plan,
                "cuda",
                protocol_from_settings(plan.settings).profile("development").dtype,
            ),
            "mode": "preflight",
        },
        device="cuda",
        prefix="edumind-chunking-preflight-",
        error_label=f"chunking/embedding preflight {candidate}",
        vram_limit_mb=vram_limit_mb,
        telemetry_interval_seconds=preflight.telemetry_interval_seconds,
        poll_interval_seconds=preflight.poll_interval_seconds,
        timeout_seconds=preflight.worker_timeout_seconds,
    )
    result.update(result.pop("_worker_supervision", {}))
    return result


def execute_payload(payload):
    candidate, manifest, lock, plan = benchmark_objects_from_payload(payload)
    device, dtype = str(payload["device"]), str(payload["dtype"])
    if payload.get("mode") == "preflight":
        return _preflight_candidate(
            candidate, manifest, lock, plan, device=device, dtype=dtype
        )
    try:
        return successful_execution_payload(
            evaluate_candidate(
                candidate, manifest, lock, plan, device=device, dtype=dtype
            )
        )
    except CandidateExecutionError as exc:
        return candidate_execution_payload(exc)


def _preflight_candidate(candidate, manifest, model_lock, plan, *, device, dtype):
    protocol = protocol_from_settings(plan.settings)
    if (device, dtype) != ("cuda", protocol.profile("development").dtype):
        raise ValueError("Preflight requires the frozen CUDA dtype")
    if (
        candidate not in protocol.development_candidates
        or plan.seed != protocol.meta.seed
        or (plan.warmups, plan.repetitions, plan.bootstrap_resamples)
        != (
            protocol.preflight.warmups,
            protocol.preflight.repetitions,
            protocol.preflight.bootstrap_resamples,
        )
    ):
        raise ValueError("Preflight plan does not match the protocol")
    seed_deterministically(plan.seed)
    chunker, model = split_candidate(candidate)
    index = build_index(
        manifest,
        chunker,
        model,
        model_lock,
        with_dense=True,
        with_bm25=False,
        device=device,
        dtype=dtype,
        chunking_protocol=protocol,
    )
    questions = answerable_questions(manifest)
    if not questions:
        raise ValueError("Preflight requires an answerable stress question")
    before = index.embedder.placement_report(device)
    ranking = dense_rank_with_scores(
        index, str(questions[0]["question"]), protocol.audit_depth
    )
    validate_ranking(index, ranking, protocol.audit_depth)
    after = index.embedder.placement_report(device)
    priority = {
        "qualified": 0,
        "wrong_device": 1,
        "placement_unverifiable": 2,
        "offload_detected": 3,
    }
    return {
        "placement": max(
            (before, after), key=lambda row: priority.get(row.get("status"), 2)
        ),
        "placement_before": before,
        "placement_after": after,
        "input_validation": dict(index.preflight),
        "stress_manifest_fingerprint": manifest.fingerprint,
        "stress_question_id": str(questions[0]["id"]),
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
    spec,
    preflight: Mapping[str, object],
    protocol: ChunkingEmbeddingProtocol,
) -> dict[str, object]:
    chunking_contract = dict(preflight.get("chunking_contract", {}))
    return {
        "candidate": candidate,
        "chunker": chunker_name,
        "chunking_fingerprint": chunking_contract.get("fingerprint", ""),
        "chunking_contract": chunking_contract,
        "embedding": embedding_name,
        "embedding_contract": asdict(spec),
        "embedding_batch_size": protocol.embedding_batch_size,
        "model_revision": entry.get("revision"),
        "model_path": entry.get("model_path"),
        "model_cache_manifest_sha256": entry.get("model_cache_manifest_sha256"),
        "tokenizer_revision": entry.get("revision"),
        "tokenizer_path": entry.get("model_path"),
        "tokenizer_cache_manifest_sha256": entry.get("model_cache_manifest_sha256"),
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
        "resolved_query_input": dict(preflight.get("resolved_query_input", {})),
        "resolved_document_input": dict(preflight.get("resolved_document_input", {})),
        "package_versions": package_versions(
            ("numpy", "sentence-transformers", "tiktoken", "torch", "transformers")
        ),
    }
