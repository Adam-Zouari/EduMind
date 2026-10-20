"""Fresh-process execution for complete retrieval/reranking candidates."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.datasets import answerable_questions, evidence_units
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
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.rag.chunking_embedding.metrics import (
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
    synchronize,
)
from experiments.benchmarks.rag.methods import (
    Reranker,
    reciprocal_rank_fusion_with_scores,
)

from . import metrics as m
from .metrics import (
    pool_evidence_unit_recall,
)
from .profiles import RetrievalCandidate, parse_candidate
from .protocol import RetrievalProtocol, protocol_from_settings

WORKER = Path(__file__).with_name("worker.py")


def evaluate_candidate(
    candidate_name,
    manifest,
    model_lock,
    plan,
    *,
    device,
    dtype,
    frozen_pool_collection,
    child_run_id,
    preparation_only=False,
    index_dependencies=None,
):
    protocol = protocol_from_settings(plan.settings)
    chunking = chunking_protocol_from_settings(plan.settings)
    protocol.validate_execution(
        plan.profile,
        seed=plan.seed,
        warmups=plan.warmups,
        repetitions=plan.repetitions,
        bootstrap_resamples=plan.bootstrap_resamples,
        device=device,
        dtype=dtype,
    )
    candidate = parse_candidate(candidate_name)
    chunker, embedding = split_candidate(
        str(plan.settings.get("chunker_embedding", ""))
    )
    questions = [dict(q) for q in answerable_questions(manifest)]
    if not questions or not any(
        row.get("kind") == "document" for row in manifest.samples
    ):
        raise ValueError(
            "Retrieval/reranking requires nonempty corpus and answerable questions"
        )
    if preparation_only and candidate.reranker != "none":
        raise ValueError("Input preparation requires a no-reranker pool owner")
    random.Random(plan.seed).shuffle(questions)
    documents = {
        str(row["id"]): row for row in manifest.samples if row.get("kind") == "document"
    }
    for question in questions:
        document = documents[str(question["document_id"])]
        question["independent_source_id"] = m.source_id({**document, **question})

    attempts = (
        []
        if preparation_only
        else [
            {
                "question_id": str(q["id"]),
                "document_id": str(q["document_id"]),
                "independent_source_id": m.source_id(q),
                "repetition": rep,
                "status": "not_executed",
                "valid": False,
                "error": "execution_stopped",
                "ordered_chunk_ids": None,
                "ranking": None,
                "retrieval_ranking": None,
                "local_reranker_output": None,
                "full_stack_latency_ms": None,
                "retrieval_latency_ms": None,
                "reranker_latency_ms": None,
                "elapsed_to_failure_ms": None,
            }
            for q in questions
            for rep in range(1, plan.repetitions + 1)
        ]
    )
    progress, errors, native = {}, [], {}
    index = reranker = None
    pools, pool_metadata, identity, index_build = {}, {}, {}, {}
    cold = None
    window_complete = False
    model_backed = candidate.model_backed or chunker == "semantic"
    monitor = ResourceMonitor(
        device=device if model_backed else "cpu",
        require_vram=device == "cuda" and model_backed,
    )
    setup_started = time.perf_counter()
    try:
        with monitor:
            seed_deterministically(plan.seed)
            initialization_started = time.perf_counter()
            index = build_index(
                manifest,
                chunker,
                embedding,
                model_lock,
                with_dense=candidate.retriever in {"dense", "rrf"},
                with_bm25=candidate.retriever in {"bm25", "rrf"},
                device=device,
                dtype=dtype,
                chunking_protocol=chunking,
                retrieval_protocol=protocol,
                progress=progress,
            )
            if not index.chunks:
                raise ValueError("Required index produced an empty corpus")
            if (
                plan.profile == "smoke"
                and len(index.chunks) != protocol.smoke_expected_chunk_count
            ):
                raise ValueError(
                    f"Retrieval smoke fixture must produce exactly {protocol.smoke_expected_chunk_count} chunks"
                )
            identity = _index_identity(index, candidate)
            index_build = _index_build_artifact(index, candidate, identity)
            index_build = _reference_dependencies(index_build, index_dependencies or {})
            reranker = _reranker(
                candidate, model_lock, protocol, device=device, dtype=dtype
            )
            if reranker is not None:
                reranker.prepare()
            synchronize(device if model_backed else "cpu")
            elapsed = time.perf_counter() - initialization_started
            preflight = float(
                index.preflight.get(
                    "input_preflight_seconds_excluded_from_corpus_build", 0
                )
            )
            # corpus_build_seconds is the disjoint outer offline construction interval,
            # not the Dense+BM25 phase sum (which omits chunking).
            cold = elapsed - index.corpus_build_seconds - preflight
            if not np.isfinite(cold) or cold < 0:
                cold = None
                errors.append(
                    "Cold initialization instrumentation produced negative/nonfinite time"
                )

            if preparation_only:
                for question in questions:
                    pool = _first_stage(
                        index, str(question["question"]), candidate.retriever, protocol
                    )
                    _require_pool(pool, index, protocol, str(question["id"]))
                    pools[str(question["id"])] = pool
            elif candidate.reranker != "none":
                pools, pool_metadata = _load_pool_collection(
                    candidate,
                    frozen_pool_collection,
                    questions,
                    index,
                    protocol,
                    index_identity=identity,
                )
            if device == "cuda" and model_backed:
                _placement(index, reranker, device)
            if not preparation_only:
                for warmup in range(plan.warmups):
                    q = questions[warmup % len(questions)]
                    _attempt(
                        index,
                        reranker,
                        candidate,
                        q,
                        pools.get(str(q["id"])),
                        protocol,
                        device,
                        {"repetition": 0},
                    )
                for row in attempts:
                    q = next(q for q in questions if str(q["id"]) == row["question_id"])
                    try:
                        _attempt(
                            index,
                            reranker,
                            candidate,
                            q,
                            pools.get(str(q["id"])),
                            protocol,
                            device,
                            row,
                        )
                        if candidate.reranker == "none":
                            pools.setdefault(str(q["id"]), row["ranking"])
                    except Exception as exc:  # noqa: BLE001 - durable per-attempt ledger
                        if _fatal(exc):
                            errors.append(
                                f"Hardware execution contract: {type(exc).__name__}: {exc}"
                            )
                            break
                    finally:
                        if row["repetition"] == 1 and row.get("native_input"):
                            native[str(q["id"])] = row["native_input"]
                if device == "cuda" and model_backed:
                    _placement(index, reranker, device)
            window_complete = preparation_only or all(
                row["status"] != "not_executed" for row in attempts
            )
    except Exception as exc:  # noqa: BLE001 - retain incomplete setup and independent observations
        errors.append(f"Preparation/warmup/lifecycle: {type(exc).__name__}: {exc}")
        if isinstance(exc, InputCompatibilityError):
            progress["input_validation"] = exc.report
        progress["elapsed_to_failure_seconds"] = time.perf_counter() - setup_started

    resource_evidence = {
        "scope": "complete" if window_complete else "partial",
        "vram_measurement_method": monitor.vram_measurement_method,
        "gpu_identity": dict(monitor.gpu_identity),
    }
    try:
        operational = monitor.metrics()
    except Exception as exc:  # noqa: BLE001 - missing telemetry is not zero
        resource_evidence.update(getattr(exc, "resource_evidence", {}))
        operational = {
            name: resource_evidence.get(name)
            for name in ("peak_process_tree_ram_mb", "peak_vram_mb")
        }
        errors.append(f"Resource telemetry unavailable: {exc}")
    if (
        operational.get("peak_vram_mb") is not None
        and operational["peak_vram_mb"] > protocol.authoritative_peak_vram_mb
    ):
        errors.append("Peak Device VRAM exceeds the protocol hardware cap")
    resource_evidence.update(operational)

    # No evaluator, token-workload calculation or bootstrap belongs to measurement.
    samples, query_rows, matches, rankings = [], [], [], []
    descriptors = {k: [] for k in protocol.quality_cutoffs}
    verification = []
    for q in questions:
        qid = str(q["id"])
        rows = [row for row in attempts if row["question_id"] == qid]
        first = next((row for row in rows if row["repetition"] == 1), None)
        values = {
            f"quality.{scope}.{name}": None
            for scope in ("overall", q["evidence_type"])
            for name in protocol.quality_metrics
        }
        quality_error, score_diagnostics = None, {}
        if first is not None and first["valid"]:
            selected = [index.chunks[pos] for pos, _ in first["ranking"]]
            for k in protocol.quality_cutoffs:
                try:
                    descriptors[k].append(
                        m.retrieved_tokens(selected, k, index.tokenizer)
                    )
                except Exception as exc:  # noqa: BLE001 - workload unavailable independently of quality
                    descriptors[k].append(None)
                    errors.append(f"{qid}: context tokens unavailable: {exc}")
            try:
                score = score_question(
                    q,
                    selected,
                    index.chunks,
                    index.tokenizer,
                    cutoffs=protocol.quality_cutoffs,
                    alpha=protocol.alpha_ndcg_alpha,
                )
                values.update(prefixed_quality(score.metrics, str(q["evidence_type"])))
                score_diagnostics = dict(score.diagnostics)
                score_diagnostics["upstream_fragmentation"] = not any(
                    m.covered_unit_ids(chunk, evidence_units(q))
                    for chunk in index.chunks
                )
                matches.extend(
                    {
                        "question_id": qid,
                        **match,
                        "matching_chunk_ids": json.dumps(match["matching_chunk_ids"]),
                    }
                    for match in score.matches
                )
            except Exception as exc:  # noqa: BLE001 - offline evaluator failure preserves ranking
                quality_error = f"{type(exc).__name__}: {exc}"
        else:
            for counts in descriptors.values():
                counts.append(None)
        if candidate.reranker == "none":
            values[protocol.pool_recall_metric] = (
                pool_evidence_unit_recall(
                    q, [index.chunks[pos] for pos, _ in pools[qid]]
                )
                if index is not None and qid in pools
                else None
            )
        repeat = m.repeatability(rows, plan.repetitions)
        values[m.REPEATABILITY] = repeat["value"]

        def median(key, rows=rows):
            values = [r[key] for r in rows if r.get(key) is not None]
            return float(np.median(values)) if values else None

        full, retrieval, learned = (
            median(key)
            for key in (
                "full_stack_latency_ms",
                "retrieval_latency_ms",
                "reranker_latency_ms",
            )
        )
        meta = {
            "document_id": str(q["document_id"]),
            "independent_source_id": m.source_id(q),
            "evidence_type": str(q["evidence_type"]),
            "first_attempt_status": first["status"] if first else "not_executed",
            "retrieval_latency_ms": retrieval,
            "reranker_latency_ms": learned,
            "reranker_applicable": candidate.reranker != "none",
        }
        if not preparation_only:
            samples.append(
                SampleResult(
                    qid, values, full / 1000 if full is not None else None, meta
                )
            )
            query_rows.append(
                {
                    "question_id": qid,
                    **meta,
                    **values,
                    "quality_status": "scored"
                    if first and first["valid"] and not quality_error
                    else "unavailable",
                    "quality_error": quality_error,
                    "quality_diagnostics": json.dumps(score_diagnostics),
                    "repeatability": json.dumps(repeat),
                    "full_stack_latency_ms": full,
                    "reranker_pair_lengths": json.dumps(
                        native.get(qid, {}).get("pair_lengths")
                    ),
                    "reranker_input_tokens": native.get(qid, {}).get("total"),
                }
            )
        for row in rows:
            verification.append(
                {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {"ranking", "retrieval_ranking", "local_reranker_output"}
                }
            )
            rankings.extend(_ranking_rows(row, index))

    pool_rows = []
    if index is not None and candidate.reranker == "none":
        for q in questions:
            qid = str(q["id"])
            if qid not in pools:
                continue
            for rank, (pos, score) in enumerate(pools[qid], 1):
                pool_rows.append(
                    {
                        "question_id": qid,
                        "rank": rank,
                        "chunk_id": index.chunks[pos].identifier,
                        "score": score,
                    }
                )
    checksum = (
        stable_hash(pool_rows)
        if candidate.reranker == "none"
        else pool_metadata.get("pool_collection_checksum")
    )
    names = [
        f"quality.{scope}.{name}"
        for scope in m.SCOPES
        for name in protocol.quality_metrics
    ]
    if candidate.reranker == "none":
        names.append(protocol.pool_recall_metric)
    aggregate, intervals, support = m.aggregate_quality(
        samples,
        names=names,
        resamples=plan.bootstrap_resamples,
        seed=plan.seed,
        confidence=protocol.confidence_level,
        minimum_documents=protocol.minimum_quality_ci_documents,
    )
    repeat_values, repeat_intervals, repeat_support = m.aggregate_quality(
        samples if plan.repetitions > 1 else [],
        names=[m.REPEATABILITY],
        resamples=plan.bootstrap_resamples,
        seed=plan.seed,
        confidence=protocol.confidence_level,
        minimum_documents=protocol.minimum_repeatability_ci_documents,
    )
    if plan.repetitions == 1:
        repeat_support[m.REPEATABILITY].update(
            status="inapplicable", reason="repeatability_not_measured"
        )
        repeat_intervals[m.REPEATABILITY]["reason"] = "repeatability_not_measured"
    aggregate.update(repeat_values)
    intervals.update(repeat_intervals)
    support.update(repeat_support)
    latency_bounds = m.latency_intervals(
        samples,
        resamples=plan.bootstrap_resamples,
        seed=plan.seed,
        minimum_documents=protocol.minimum_latency_ci_documents,
        confidence=protocol.confidence_level,
    )
    intervals.update(latency_bounds)
    operational.update(m.latency_summary(samples))
    operational["cold_initialization_seconds"] = cold
    for name, bounds in latency_bounds.items():
        aggregate[name + ".sample_count"] = float(
            sum(
                m._latency(row, name.split(".")[1].split("_latency")[0]) is not None
                for row in samples
            )
        )
        support[name] = {
            **bounds,
            "eligible_question_count": len(questions),
            "contributing_question_count": int(aggregate[name + ".sample_count"]),
        }
    workload_support = {}
    for k, counts in descriptors.items():
        details = m.descriptor(counts, len(questions))
        for suffix in ("mean", "p95"):
            key = f"workload.retrieved_context_tokens_at_{k}_{suffix}"
            aggregate[key] = details[suffix] if details["complete"] else None
            workload_support[key] = details
    if candidate.reranker != "none":
        details = m.descriptor(
            [native.get(str(q["id"]), {}).get("total") for q in questions],
            len(questions),
        )
        for suffix in ("mean", "p95"):
            key = f"workload.reranker_input_tokens_per_query_{suffix}"
            aggregate[key] = details[suffix] if details["complete"] else None
            workload_support[key] = details
    if index is not None:
        aggregate.update(_storage_metrics(index, candidate, reranker))
        if candidate.reranker == "none" and candidate.retriever != "rrf":
            operational["index_build_seconds"] = (
                index_build.get("dependencies", {})
                .get(candidate.retriever, {})
                .get("build_seconds")
            )
    elif candidate.reranker == "none" and candidate.retriever != "rrf":
        completed = progress.get("phase_seconds", {}).get(
            f"{candidate.retriever}_index_build_seconds"
        )
        if completed is not None:
            operational["index_build_seconds"] = completed
    integrity_rows = [r for r in attempts if r.get("integrity_error")]
    checked = [r for r in attempts if r.get("permutation_checked")]
    validity = {
        "validity.expected_query_count": float(len(questions)),
        "validity.processed_query_count": float(len(query_rows)),
        "validity.failed_query_count": float(
            len({r["question_id"] for r in attempts if r["status"] == "failed"})
        ),
        "validity.truncated_input_count": float(
            index.preflight.get("truncated_inputs", 0)
        )
        if index
        else 0.0,
        "validity.nonfinite_score_count": float(
            sum("non-finite" in str(r.get("error", "")).lower() for r in attempts)
        ),
        "validity.exact_pool_permutation": 0.0
        if any(r.get("permutation_checked") is False for r in integrity_rows)
        else 1.0
        if checked
        else None,
        **m.reference_counts(questions),
    }
    aggregate.update(validity)
    accounting = {
        "scheduled_attempt_count": len(attempts),
        "recorded_attempt_count": len(attempts),
        "executed_attempt_count": sum(r["status"] != "not_executed" for r in attempts),
        "failed_attempt_count": sum(r["status"] == "failed" for r in attempts),
        "unexecuted_attempt_count": sum(
            r["status"] == "not_executed" for r in attempts
        ),
        "accounting_complete": True,
        "execution_complete": all(r["status"] != "not_executed" for r in attempts),
        "quality_complete": all(
            support[name]["complete"]
            for name in names
            if support[name]["eligible_question_count"]
        ),
        "integrity_complete": not integrity_rows and bool(checked) and not errors,
    }
    if not preparation_only:
        if not accounting["execution_complete"]:
            errors.append("Required execution coverage is incomplete")
        if not accounting["quality_complete"]:
            errors.append(
                "Required designated-first quality is unavailable or incomplete"
            )
        if integrity_rows:
            errors.append("Measured output integrity failed")
        errors.extend(
            _candidate_contract_errors(candidate, aggregate, operational, protocol)
        )
        if plan.profile != "smoke" and (
            protocol.minimum_quality_ci_documents is None
            or protocol.minimum_repeatability_ci_documents is None
        ):
            errors.append(
                "Exploratory only: CI support policy must be frozen before authoritative advancement"
            )
    parameters = (
        {
            **_parameters(
                candidate,
                chunker,
                embedding,
                model_lock,
                index,
                reranker,
                plan,
                protocol,
                device=device,
                dtype=dtype,
                manifest=manifest,
                pool_collection_checksum=checksum,
                index_identity=identity,
                pool_owner_child_run_id=child_run_id
                if candidate.reranker == "none"
                else pool_metadata.get("owner_child_run_id"),
            )
        }
        if index is not None
        else {"candidate": candidate_name, **identity}
    )
    parameters.update(
        {
            "measurement.resource_scope": resource_evidence["scope"],
            "vram_measurement_method": monitor.vram_measurement_method,
            "preparation_only": preparation_only,
            "execution_accounting": accounting,
            "prediction_evidence": {
                "artifact": "rankings.parquet",
                "retention": "local-and-uploaded",
                "manifest_fingerprint": manifest.fingerprint,
            },
            "input_inventory": {
                "corpus_document_count": len(documents),
                "answerable_query_count": len(questions),
                "chunk_count": len(index.chunks)
                if index is not None
                else len(progress.get("chunks", [])),
            },
        }
    )
    report = {
        "candidate": candidate_name,
        "status": "failed" if errors else "passed",
        "manifest_checksum": manifest.checksum,
        "manifest_fingerprint": manifest.fingerprint,
        "pool_collection_checksum": checksum,
        **identity,
        "execution_accounting": accounting,
        "attempt_checks": verification,
        "metric_support": support,
        "workload_support": workload_support,
        "resources": resource_evidence,
        "native_inputs": native,
        "preflight": dict(index.preflight)
        if index
        else progress.get("input_validation", {}),
        "errors": errors,
        "request_failures": [r for r in verification if r["status"] == "failed"],
        "warnings": [],
    }
    artifacts = {
        "query_metrics": query_rows,
        "rankings": rankings,
        "evidence_matches": matches,
        "timings": [
            {
                k: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
                if k
                not in {
                    "ranking",
                    "retrieval_ranking",
                    "local_reranker_output",
                    "ordered_chunk_ids",
                }
            }
            for row in attempts
        ],
        "resources": [
            {**row, "scope": resource_evidence["scope"]} for row in monitor.samples()
        ],
        "validation_report": report,
    }
    if candidate.reranker == "none":
        if index_build is None:
            index_build = _interrupted_index_artifact(progress, candidate)
            for name, observation in index_build["dependencies"].items():
                key = (
                    f"storage.{name}_index_size_bytes"
                    if candidate.retriever == "rrf"
                    else "storage.index_size_bytes"
                )
                aggregate[key] = float(observation["size_bytes"])
                if candidate.retriever != "rrf":
                    operational["index_build_seconds"] = observation["build_seconds"]
            aggregate.setdefault("storage.index_size_bytes", None)
        artifacts.update(
            pool_collection=pool_rows,
            index_build=index_build,
        )
    if errors:
        raise CandidateExecutionError(
            "; ".join(errors),
            samples=tuple(samples),
            operational=operational,
            metrics=aggregate,
            parameters=parameters,
            intervals=intervals,
            artifacts=artifacts,
        )
    return samples, operational, aggregate, parameters, intervals, artifacts


def _fatal(exc):
    text = str(exc).lower()
    return any(
        word in text
        for word in ("out of memory", "offload", "placement", "silently fell back")
    )


def _placement(index, reranker, device):
    for model in (index.embedder, reranker):
        if model is None:
            continue
        report = (
            model.placement_report(device)
            if model is index.embedder
            else model.placement_report()
        )
        if report.get("status") != "qualified":
            raise RuntimeError(f"Placement contract failed: {report}")


def _attempt(index, reranker, candidate, q, pool, protocol, device, row):
    started = phase_started = time.perf_counter()
    phase = "retrieval"
    row.update(status="failed", valid=False, error=None)
    try:
        component_timings = {}
        row["component_timings"] = component_timings
        live = _first_stage(
            index,
            str(q["question"]),
            candidate.retriever,
            protocol,
            timings=component_timings,
        )
        row["retrieval_ranking"] = live
        _require_pool(live, index, protocol, str(q["id"]))
        row["pool_verified"] = True
        if reranker is not None:
            _require_same_pool(live, pool, str(q["id"]))
            row["live_order_verified"] = True
        synchronize(device if candidate.model_backed else "cpu")
        row["retrieval_latency_ms"] = (time.perf_counter() - phase_started) * 1000
        final = live
        if reranker is not None:
            phase, phase_started = "reranker", time.perf_counter()
            texts = [index.chunks[pos].text for pos, _ in pool]
            counts = reranker.input_token_counts(str(q["question"]), texts)
            row["native_input"] = {
                "pair_lengths": list(counts),
                "total": None,
                "prepared_pair_count": len(counts),
                "completed_inference_pair_count": 0,
            }
            if len(counts) != len(pool) or any(
                isinstance(count, bool)
                or not isinstance(count, (int, np.integer))
                or count <= 0
                for count in counts
            ):
                raise ValueError("Malformed native reranker token counts")
            row["native_input"]["total"] = sum(counts)
            if any(count > reranker.maximum_length for count in counts):
                raise InputCompatibilityError(
                    "Reranker input_length_exceeded; truncation is forbidden",
                    row["native_input"],
                )
            local = reranker.rank_with_scores(
                str(q["question"]), texts, validate_inputs=False
            )
            row["local_reranker_output"] = local
            row["permutation_checked"] = False
            _require_permutation(
                local,
                [(pos, score) for pos, (_, score) in enumerate(pool)],
                str(q["id"]),
            )
            _require_finite_scores(local)
            final = [(pool[pos][0], score) for pos, score in local]
            row["native_input"]["completed_inference_pair_count"] = len(local)
        row["ranking"] = list(final)
        row["permutation_checked"] = False
        _require_permutation(
            final, pool if reranker is not None else live, str(q["id"])
        )
        _require_finite_scores(final)
        row["permutation_checked"] = True
        row["prefix_verified"] = (
            True  # cutoffs use this exact ordered prefix, never a subset
        )
        synchronize(device if candidate.model_backed else "cpu")
        if reranker is not None:
            row["reranker_latency_ms"] = (time.perf_counter() - phase_started) * 1000
        row.update(
            status="success",
            valid=True,
            error=None,
            ordered_chunk_ids=[index.chunks[pos].identifier for pos, _ in final],
            full_stack_latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_to_failure_ms=(time.perf_counter() - started) * 1000,
            failed_phase=phase,
            phase_elapsed_to_failure_ms=(time.perf_counter() - phase_started) * 1000,
        )
        row["integrity_error"] = isinstance(
            exc, (ValueError, IndexError, TypeError)
        ) or any(
            word in str(exc).lower()
            for word in ("non-finite", "malformed", "incomplete ranking", "truncat")
        )
        raise


def _ranking_rows(row, index):
    base = {
        key: row.get(key)
        for key in (
            "question_id",
            "document_id",
            "independent_source_id",
            "repetition",
            "status",
            "valid",
            "error",
        )
    }
    result = []
    for representation, output in (
        ("final", row.get("ranking")),
        ("retrieval", row.get("retrieval_ranking")),
        ("local_reranker", row.get("local_reranker_output")),
    ):
        if output is None:
            continue
        raw = json.dumps(output, default=str, ensure_ascii=False)
        emitted = False
        if isinstance(output, (list, tuple)):
            for rank, item in enumerate(output, 1):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                pos, score = item
                valid_pos = (
                    isinstance(pos, (int, np.integer))
                    and not isinstance(pos, bool)
                    and index is not None
                    and 0 <= pos < len(index.chunks)
                )
                chunk = (
                    index.chunks[pos]
                    if valid_pos and representation != "local_reranker"
                    else None
                )
                try:
                    numeric_score = float(score) if np.isfinite(float(score)) else None
                except (ValueError, TypeError):
                    numeric_score = None
                result.append(
                    {
                        **base,
                        "representation": representation,
                        "rank": rank,
                        "position": str(pos),
                        "chunk_id": chunk.identifier if chunk else None,
                        "chunk_document_id": chunk.document_id if chunk else None,
                        "start": chunk.start if chunk else None,
                        "end": chunk.end if chunk else None,
                        "score": numeric_score,
                        "raw_output_json": raw if not emitted else None,
                        "output_status": "complete" if row["valid"] else "fragment",
                    }
                )
                emitted = True
        if not emitted:
            result.append(
                {
                    **base,
                    "representation": representation,
                    "rank": None,
                    "chunk_id": None,
                    "score": None,
                    "raw_output_json": raw,
                    "output_status": "fragment",
                }
            )
    if not result:
        result.append(
            {
                **base,
                "representation": "final",
                "rank": None,
                "chunk_id": None,
                "score": None,
                "raw_output_json": None,
                "output_status": "not_produced",
            }
        )
    return result


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
    preparation_only: bool = False,
    index_dependencies: Mapping[str, object] | None = None,
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
        "preparation_only": preparation_only,
        "index_dependencies": dict(index_dependencies or {}),
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
            preparation_only=bool(payload.get("preparation_only", False)),
            index_dependencies=payload_mapping(payload.get("index_dependencies")),
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
    *,
    timings: dict[str, float] | None = None,
) -> list[tuple[int, float]]:
    if retriever == "dense":
        return dense_rank_with_scores(index, query, protocol.pool_size, timings=timings)
    if index.bm25 is None:
        raise RuntimeError("BM25 state is unavailable")
    lexical = index.bm25.rank(query, protocol.pool_size)
    if retriever == "bm25":
        return lexical
    dense = dense_rank_with_scores(index, query, protocol.pool_size, timings=timings)
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
        if len(values) != min(protocol.pool_size, len(index.chunks)):
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
        _require_pool(values, index, protocol, question_id)
    return by_question, {
        "pool_collection_checksum": checksum,
        "owner_child_run_id": owner_child_run_id,
    }


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
    if (
        any(
            isinstance(pos, bool) or not isinstance(pos, (int, np.integer))
            for pos in observed_ids
        )
        or len(set(pool_ids)) != len(pool_ids)
        or len(set(observed_ids)) != len(observed_ids)
        or len(observed_ids) != len(pool_ids)
        or sorted(observed_ids) != sorted(pool_ids)
    ):
        raise ValueError(f"Reranker output is not a pool permutation for {question_id}")


def _require_finite_scores(rows: Sequence[tuple[int, float]]) -> None:
    if any(not np.isfinite(score) for _, score in rows):
        raise ValueError("Ranking contains non-finite scores")


def _require_pool(rows, index, protocol, question_id):
    if len(rows) != min(protocol.pool_size, len(index.chunks)):
        raise ValueError(f"Incomplete query pool for {question_id}")
    ids = [position for position, _ in rows]
    if any(
        isinstance(pos, bool)
        or not isinstance(pos, (int, np.integer))
        or not 0 <= pos < len(index.chunks)
        for pos in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid/duplicate chunk IDs in pool for {question_id}")
    _require_finite_scores(rows)
    if any(not index.tokenizer.spans(index.chunks[pos].text) for pos in ids):
        raise ValueError(f"Token-free chunk in pool for {question_id}")


def _storage_metrics(index, candidate, reranker=None):
    if candidate.reranker != "none":
        return {}
    dense = int(index.vectors.nbytes) if index.vectors is not None else None
    bm25 = index.bm25.storage_bytes if index.bm25 is not None else None
    if candidate.retriever == "dense":
        return {"storage.index_size_bytes": float(dense) if dense is not None else None}
    if candidate.retriever == "bm25":
        return {"storage.index_size_bytes": float(bm25) if bm25 is not None else None}
    return {
        "storage.index_size_bytes": float(dense + bm25)
        if dense is not None and bm25 is not None
        else None,
        "storage.dense_index_size_bytes": float(dense) if dense is not None else None,
        "storage.bm25_index_size_bytes": float(bm25) if bm25 is not None else None,
    }


def _candidate_contract_errors(candidate, metrics, operational, protocol):
    required_metrics = {
        f"workload.retrieved_context_tokens_at_{k}_{suffix}"
        for k in protocol.quality_cutoffs
        for suffix in ("mean", "p95")
    }
    required_operational = {
        "full_stack_latency_ms_p50",
        "full_stack_latency_ms_p95",
        "retrieval_latency_ms_p50",
        "retrieval_latency_ms_p95",
        "cold_initialization_seconds",
    }
    if candidate.reranker == "none":
        required_metrics.update(
            {protocol.pool_recall_metric, "storage.index_size_bytes"}
        )
        if candidate.retriever == "rrf":
            required_metrics.update(
                {"storage.dense_index_size_bytes", "storage.bm25_index_size_bytes"}
            )
        else:
            required_operational.add("index_build_seconds")
    else:
        required_metrics.update(
            {
                "workload.reranker_input_tokens_per_query_mean",
                "workload.reranker_input_tokens_per_query_p95",
            }
        )
        required_operational.update(
            {"reranker_latency_ms_p50", "reranker_latency_ms_p95"}
        )
    errors = []
    for label, names, values in (
        ("candidate", required_metrics, metrics),
        ("operational", required_operational, operational),
    ):
        missing = sorted(name for name in names if values.get(name) is None)
        invalid = sorted(
            name
            for name in names
            if values.get(name) is not None
            and (not np.isfinite(values[name]) or values[name] < 0)
        )
        if missing:
            errors.append(f"missing {label} metrics: " + ", ".join(missing))
        if invalid:
            errors.append(f"invalid {label} metrics: " + ", ".join(invalid))
    if metrics.get("validity.truncated_input_count", 0) != 0:
        errors.append("Prohibited truncated inputs")
    if metrics.get("validity.nonfinite_score_count", 0) != 0:
        errors.append("Non-finite output scores")
    if metrics.get("validity.exact_pool_permutation") != 1:
        errors.append("Required pool permutation verification incomplete or failed")
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


def _index_build_artifact(index, candidate, identity=None):
    identity = dict(identity or _index_identity(index, candidate))
    dependencies = {}
    for name in ("dense", "bm25"):
        checksum = identity.get(f"{name}_index_sha256")
        if checksum is None:
            continue
        dense = name == "dense"
        dependencies[name] = {
            "index_sha256": checksum,
            "chunks_sha256": identity["chunks_sha256"],
            "build_seconds": index.phase_seconds.get(f"{name}_index_build_seconds"),
            "size_bytes": int(index.vectors.nbytes)
            if dense
            else index.bm25.storage_bytes,
            "representation": "raw-contiguous-vector-matrix"
            if dense
            else "bm25-serialized-state",
            "included_components": ["vectors"]
            if dense
            else ["serialized-lexical-state"],
            "excluded_components": [
                "model-weights",
                "saved-pools",
                "canonical-chunk-inputs",
            ],
            "shape": list(index.vectors.shape) if dense else None,
            "dtype": str(index.vectors.dtype) if dense else None,
            "owner_candidate": f"{name}|none",
            "observation_provenance": "candidate-preparation",
        }
    return {
        "status": "complete",
        "chunking_fingerprint": index.chunking_fingerprint,
        **identity,
        "phase_seconds": dict(index.phase_seconds),
        "corpus_build_seconds": index.corpus_build_seconds,
        "dependencies": dependencies,
    }


def _interrupted_index_artifact(progress, candidate):
    """Retain verified completed dependencies, never a partial RRF total."""
    dependencies = {}
    unavailable = {}
    for name in ("dense", "bm25"):
        value = progress.get("vectors" if name == "dense" else "bm25")
        if value is None:
            continue
        view = SimpleNamespace(
            chunks=progress["chunks"],
            vectors=value if name == "dense" else None,
            bm25=value if name == "bm25" else None,
            chunking_fingerprint=progress["chunking_contract"]["fingerprint"],
            phase_seconds=progress.get("phase_seconds", {}),
            corpus_build_seconds=None,
        )
        try:
            component = _index_build_artifact(view, parse_candidate(f"{name}|none"))
        except Exception as exc:  # noqa: BLE001 - retain other dependencies after a failed-state measurement
            unavailable[name] = f"{type(exc).__name__}: {exc}"
        else:
            dependencies.update(component["dependencies"])
    return {
        "status": "incomplete",
        "retriever": candidate.retriever,
        "dependencies": dependencies,
        "unavailable_dependencies": unavailable,
        "phase_seconds": dict(progress.get("phase_seconds", {})),
        "elapsed_to_failure_seconds": progress.get("build_elapsed_to_failure"),
        "phase_elapsed_to_failure_seconds": progress.get(
            "phase_elapsed_to_failure_seconds", {}
        ),
    }


def _reference_dependencies(observed, references):
    result = {**observed, "dependencies": dict(observed["dependencies"])}
    for name, original in observed["dependencies"].items():
        if name not in references:
            continue
        reference = dict(references[name])
        for key in (
            "index_sha256",
            "chunks_sha256",
            "size_bytes",
            "representation",
            "shape",
            "dtype",
        ):
            if reference.get(key) != original.get(key):
                raise ValueError(f"{name} index dependency identity mismatch: {key}")
        result["dependencies"][name] = {
            **reference,
            "observation_provenance": "referenced-independent-build",
        }
    return result


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
