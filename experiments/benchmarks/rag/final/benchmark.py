"""One live selected-server request path and offline Final RAG reporting."""

import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from edumind.common.artifacts import atomic_write_json, stable_hash
from edumind.common.gpu import release_runtime_memory
from experiments.benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.datasets import evidence_units
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.rag.chunking_embedding.metrics import (
    covered_unit_ids,
    score_question,
)
from experiments.benchmarks.rag.generation.evaluate import (
    GenerationPlacementError,
    _empty_attempt,
    score_execution,
)
from experiments.benchmarks.rag.generation.evidence import (
    retain_schedule,
    validate_schedule,
    write_new_json,
)
from experiments.benchmarks.rag.generation.metrics import (
    cell,
    parse_response,
    score_attempt,
)
from experiments.benchmarks.rag.generation.models import generator_for
from experiments.benchmarks.rag.generation.validation import questions_from_manifest
from experiments.benchmarks.rag.retrieval_reranking.metrics import (
    aggregate_quality,
    source_interval,
)


def context_question(question, positions, chunks):
    """Only actual retrieved canonical spans enter the model-visible evidence."""
    units = evidence_units(question)
    blocks, identities = [], {}
    for rank, position in enumerate(positions, 1):
        chunk = chunks[position]
        label = f"E{rank}"
        block = {
            "id": label,
            "text": chunk.text,
            "intervals": [
                {
                    "document_id": chunk.document_id,
                    "start": chunk.start,
                    "end": chunk.end,
                }
            ],
            "gold_unit_ids": sorted(covered_unit_ids(chunk, units)),
        }
        blocks.append(block)
        identities[label] = stable_hash(
            {
                "chunk_id": chunk.identifier,
                "document_id": chunk.document_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
            }
        )
    return blocks, identities


def stable_citations(attempt):
    mapping = attempt["citation_identities"]
    supplied = {block["id"] for block in attempt["live_question"]["blocks"]}
    identities = set()
    for label in attempt["response"]["citations"]:
        if label not in supplied:
            identities.add("unknown:" + label)
            continue
        identity = mapping.get(label)
        if not isinstance(identity, str) or not identity:
            raise ValueError(
                f"Missing or invalid stable citation identity mapping for {label}"
            )
        identities.add("known:" + identity)
    return identities


def _placement(server, backend, device):
    reports = []
    if server.index.embedder is not None:
        reports.append(server.index.embedder.placement_report(device))
    if server.reranker is not None:
        reports.append(server.reranker.placement_report())
    if backend is not None:
        reports.append(backend.placement())
    for report in reports:
        if report.get("status") != "qualified":
            raise GenerationPlacementError(report)
    return reports


def _unload_retrieval(server):
    if server.index.embedder is not None:
        server.index.embedder.unload()
    if server.reranker is not None:
        server.reranker.model = None
    release_runtime_memory()


def execute(
    candidate,
    manifest,
    model_lock,
    plan,
    generation_protocol,
    composition,
    server,
    *,
    backend_factory=generator_for,
    monitor_factory=ResourceMonitor,
    evidence_directory=None,
    preflight=False,
):
    """Server/index preparation must finish before entering this lifecycle."""
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    raw_questions = {
        row["id"]: row for row in manifest.samples if row.get("kind") == "question"
    }
    seeds = list(generation_protocol.measured_seeds[: plan.repetitions])
    schedule = retain_schedule(
        candidate, manifest, model_lock, plan, generation_protocol, questions
    )
    schedule["final_composition"] = composition.checksum
    schedule["final_protocols"] = composition.protocols
    raw = {
        "candidate": candidate,
        "mode": generation_protocol.mode(candidate),
        "model_id": generation_protocol.model_id(candidate),
        "device": plan.settings["device"],
        "dtype": plan.settings["dtype"],
        "schedule": schedule,
        "schedule_checksum": stable_hash(schedule),
        "phase": "loading",
        "error": None,
        "cold_model_load_seconds": None,
        "input_compatibility": {"status": "passed", "questions": []},
        "records": [
            {"question": q, "attempts": [_empty_attempt(seed) for seed in seeds]}
            for q in questions
        ],
        "resource_samples": [],
        "resource_status": "unavailable",
        "monitoring_complete": False,
        "operational": {},
        "composition": composition.system,
        "composition_checksum": composition.checksum,
        "server_index_identity": server.identity,
        "chunk_inventory": [asdict(chunk) for chunk in server.index.chunks],
    }
    if evidence_directory:
        write_new_json(Path(evidence_directory) / "generation_schedule.json", schedule)

    def checkpoint():
        if monitor is not None:
            raw["resource_samples"] = monitor.samples()
            try:
                raw["resource_evidence"] = monitor.metrics()
            except RuntimeError as exc:
                raw["resource_evidence"] = dict(getattr(exc, "resource_evidence", {}))
        if evidence_directory:
            atomic_write_json(
                Path(evidence_directory) / "generation_progress.json", raw
            )

    backend = monitor = None
    residency = composition.system["residency"]
    try:
        monitor = monitor_factory(
            device=raw["device"], require_vram=raw["device"] == "cuda"
        )
        with monitor:
            backend = backend_factory(
                candidate, model_lock, raw["device"], raw["dtype"], generation_protocol
            )
            started = time.perf_counter()
            if residency == "sequential":
                _unload_retrieval(server)
            else:
                if server.index.embedder is not None:
                    server.index.embedder.prepare()
                if server.reranker is not None:
                    server.reranker.prepare()
            backend.load()
            backend.synchronize()
            raw["cold_model_load_seconds"] = time.perf_counter() - started
            raw["placement_before"] = backend.placement()
            _placement(
                server, backend, raw["device"]
            ) if residency == "resident" else _placement_generator(backend)
            raw["resolved_decoder"] = dict(backend.resolved_decoder)
            raw["inherited_generation_config"] = getattr(
                backend, "inherited_generation_config", {}
            )

            def request(q, seed):
                started = time.perf_counter()
                attempt = {
                    **_empty_attempt(seed),
                    "request_status": "failed",
                    "failure_stage": "retrieval",
                    "generation_status": "not_started",
                    "generation_eligible": False,
                    "retrieval": {},
                    "live_question": q,
                    "citation_identities": {},
                }
                try:
                    if residency == "sequential":
                        backend.close()
                    if server.index.embedder is not None:
                        server.index.embedder.prepare()
                    if server.reranker is not None:
                        server.reranker.prepare()
                    attempt["retrieval_placement_before"] = _placement(
                        server, None, raw["device"]
                    )
                    envelope = plan.settings.get("qualified_input_envelope")
                    server.retrieve(
                        raw_questions[q["id"]],
                        evidence=attempt["retrieval"],
                        **({"input_envelope": envelope} if envelope else {}),
                    )
                    attempt["retrieval_placement_after"] = _placement(
                        server, None, raw["device"]
                    )
                    attempt["retrieval"]["latency_seconds"] = (
                        time.perf_counter() - started
                    )
                    positions = attempt["retrieval"]["ranking"][
                        : composition.system["top_k"]
                    ]
                    if positions != attempt["retrieval"]["ranking"][: len(positions)]:
                        raise ValueError(
                            "Selected context is not the validated ranking prefix"
                        )
                    attempt["failure_stage"] = "input_validation"
                    generation_started = time.perf_counter()
                    blocks, identities = context_question(
                        raw_questions[q["id"]], positions, server.index.chunks
                    )
                    live = {**q, "blocks": blocks}
                    attempt.update(live_question=live, citation_identities=identities)
                    if residency == "sequential":
                        _unload_retrieval(server)
                        backend.load()
                    _placement_generator(backend)
                    attempt["generator_placement_before"] = backend.placement()
                    prepared = backend.prepare(live)
                    prompt_tokens = prepared["prompt_tokens"]
                    check = {
                        "question_id": q["id"],
                        "seed": seed,
                        "prompt_tokens": prompt_tokens,
                    }
                    raw["input_compatibility"]["questions"].append(check)
                    if not 0 < prompt_tokens < generation_protocol.context_tokens:
                        raw["input_compatibility"]["status"] = "failed"
                        raise ValueError(
                            "input_context_overflow: actual packed Final prompt"
                        )
                    envelope = plan.settings.get("qualified_input_envelope")
                    if envelope and prompt_tokens > envelope["maximum_prompt_tokens"]:
                        raise ValueError("Final preflight input envelope exceeded")
                    preparation = time.perf_counter() - generation_started
                    attempt.update(
                        generation_status="started",
                        generation_eligible=True,
                        failure_stage="generation",
                    )
                    produced = backend.infer(prepared, seed)
                    # Retain the returned stream before synchronization/placement
                    # validation can fail independently of generation.
                    attempt.update(produced, generation_status=produced["outcome"])
                    timing = dict(produced.get("timing", {}))
                    for key in ("first_token_seconds", "last_token_seconds"):
                        if timing.get(key) is not None:
                            timing[key] += preparation
                    timing["generation_call_offset_seconds"] = (
                        timing.get("generation_call_offset_seconds", 0) + preparation
                    )
                    timing["latency_seconds"] = time.perf_counter() - generation_started
                    attempt.update(
                        produced, timing=timing, generation_status=produced["outcome"]
                    )
                    if isinstance(produced.get("token_event_seconds"), list):
                        attempt["token_event_seconds"] = [
                            t + preparation for t in produced["token_event_seconds"]
                        ]
                    attempt["failure_stage"] = "post_inference_integrity"
                    backend.synchronize()
                    _placement_generator(backend)
                    attempt["generator_placement_after"] = backend.placement()
                    response, _ = parse_response(attempt.get("visible_text", ""))
                    attempt["request_status"] = (
                        "completed"
                        if produced["outcome"] == "completed" and response is not None
                        else "failed"
                    )
                    if produced["outcome"] == "completed" and response is None:
                        attempt["failure_stage"] = "output_schema"
                    if attempt["request_status"] == "completed":
                        attempt["generation_latency_seconds"] = (
                            time.perf_counter() - generation_started
                        )
                        attempt["failure_stage"] = None
                    else:
                        attempt["generation_elapsed_to_failure_seconds"] = (
                            time.perf_counter() - generation_started
                        )
                except Exception as exc:  # noqa: BLE001 - retain stage-complete evidence
                    attempt["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, GenerationPlacementError):
                        raw["placement"] = exc.report
                        raw["error"] = f"{type(exc).__name__}: {exc}"
                        attempt["integrity_error"] = exc.report
                    elif attempt["failure_stage"] == "post_inference_integrity":
                        raw["error"] = f"{type(exc).__name__}: {exc}"
                        attempt["integrity_error"] = {
                            "status": "placement_unverifiable",
                            "reason": raw["error"],
                        }
                    elif (
                        attempt["generation_eligible"]
                        and attempt["failure_stage"] != "post_inference_integrity"
                    ):
                        attempt.update(
                            outcome="runtime_error", generation_status="runtime_error"
                        )
                    elif not attempt["generation_eligible"]:
                        attempt["outcome"] = (
                            "input_rejected"
                            if attempt["failure_stage"] == "input_validation"
                            else "not_executed"
                        )
                attempt["request_elapsed_seconds"] = time.perf_counter() - started
                if attempt["request_status"] == "completed":
                    attempt["request_latency_seconds"] = attempt[
                        "request_elapsed_seconds"
                    ]
                return attempt

            if not preflight:
                raw["phase"] = "warmup"
                warmup = request(questions[0], generation_protocol.warmup_seed)
                if (
                    warmup["request_status"] != "completed"
                    or parse_response(warmup.get("visible_text", ""))[0] is None
                ):
                    raise ValueError("Final complete-request warmup failed")
            raw["phase"] = "measured"
            for record in raw["records"]:
                for number, seed in enumerate(seeds):
                    raw["in_flight"] = {
                        "question_id": record["question"]["id"],
                        "seed": seed,
                    }
                    checkpoint()
                    record["attempts"][number] = request(record["question"], seed)
                    raw.pop("in_flight", None)
                    checkpoint()
                    if record["attempts"][number].get("integrity_error"):
                        raise GenerationPlacementError(
                            record["attempts"][number]["integrity_error"]
                        )
            raw["phase"] = "finished"
        raw["monitoring_complete"] = True
    except Exception as exc:  # noqa: BLE001 - loading, placement and telemetry are distinct
        raw["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if monitor is not None:
            raw["resource_samples"] = monitor.samples()
            raw["vram_measurement_method"] = monitor.vram_measurement_method
            try:
                resource = monitor.metrics()
                raw["resource_status"] = "observed"
            except RuntimeError as exc:
                resource = getattr(exc, "resource_evidence", {})
                raw["resource_error"] = str(exc)
            raw["operational"] = {
                "peak_ram_mib": resource.get("peak_process_tree_ram_mb"),
                "peak_vram_mib": resource.get("peak_vram_mb")
                if raw["device"] == "cuda"
                else None,
            }
        try:
            if backend is not None:
                backend.close()
        except Exception as exc:  # noqa: BLE001 - retain outputs when cleanup fails
            raw["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            raw["error"] = raw["error"] or raw["cleanup_error"]
        finally:
            try:
                _unload_retrieval(server)
            except Exception as exc:  # noqa: BLE001 - checkpoint even on second cleanup failure
                raw["retrieval_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                raw["error"] = raw["error"] or raw["retrieval_cleanup_error"]
            checkpoint()
    return raw


def _placement_generator(backend):
    report = backend.placement()
    if report.get("status") != "qualified":
        raise GenerationPlacementError(report)


def validate_final_schedule(raw, protocol, profile, *, judge=None):
    if raw["schedule"].get("final_composition") != raw.get("composition_checksum"):
        raise ValueError("Final composition execution identity mismatch")
    records = validate_schedule(raw, protocol, profile)
    if judge is not None:
        retained = raw["schedule"]["evaluator"]
        if judge.contract.get("contract_sha256") != retained.get(
            "contract_sha256"
        ) or judge.contract.get("checksum") != retained.get("contract_checksum"):
            raise ValueError(
                "Saved Final execution and semantic judge identity mismatch"
            )
    return records


def _score_attempt(question, attempt, judgment):
    result = score_attempt(question, attempt, judgment)
    if not attempt.get("generation_eligible", False):
        # A known upstream failure is not a generator request or model error.
        for key in result["metrics"]:
            if (
                key.startswith("reliability.")
                or key == "validity.malformed_output_rate"
            ):
                result["metrics"][key] = cell(
                    status="inapplicable", reason="upstream_stage_not_started"
                )
        if attempt.get("request_status") == "failed":
            key = (
                "validity.response_validity_rate"
                if question["answerable"]
                else "validity.refusal_validity_rate"
            )
            result["metrics"][key] = cell(0.0)
    return result


def score(raw, manifest, generation_protocol, retrieval_protocol, *, judge=None):
    profile = raw["schedule"]["profile"]
    failure = None
    try:
        evaluated = score_execution(
            raw,
            generation_protocol,
            profile,
            judge=judge,
            record_validator=partial(validate_final_schedule, judge=judge),
            question_selector=lambda record, attempt: attempt.get(
                "live_question", record["question"]
            ),
            citation_identity=stable_citations,
            attempt_scorer=_score_attempt,
            execution_predicate=lambda record, attempt: (
                attempt.get("request_status") in {"completed", "failed"}
            ),
            pair_failure_predicate=lambda attempt: (
                attempt.get("request_status") == "failed"
            ),
        )
    except CandidateExecutionError as exc:
        failure = exc
        evaluated = (
            list(exc.samples),
            exc.operational,
            exc.metrics,
            exc.parameters,
            exc.intervals,
            exc.artifacts,
        )
    samples, operational, metrics, parameters, intervals, artifacts = evaluated
    operational, metrics, intervals, parameters, artifacts = map(
        dict, (operational, metrics, intervals, parameters, artifacts)
    )
    index_chunks = raw["chunk_inventory"]
    from edumind.rag.tokenizers import TiktokenOffsetTokenizer
    from experiments.benchmarks.rag.evaluation import Chunk

    chunks = [Chunk(**row) for row in index_chunks]
    questions = {q["id"]: q for q in manifest.samples if q.get("kind") == "question"}
    cutoff = raw["composition"]["top_k"]
    cutoffs = [k for k in retrieval_protocol.quality_cutoffs if k <= cutoff]
    quality_rows, timing_groups, request_rows = [], {}, []
    for record in raw["records"]:
        q = record["question"]
        first = record["attempts"][0]
        values = {}
        ranking = first.get("retrieval", {}).get("ranking")
        if q["answerable"]:
            names = [
                f"diagnostic.{name}_at_{k}"
                for name in (
                    "ndcg",
                    "evidence_unit_recall",
                    "evidence_token_precision",
                    "alpha_ndcg",
                )
                for k in cutoffs
            ]
            values = dict.fromkeys(names)
            values["diagnostic.pool_evidence_unit_recall_at_20"] = None
            if ranking is not None:
                selected = [chunks[i] for i in ranking[:cutoff]]
                scored = score_question(
                    questions[q["id"]],
                    selected,
                    chunks,
                    TiktokenOffsetTokenizer("cl100k_base"),
                    cutoffs=cutoffs,
                    alpha=retrieval_protocol.alpha_ndcg_alpha,
                )
                values.update(
                    {
                        "diagnostic." + key: value
                        for key, value in scored.metrics.items()
                    }
                )
            pool = first.get("retrieval", {}).get("pool")
            if pool is not None:
                units = evidence_units(questions[q["id"]])
                recovered = set().union(
                    *(covered_unit_ids(chunks[i], units) for i in pool)
                )
                values["diagnostic.pool_evidence_unit_recall_at_20"] = len(
                    recovered
                ) / len(units)
        executed = [a for a in record["attempts"] if "request_status" in a]
        values["reliability.request_failure_rate"] = (
            sum(a["request_status"] != "completed" for a in executed) / len(executed)
            if len(executed) == len(record["attempts"])
            else None
        )
        quality_rows.append(
            SampleResult(
                q["id"],
                values,
                None,
                {
                    "document_id": q["document_id"],
                    "source_id": q["source_id"],
                    "independent_source_id": q["source_id"],
                    "evidence_type": q["evidence_type"],
                    "answerable": q["answerable"],
                },
            )
        )
        for a in record["attempts"]:
            request_rows.append(
                {
                    "question_id": q["id"],
                    "seed": a["seed"],
                    "request_status": a.get("request_status", "not_executed"),
                    "failure_stage": a.get("failure_stage"),
                    "generation_status": a.get("generation_status", "not_started"),
                    "evidence": json.dumps(a, ensure_ascii=False),
                }
            )
        group = timing_groups.setdefault(q["source_id"], [])
        group.append(
            SampleResult(
                q["id"], {}, None, {"document_id": q["document_id"], "record": record}
            )
        )
    names = sorted(set().union(*(row.metrics for row in quality_rows)))
    support = {}
    for diagnostic in (False, True):
        cohort = [
            row for row in quality_rows if not diagnostic or row.metadata["answerable"]
        ]
        selected_names = [
            name for name in names if name.startswith("diagnostic.") == diagnostic
        ]
        added, bounds, details = aggregate_quality(
            cohort,
            names=selected_names,
            resamples=generation_protocol.profile(profile).bootstrap_resamples,
            seed=generation_protocol.meta.seed,
            confidence=generation_protocol.confidence_level,
            minimum_documents=retrieval_protocol.minimum_quality_ci_documents,
        )
        metrics.update(added)
        intervals.update(bounds)
        support.update(details)
    for field, path in (
        ("request", "request_latency_seconds"),
        ("retrieval", "retrieval"),
        ("generation", "generation_latency_seconds"),
        ("server_call", "server"),
    ):

        def durations(groups, path=path):
            result = []
            for group in groups:
                for row in group:
                    record = row.metadata["record"]
                    values = []
                    for a in record["attempts"]:
                        if path == "generation_latency_seconds" and (
                            a.get("request_status") != "completed"
                            or a.get("outcome") != "completed"
                            or parse_response(a.get("visible_text", ""))[0] is None
                        ):
                            continue
                        value = (
                            a.get("retrieval", {}).get("latency_seconds")
                            if path == "retrieval"
                            else a.get("retrieval", {}).get("database_latency_seconds")
                            if path == "server"
                            else a.get(path)
                        )
                        if value is not None:
                            values.append(value)
                    if values:
                        result.append(float(np.median(values)))
            return result

        for percentile, quantile in (("p50", 0.5), ("p95", 0.95)):

            def statistic(groups, quantile=quantile, durations=durations):
                observed = durations(groups)
                return float(np.quantile(observed, quantile)) if observed else None

            key = f"{field}_{percentile}_seconds"
            operational[key] = statistic(list(timing_groups.values()))
            intervals["operational." + key] = source_interval(
                list(timing_groups.values()),
                statistic,
                resamples=generation_protocol.profile(profile).bootstrap_resamples,
                seed=generation_protocol.meta.seed,
                confidence=generation_protocol.confidence_level,
                minimum_documents=retrieval_protocol.minimum_latency_ci_documents,
            )
    artifacts.update(
        final_attempts=request_rows,
        retrieval_metric_support=support,
        final_composition={
            "system": raw["composition"],
            "checksum": raw["composition_checksum"],
        },
    )
    parameters.update(
        composition_checksum=raw["composition_checksum"], run_type="selected-system"
    )
    complete_retrieval = all(row["complete"] for row in support.values())
    if failure or not complete_retrieval:
        raise CandidateExecutionError(
            str(failure) if failure else "Final retrieval quality incomplete",
            samples=tuple(samples),
            operational=operational,
            metrics=metrics,
            parameters=parameters,
            intervals=intervals,
            artifacts=artifacts,
        )
    return samples, operational, metrics, parameters, intervals, artifacts
