"""Generation execution and offline evaluation; no retrieval or live judge API.

The old evaluate_candidate keyword interface is retained. Retrieved-context
integration must migrate explicitly and fails closed instead of using lexical
or HHEM scores. Raw generation finishes before any semantic judging begins.
"""

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from edumind.common.artifacts import atomic_write_json, sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.process import (
    benchmark_objects_from_payload,
    run_json_worker,
)
from experiments.benchmarks.common.resources import ResourceMonitor

from .judge import load_judge_contract
from .metrics import (
    BASE_KEYS,
    GENERATION_DIRECTIONS,
    OPERATIONAL_KEYS,
    aggregate,
    finish_inventory,
    parse_response,
    score_attempt,
    score_pairs,
)
from .models import generator_for
from .protocol import load_protocol
from .validation import questions_from_manifest

__all__ = ["GENERATION_DIRECTIONS", "evaluate_candidate"]


class GenerationPlacementError(RuntimeError):
    """Typed placement evidence must survive generic worker error handling."""

    def __init__(self, report):
        self.report = report
        self.reason_code = report.get("status", "placement_unverifiable")
        super().__init__(self.reason_code)


def _empty_attempt(seed, reason="not_started"):
    return {
        "seed": seed,
        "outcome": "not_executed",
        "finish_reason": None,
        "error": reason,
        "raw_text": None,
        "visible_text": "",
        "timing": {},
        "workload": {},
    }


def _instrumentation_issues(raw):
    issues = []

    def number(value):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    for record in raw["records"]:
        for attempt in record["attempts"]:
            if attempt["outcome"] != "completed":
                continue  # Known failed requests have conditional timing cohorts.
            label = f"{record['question']['id']}/{attempt['seed']}"
            timing, workload = attempt.get("timing", {}), attempt.get("workload", {})
            for field in ("latency_seconds", "generation_seconds"):
                if not number(timing.get(field)):
                    issues.append(label + ": missing/invalid " + field)
            for field in ("prompt_tokens", "context_tokens", "total_output_tokens"):
                value = workload.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    issues.append(label + ": missing/invalid " + field)
            events, count = (
                attempt.get("token_event_seconds"),
                workload.get("total_output_tokens"),
            )
            if (
                not isinstance(events, list)
                or any(not number(v) for v in events)
                or events != sorted(events)
                or len(events) != count
            ):
                issues.append(label + ": missing/invalid token event inventory")
                continue
            if events:
                for field, expected in (
                    (
                        "ttft_seconds",
                        events[0] - timing.get("generation_call_offset_seconds", 0),
                    ),
                    ("first_token_seconds", events[0]),
                    ("last_token_seconds", events[-1]),
                ):
                    if not number(timing.get(field)) or not math.isclose(
                        timing[field], expected
                    ):
                        issues.append(label + ": missing/inconsistent " + field)
            response, _ = parse_response(attempt.get("visible_text", ""))
            if response is not None:
                value = workload.get("visible_answer_tokens")
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    issues.append(label + ": missing/invalid visible_answer_tokens")
    return issues


def _merge_supervision(raw):
    """Keep real device-total peaks from either observer, never subtract baseline."""
    supervision = raw.get("_worker_supervision", {})
    operational = dict(raw.get("operational", {}))
    values = [operational.get("peak_vram_mib"), supervision.get("peak_vram_mb")]
    values = [
        v
        for v in values
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and math.isfinite(v)
        and v >= 0
    ]
    operational["peak_vram_mib"] = (
        max(values) if values and raw.get("device") != "cpu" else None
    )
    # Supervisor RAM includes this parent and is not the worker process tree.
    return {
        **raw,
        "operational": operational,
        "supervisor_resource_samples": supervision.get("resource_samples", []),
        "supervisor_resource_evidence": supervision,
    }


def _plan_settings(plan, protocol, preflight):
    if preflight:
        expected = protocol.preflight
    else:
        expected = protocol.profile(plan.profile)
    if (plan.seed, plan.warmups, plan.repetitions, plan.bootstrap_resamples) != (
        protocol.meta.seed,
        expected.warmups,
        expected.repetitions,
        expected.bootstrap_resamples,
    ):
        raise ValueError("Generation plan does not match its resolved protocol")
    device, dtype = plan.settings["device"], plan.settings["dtype"]
    profile = protocol.profile("development" if preflight else plan.profile)
    if device not in (
        ("cpu", "cuda") if plan.profile == "smoke" else (profile.device,)
    ) or dtype != profile.dtype_for(device):
        raise ValueError(
            "Generation device/dtype override violates the resolved protocol"
        )


def execute_generation(
    candidate,
    manifest,
    model_lock,
    plan,
    protocol,
    *,
    backend_factory=generator_for,
    monitor_factory=ResourceMonitor,
    preflight=False,
    evidence_directory=None,
):
    """Run one fresh backend lifecycle; injectable factories require no GPU/model."""
    _plan_settings(plan, protocol, preflight)
    questions = questions_from_manifest(manifest)
    seeds = protocol.measured_seeds[: plan.repetitions]
    result = {
        "candidate": candidate,
        "mode": protocol.mode(candidate),
        "model_id": protocol.model_id(candidate),
        "device": plan.settings["device"],
        "dtype": plan.settings["dtype"],
        "phase": "input_compatibility",
        "error": None,
        "cold_model_load_seconds": None,
        "input_compatibility": {"status": "failed", "questions": []},
        "records": [
            {"question": q, "attempts": [_empty_attempt(seed) for seed in seeds]}
            for q in questions
        ],
        "resource_samples": [],
        "resource_status": "unavailable",
        "monitoring_complete": False,
        "placement_before": None,
        "placement_after": None,
        "operational": {},
    }
    backend = monitor = None

    def checkpoint():
        if evidence_directory is not None:
            atomic_write_json(
                Path(evidence_directory) / "generation_progress.json", result
            )

    try:
        backend = backend_factory(
            candidate,
            model_lock,
            plan.settings["device"],
            plan.settings["dtype"],
            protocol,
        )
        prepared = [backend.prepare(q) for q in questions]
        checks = [
            {
                "question_id": q["id"],
                "candidate": candidate,
                "prompt_tokens": p["prompt_tokens"],
                "context_capacity": protocol.context_tokens,
                "output_capacity": protocol.context_tokens - p["prompt_tokens"],
                "status": "passed"
                if 0 < p["prompt_tokens"] < protocol.context_tokens
                else "failed",
            }
            for q, p in zip(questions, prepared, strict=True)
        ]
        result["input_compatibility"] = {
            "status": "passed"
            if all(c["status"] == "passed" for c in checks)
            else "failed",
            "questions": checks,
        }
        result["prompts"] = [
            {"question_id": q["id"], **p}
            for q, p in zip(questions, prepared, strict=True)
        ]
        checkpoint()
        if result["input_compatibility"]["status"] != "passed":
            raise ValueError(
                "input_context_overflow: complete native prompt leaves no output capacity"
            )
        if not preflight and plan.profile != "smoke":
            envelope = plan.settings.get("qualified_input_envelopes", {}).get(candidate)
            limit = (
                envelope.get("maximum_prompt_tokens")
                if isinstance(envelope, dict)
                else None
            )
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or not 0 < limit < protocol.context_tokens
                or envelope.get("context_tokens") != protocol.context_tokens
            ):
                raise ValueError("preflight_input_envelope_unverified: rerun preflight")
            result["qualified_input_envelope"] = dict(envelope)
            if any(check["prompt_tokens"] > limit for check in checks):
                raise ValueError(
                    "preflight_input_envelope_exceeded: select a longer reviewed development stress input and requalify"
                )
        monitor = (
            monitor_factory(
                interval_seconds=protocol.preflight.telemetry_interval_seconds,
                require_vram=plan.settings["device"] == "cuda",
                device=plan.settings["device"],
            )
            if monitor_factory
            else None
        )
        with monitor if monitor is not None else nullcontext():
            result["phase"] = "loading"
            checkpoint()
            started = time.perf_counter()
            backend.load()
            backend.synchronize()
            result["cold_model_load_seconds"] = time.perf_counter() - started
            result["resolved_decoder"] = dict(backend.resolved_decoder)
            result["inherited_generation_config"] = getattr(
                backend, "inherited_generation_config", {}
            )
            result["decoder_source"] = getattr(backend, "decoder_source", None)
            result["placement_before"] = backend.placement()
            if result["placement_before"].get("status") != "qualified":
                raise GenerationPlacementError(result["placement_before"])
            if preflight:
                stress_ids = {
                    row["id"]
                    for row in manifest.samples
                    if row.get("kind") == "question"
                    and row.get("preflight_stress") is True
                }
                choices = [
                    (q, p)
                    for q, p in zip(questions, prepared, strict=True)
                    if q["id"] in stress_ids
                ]
                if not choices:
                    raise ValueError(
                        "missing_stress_input: mark reviewed development/stress question(s) preflight_stress=true"
                    )
                q, p = max(
                    choices, key=lambda pair: (pair[1]["prompt_tokens"], pair[0]["id"])
                )
                result["phase"] = "stress_inference"
                result["stress_question_id"] = q["id"]
                result["tested_input_envelope"] = {
                    "maximum_prompt_tokens": p["prompt_tokens"],
                    "context_tokens": protocol.context_tokens,
                    "stress_question_id": q["id"],
                }
                result["stress_output"] = backend.infer(p, protocol.measured_seeds[0])
                backend.synchronize()
                if result["stress_output"]["outcome"] != "completed":
                    raise RuntimeError(
                        "stress_inference_incomplete: "
                        + result["stress_output"]["outcome"]
                    )
            else:
                result["phase"] = "warmup"
                checkpoint()
                result["warmup"] = backend.infer(
                    backend.prepare(questions[0]), protocol.warmup_seed
                )
                backend.synchronize()
                warmup_response, _ = parse_response(
                    result["warmup"].get("visible_text", "")
                )
                valid_ids = {b["id"] for b in questions[0]["blocks"]}
                if (
                    result["warmup"]["outcome"] != "completed"
                    or warmup_response is None
                    or not set(warmup_response["citations"]) <= valid_ids
                ):
                    raise ValueError(
                        "Warmup did not complete with a valid structural/ID envelope"
                    )
                result["phase"] = "measured"
                for record in result["records"]:
                    q = record["question"]
                    for index, seed in enumerate(seeds):
                        result["in_flight"] = {"question_id": q["id"], "seed": seed}
                        checkpoint()
                        started = time.perf_counter()
                        try:
                            request = backend.prepare(q)
                            if request["prompt_tokens"] != next(
                                c["prompt_tokens"]
                                for c in checks
                                if c["question_id"] == q["id"]
                            ):
                                raise ValueError(
                                    "Native prompt changed after compatibility verification"
                                )
                            preparation_seconds = time.perf_counter() - started
                            attempt = backend.infer(request, seed)
                            # Events use request-relative time; model TTFT stays
                            # relative to generation, not prompt preparation.
                            timing = dict(attempt.get("timing", {}))
                            for field in (
                                "first_token_seconds",
                                "last_token_seconds",
                            ):
                                if timing.get(field) is not None:
                                    timing[field] += preparation_seconds
                            timing["generation_call_offset_seconds"] = (
                                timing.get("generation_call_offset_seconds", 0)
                                + preparation_seconds
                            )
                            attempt["timing"] = timing
                            if isinstance(attempt.get("token_event_seconds"), list):
                                attempt["token_event_seconds"] = [
                                    value + preparation_seconds
                                    for value in attempt["token_event_seconds"]
                                ]
                            backend.synchronize()
                            response, _ = (
                                parse_response(attempt.get("visible_text", ""))
                                if attempt["outcome"] == "completed"
                                else (None, None)
                            )
                            # Include complete output parsing and supplied-ID checks.
                            if response is not None:
                                set(response["citations"]) - {
                                    b["id"] for b in q["blocks"]
                                }
                        except Exception as exc:  # noqa: BLE001 - retain arbitrary backend failures
                            attempt = {
                                **_empty_attempt(seed),
                                "outcome": "runtime_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "finish_reason": "backend_exception",
                            }
                        elapsed = time.perf_counter() - started
                        attempt = {
                            **attempt,
                            "seed": seed,
                            "elapsed_seconds": elapsed,
                            "timing": {
                                **attempt.get("timing", {}),
                                "latency_seconds": elapsed,
                            },
                        }
                        record["attempts"][index] = attempt
                        result.pop("in_flight", None)
                        checkpoint()
            result["placement_after"] = backend.placement()
            if result["placement_after"].get("status") != "qualified":
                raise GenerationPlacementError(result["placement_after"])
            backend.synchronize()
        result["monitoring_complete"] = True
        result["phase"] = "finished"
    except Exception as exc:  # noqa: BLE001 - preserve lifecycle and placement evidence
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, GenerationPlacementError):
            result.update(failure_class=exc.reason_code, placement=exc.report)
    finally:
        # Stop the generator monitor and release weights BEFORE offline judging.
        if monitor is not None:
            result["resource_samples"] = monitor.samples()
            result["vram_measurement_method"] = monitor.vram_measurement_method
            result["gpu_identity"] = dict(getattr(monitor, "gpu_identity", {}))
            try:
                resource = monitor.metrics()
                result["resource_status"] = "observed"
            except RuntimeError as exc:
                resource = getattr(exc, "resource_evidence", {})
                result["resource_error"] = str(exc)
            result["operational"] = {
                "peak_ram_mib": resource.get("peak_process_tree_ram_mb"),
                "peak_vram_mib": resource.get("peak_vram_mb")
                if plan.settings["device"] == "cuda"
                else None,
            }
            if plan.settings["device"] == "cpu":
                result["vram_measurement_method"] = "not-applicable"
        if backend is not None:
            backend.close()
        checkpoint()
    return result


def score_execution(raw, protocol, profile, *, judge=None):
    """Offline rescoring of retained generator evidence; injectable/replay judge."""
    records = []
    for record in raw["records"]:
        q, attempts = record["question"], []
        for attempt in record["attempts"]:
            response, _ = (
                parse_response(attempt.get("visible_text", ""))
                if attempt["outcome"] == "completed"
                else (None, None)
            )
            judgment = None
            if response is not None and q["answerable"] and judge is not None:
                try:
                    judgment = judge.response(q, response)
                except Exception as exc:  # noqa: BLE001 - evaluator failures are not model failures
                    judgment = {
                        "status": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
            attempts.append(score_attempt(q, attempt, judgment))
        records.append(
            {
                "question": q,
                "attempts": attempts,
                "pairs": score_pairs(q, attempts, judge),
            }
        )
    execution = protocol.profile(profile)
    summary = aggregate(
        records,
        profile=profile,
        resamples=execution.bootstrap_resamples,
        seed=protocol.meta.seed,
        confidence=protocol.confidence_level,
        minimum_ci_sources=protocol.minimum_ci_sources,
    )
    # Input rejection or loading/warmup failure does not fabricate reliability events.
    operational = {
        **raw.get("operational", {}),
        "cold_model_load_seconds": raw.get("cold_model_load_seconds"),
    }
    for key in OPERATIONAL_KEYS:
        operational[key.removeprefix("operational.")] = summary[key]["value"]
    lifecycle_complete = raw.get("phase") == "finished" and raw.get("error") is None
    quality_complete = all(
        row["evidence_complete"] for key, row in summary.items() if key in BASE_KEYS
    )
    problems = []
    if not lifecycle_complete:
        problems.append(raw.get("error") or "Generator lifecycle incomplete")
    if raw.get("resource_status") != "observed":
        problems.append("Required lifecycle resource telemetry unavailable")
    for field in (
        "peak_ram_mib",
        *(("peak_vram_mib",) if raw.get("device") == "cuda" else ()),
    ):
        value = operational.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            problems.append(f"Required {field} telemetry unavailable")
    instrumentation_issues = _instrumentation_issues(raw)
    if instrumentation_issues:
        problems.append("Required timing/token instrumentation incomplete")
    if profile != "smoke":
        if not quality_complete:
            problems.append("Required output/semantic evaluation incomplete")
        if judge is None:
            problems.append("judge_calibration_pending")
        if any(
            row["interval"]["reason"] == "ci_support_not_frozen"
            for key, row in summary.items()
            if not key.startswith("workload.") and row["eligible_questions"]
        ):
            problems.append("ci_support_policy_pending")
    samples = []
    for record in records:
        q = record["question"]
        single = aggregate([record], profile="smoke", resamples=0)
        samples.append(
            SampleResult(
                q["id"],
                {key: single[key]["value"] for key in BASE_KEYS},
                single["operational.latency_p50_seconds"]["value"],
                {
                    "document_id": q["document_id"],
                    "source_id": q["source_id"],
                    "answerable": q["answerable"],
                    "evidence_type": q["evidence_type"],
                },
            )
        )
    metrics = {
        key: None
        if (key.startswith("quality.") or ".quality." in key)
        and not value["evidence_complete"]
        else value["value"]
        for key, value in summary.items()
        if not key.startswith("operational.")
    }
    intervals = {
        key: value["interval"]
        for key, value in summary.items()
        if not key.startswith("workload.")
    }
    parameters = {
        "input_compatibility": raw["input_compatibility"]["status"],
        "result_scope": "wiring_only"
        if profile == "smoke"
        else "authoritative_contract",
        "result_status": "incomplete" if problems else "complete",
        "quality_complete": quality_complete,
        "mode": raw["mode"],
        "model_id": raw["model_id"],
        "measured_seeds": list(protocol.measured_seeds[: execution.repetitions]),
        "resolved_decoder": raw.get("resolved_decoder"),
        "inherited_generation_config": raw.get("inherited_generation_config"),
        "measurement.resource_scope": "complete"
        if raw.get("monitoring_complete")
        else "partial",
        "vram_measurement_method": raw.get("vram_measurement_method", "unavailable"),
        "semantic_judge": judge.contract
        if judge is not None
        else {"status": "not_configured"},
    }
    artifacts = {
        "generation_evidence": {**raw, "records": records},
        "metric_support": summary,
        "instrumentation_issues": instrumentation_issues,
        "input_compatibility": raw["input_compatibility"],
        "finish_reasons": finish_inventory(records),
        "timings": [
            {
                "question_id": r["question"]["id"],
                "seed": a["seed"],
                "outcome": a["outcome"],
                "finish_reason": a.get("finish_reason"),
                "elapsed_seconds": a.get("elapsed_seconds"),
                **a.get("timing", {}),
                **a.get("workload", {}),
            }
            for r in records
            for a in r["attempts"]
        ],
    }
    if problems:
        raise CandidateExecutionError(
            "; ".join(problems),
            samples=tuple(samples),
            metrics=metrics,
            operational=operational,
            intervals=intervals,
            parameters=parameters,
            artifacts=artifacts,
        )
    return samples, operational, metrics, parameters, intervals, artifacts


def execute_payload(payload):
    candidate, manifest, model_lock, plan = benchmark_objects_from_payload(payload)
    protocol = load_protocol(Path(payload["protocol_path"]))
    protocol.meta.validate_worker_payload(payload["protocol"])
    return execute_generation(
        candidate,
        manifest,
        model_lock,
        plan,
        protocol,
        preflight=bool(payload.get("preflight")),
        evidence_directory=payload.get("evidence_directory"),
    )


def run_in_fresh_process(
    candidate,
    manifest,
    model_lock,
    plan,
    protocol,
    *,
    judge=None,
    preflight=False,
    evidence_directory=None,
):
    evidence_directory = (
        Path(evidence_directory)
        if evidence_directory is not None
        else (
            PROJECT_ROOT
            / "artifacts/benchmarks/rag/generation/worker-evidence"
            / uuid4().hex
        )
    )
    payload = {
        "candidate": candidate,
        "manifest": asdict(manifest),
        "model_lock": model_lock,
        "plan": asdict(plan),
        "protocol_path": str(protocol.meta.source_path),
        "protocol": protocol.meta.worker_payload(),
        "preflight": preflight,
        "evidence_directory": str(evidence_directory.resolve()),
    }
    device = plan.settings["device"]
    try:
        raw = run_json_worker(
            Path(__file__).with_name("worker.py"),
            payload,
            device=device,
            prefix="edumind-generation-",
            error_label="Generation worker",
            vram_limit_mb=protocol.authoritative_peak_vram_mb
            if device == "cuda"
            else None,
            require_vram_measurement=device == "cuda",
            telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
            poll_interval_seconds=protocol.preflight.poll_interval_seconds,
            timeout_seconds=protocol.preflight.worker_timeout_seconds,
        )
    except CandidateExecutionError as exc:
        checkpoint = evidence_directory / "generation_progress.json"
        if checkpoint.is_file():
            try:
                raw = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (
                    not isinstance(raw, dict)
                    or raw.get("candidate") != candidate
                    or not isinstance(raw.get("records"), list)
                ):
                    raise ValueError(
                        "Checkpoint has no matching candidate/record inventory"
                    )
            except (ValueError, OSError):
                exc.artifacts["prediction_evidence"] = {
                    "path": str(checkpoint),
                    "status": "unreadable",
                }
            else:
                raw.update(
                    phase="interrupted", error=str(exc), monitoring_complete=False
                )
                in_flight = raw.get("in_flight")
                if in_flight:
                    for record in raw["records"]:
                        for attempt in record["attempts"]:
                            if (
                                record["question"]["id"] == in_flight["question_id"]
                                and attempt["seed"] == in_flight["seed"]
                            ):
                                attempt.update(
                                    outcome="runtime_error",
                                    error=str(exc),
                                    finish_reason="worker_terminated",
                                )
                raw["_worker_supervision"] = getattr(exc, "resource_evidence", {})
                raw = _merge_supervision(raw)
                exc.artifacts["generation_evidence"] = raw
                exc.parameters["prediction_evidence"] = {
                    "path": str(checkpoint.resolve()),
                    "sha256": sha256_file(checkpoint),
                }
                if not preflight:
                    try:
                        score_execution(raw, protocol, plan.profile, judge=judge)
                    except CandidateExecutionError as scored:
                        exc.samples, exc.metrics, exc.intervals = (
                            scored.samples,
                            scored.metrics,
                            scored.intervals,
                        )
                        exc.artifacts.update(scored.artifacts)
                        exc.parameters.update(scored.parameters)
                        exc.operational = scored.operational
                else:
                    exc.resource_evidence = {
                        **getattr(exc, "resource_evidence", {}),
                        "prediction_evidence": exc.parameters["prediction_evidence"],
                        "worker_result": raw,
                    }
        raise
    raw = _merge_supervision(raw)
    checkpoint = evidence_directory / "generation_progress.json"
    if checkpoint.is_file():
        raw["prediction_evidence"] = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
        }
    if preflight:
        if raw.get("error") and raw.get("failure_class") not in {
            "offload_detected",
            "placement_unverifiable",
            "wrong_device",
        }:
            error = RuntimeError(raw["error"])
            error.resource_evidence = raw
            raise error
        return {
            "placement": raw.get("placement") or raw["placement_after"],
            "placement_before": raw["placement_before"],
            "placement_after": raw["placement_after"],
            "vram_measurement_method": raw.get("vram_measurement_method"),
            "peak_vram_mb": raw.get("operational", {}).get("peak_vram_mib"),
            "resource_samples": raw.get("resource_samples", []),
            "input_validation": raw["input_compatibility"],
            "supervisor_resource_samples": raw.get("supervisor_resource_samples", []),
            "stress_question_id": raw.get("stress_question_id"),
            "tested_input_envelope": raw.get("tested_input_envelope"),
            "stress_output": raw.get("stress_output"),
            "resolved_decoder": raw.get("resolved_decoder"),
            "prediction_evidence": raw.get("prediction_evidence"),
        }
    return score_execution(raw, protocol, plan.profile, judge=judge)


def evaluate_candidate(
    candidate,
    manifest,
    model_lock,
    *,
    protocol,
    final_index=None,
    retrieval_method="frozen",
    top_k=None,
    repetitions=None,
    device=None,
    dtype=None,
    bootstrap_resamples=None,
    bootstrap_seed=None,
    retrieval_protocol=None,
    warmups=None,
    question_scope="all",
    backend_factory=None,
    judge=None,
    profile="smoke",
    qualified_input_envelope=None,
):
    """Protocol-driven Python entrypoint; live retrieved-context composition is deferred."""
    if final_index is not None or retrieval_method != "frozen" or top_k is not None:
        raise ValueError(
            "Retrieved generation integration is deferred: migrate prompt schema, live input gate, timing and calibrated judge explicitly; no legacy semantic fallback"
        )
    if question_scope != "all":
        raise ValueError(
            "Generation evaluates the complete frozen question set; development screening is retired"
        )
    execution = protocol.profile(profile)
    overrides = (repetitions, bootstrap_resamples, bootstrap_seed, warmups)
    resolved = (
        execution.repetitions,
        execution.bootstrap_resamples,
        protocol.meta.seed,
        execution.warmups,
    )
    if any(
        value is not None and value != frozen
        for value, frozen in zip(overrides, resolved, strict=True)
    ):
        raise ValueError(
            "Generation execution overrides must match the resolved protocol"
        )
    if profile != "smoke":
        if judge is None:
            raise ValueError(
                "judge_calibration_pending: inject a calibrated semantic judge"
            )
        load_judge_contract(judge.contract["contract_path"])
    plan = BenchmarkPlan(
        "rag",
        "generation",
        profile,
        manifest.name,
        (candidate,),
        protocol.meta.seed,
        execution.repetitions,
        execution.bootstrap_resamples,
        execution.warmups,
        {
            "device": device or execution.device,
            "dtype": dtype or execution.dtype_for(device or execution.device),
            "qualified_input_envelopes": {candidate: qualified_input_envelope},
        },
    )
    if backend_factory is not None:
        raw = execute_generation(
            candidate,
            manifest,
            model_lock,
            plan,
            protocol,
            backend_factory=backend_factory,
        )
        return score_execution(raw, protocol, profile, judge=judge)
    return run_in_fresh_process(
        candidate, manifest, model_lock, plan, protocol, judge=judge
    )
