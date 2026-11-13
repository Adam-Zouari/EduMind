"""Smoke, qualify, and report one reviewed live-server RAG system."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import execution_devices, parser
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.preflight import (
    current_qualification_fingerprint,
    eligible_candidates,
    model_lock_fingerprints,
    run_preflight,
)
from experiments.benchmarks.common.preflight_reports import resolve_preflight_report
from experiments.benchmarks.common.process import (
    WorkerMeasurementError,
    run_json_worker,
)
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import (
    load_selected_model_lock,
    model_revisions,
)
from experiments.benchmarks.rag.final.benchmark import score
from experiments.benchmarks.rag.final.composition import (
    load_composition,
    reserve_locked,
)
from experiments.benchmarks.rag.final.readiness import require_locked_ci_readiness
from experiments.benchmarks.rag.final.validation import manifest_path, verify_inputs
from experiments.benchmarks.rag.final.worker import LOADERS
from experiments.benchmarks.rag.generation.evidence import (
    EXECUTION_SCHEMA,
    evaluator_identity,
    load_execution,
    save_evaluation,
    seal_execution,
)
from experiments.benchmarks.rag.generation.judge import configured_judge, replay_judge
from experiments.benchmarks.rag.generation.metrics import (
    GENERATION_DIRECTIONS,
    PRIMARY_METRICS,
)
from experiments.benchmarks.rag.generation.validation import questions_from_manifest
from experiments.benchmarks.rag.retrieval_reranking.profiles import parse_candidate
from experiments.benchmarks.vectordb.selection import selected_candidates


def _judge(args, protocol):
    if args.judge_contract is not None and (
        protocol.judge_contract_path is None
        or args.judge_contract.resolve() != protocol.judge_contract_path.resolve()
    ):
        raise ValueError(
            "Freeze judge.contract_path in the generation protocol before execution"
        )
    contract = args.judge_contract or protocol.judge_contract_path
    if contract is None:
        if args.judge_transport or args.judge_records:
            raise ValueError(
                "A judge transport requires the frozen calibrated judge contract"
            )
        return None
    if args.judge_transport:
        return configured_judge(contract, args.judge_transport)
    if args.judge_records:
        return replay_judge(contract, args.judge_records)
    raise ValueError("Provide a configured judge transport or retained judge replies")


def _stress(manifest):
    questions = [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("preflight_stress") is True
    ]
    if not questions:
        raise ValueError("No explicitly reviewed Final preflight stress questions")
    documents = {row["document_id"] for row in questions}
    return replace(
        manifest,
        samples=tuple(
            row
            for row in manifest.samples
            if row in questions
            or row.get("kind") == "document"
            and row["id"] in documents
        ),
    )


def _verify_model_lineage(
    composition, model_lock, generation, retrieval, model_lock_path
):
    models = {
        "chunking_embedding": {
            composition.system["chunking_embedding"].split("|", 1)[1]
        },
        "generator": {generation.model_id(composition.system["generator"])},
        "retrieval": {composition.system["chunking_embedding"].split("|", 1)[1]},
    }
    if retrieval.reranker_model:
        models["retrieval"].add(retrieval.reranker_model)
    current = model_revisions(model_lock)
    for key, names in models.items():
        summary = json.loads(
            Path(composition.upstream[key]["source_summary"]).read_text(
                encoding="utf-8"
            )
        )
        previous = summary.get("provenance", {}).get("model_revisions", {})
        if any(previous.get(name) != current[name] for name in names):
            raise ValueError(
                f"Final {key} model identities differ from reviewed validation evidence"
            )
    database_summary = json.loads(
        Path(composition.upstream["database"]["source_summary"]).read_text(
            encoding="utf-8"
        )
    )
    lineage = (
        database_summary["plan"].get("settings", {}).get("integration_provenance", {})
    )
    if lineage.get("model_lock_sha256") != sha256_file(model_lock_path):
        raise ValueError("Final model lock differs from reviewed database integration")


def _worker(
    manifest, model_lock, plan, protocols, composition, directory, *, preflight=False
):
    try:
        raw = _run_worker(
            manifest,
            model_lock,
            plan,
            protocols,
            composition,
            directory,
            preflight=preflight,
        )
        _validate_returned_execution(
            raw, manifest, model_lock, plan, protocols, composition
        )
        if not preflight:
            seal_execution(raw, directory)
        return raw
    except CandidateExecutionError as exc:
        checkpoint = directory / "generation_progress.json"
        if not checkpoint.is_file():
            raise
        raw = json.loads(checkpoint.read_text(encoding="utf-8"))
        schedule = raw.get("schedule", {})
        if (
            raw.get("composition_checksum") != composition.checksum
            or schedule.get("manifest_identity") != manifest.fingerprint
            or raw.get("schedule_checksum") != stable_hash(schedule)
        ):
            raise ValueError(
                "Interrupted Final worker checkpoint identity mismatch"
            ) from exc
        raw.update(phase="interrupted", error=str(exc), monitoring_complete=False)
        raw["supervisor_resource_evidence"] = getattr(exc, "resource_evidence", {})
        observed = raw.get("resource_evidence", {})
        raw["operational"] = {
            "peak_ram_mib": observed.get("peak_process_tree_ram_mb"),
            "peak_vram_mib": max(
                (
                    v
                    for v in (
                        observed.get("peak_vram_mb"),
                        raw["supervisor_resource_evidence"].get("peak_vram_mb"),
                    )
                    if isinstance(v, (int, float))
                ),
                default=None,
            )
            if plan.settings["device"] == "cuda"
            else None,
        }
        raw["resource_status"] = "partial"
        current = raw.get("in_flight", {})
        for record in raw["records"]:
            for attempt in record["attempts"]:
                if record["question"]["id"] == current.get("question_id") and attempt[
                    "seed"
                ] == current.get("seed"):
                    attempt.update(
                        outcome="infrastructure_error",
                        error=str(exc),
                        failure_stage="infrastructure",
                    )
        atomic_write_json(checkpoint, raw)
        if not (directory / "generation_execution.json").exists():
            seal_execution(raw, directory)
        exc.artifacts["final_evidence"] = raw
        if preflight:
            raise
        return raw


def _validate_returned_execution(
    raw, manifest, model_lock, plan, protocols, composition
):
    """Prove coverage from the frozen request, never from surviving child rows."""
    generation = protocols["generation"]
    candidate = composition.system["generator"]
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    seeds = list(generation.measured_seeds[: plan.repetitions])
    schedule = raw.get("schedule", {})
    expected = {
        "schema": EXECUTION_SCHEMA,
        "candidate": candidate,
        "mode": generation.mode(candidate),
        "model_id": generation.model_id(candidate),
        "model_identity": stable_hash(
            model_lock.get(generation.model_id(candidate), {})
        ),
        "manifest_identity": manifest.fingerprint,
        "protocol_checksum": generation.meta.checksum,
        "profile": plan.profile,
        "device": plan.settings["device"],
        "dtype": plan.settings["dtype"],
        "seeds": seeds,
        "questions": questions,
        "evaluator": evaluator_identity(generation),
        "final_composition": composition.checksum,
        "final_protocols": composition.protocols,
    }
    if any(schedule.get(key) != value for key, value in expected.items()) or raw.get(
        "schedule_checksum"
    ) != stable_hash(schedule):
        raise ValueError("Final worker returned a mismatched frozen execution schedule")
    for key in ("candidate", "mode", "model_id", "device", "dtype"):
        if raw.get(key) != expected[key]:
            raise ValueError(
                "Final worker returned mismatched candidate/model execution"
            )
    rows = raw.get("records")
    if not isinstance(rows, list) or len(rows) != len(questions):
        raise ValueError("Final worker omitted required question outcomes")
    by_id = {row["question"]["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {q["id"] for q in questions}:
        raise ValueError("Final worker duplicated or changed question identities")
    for question in questions:
        row = by_id[question["id"]]
        if (
            row["question"] != question
            or [a.get("seed") for a in row["attempts"]] != seeds
        ):
            raise ValueError(
                "Final worker omitted, duplicated or changed scheduled attempts"
            )


def _run_worker(
    manifest, model_lock, plan, protocols, composition, directory, *, preflight=False
):
    raw = run_json_worker(
        Path(__file__).with_name("worker.py"),
        {
            "candidate": "selected-system",
            "manifest": asdict(manifest),
            "model_lock": model_lock,
            "plan": asdict(plan),
            "composition_path": str(composition.path),
            "composition_checksum": composition.checksum,
            "protocols": {
                key: value.meta.worker_payload() for key, value in protocols.items()
            },
            "protocol_paths": {
                key: str(value.meta.source_path) for key, value in protocols.items()
            },
            "evidence_directory": str(directory),
            "preflight": preflight,
        },
        device=plan.settings["device"],
        prefix="final-rag-",
        error_label="Final RAG worker",
        vram_limit_mb=min(
            protocols[key].authoritative_peak_vram_mb
            for key in ("generation", "retrieval")
        )
        if plan.settings["device"] == "cuda"
        else None,
        require_vram_measurement=plan.settings["device"] == "cuda",
        telemetry_interval_seconds=protocols[
            "final_rag"
        ].preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocols["final_rag"].preflight.poll_interval_seconds,
        timeout_seconds=protocols["final_rag"].preflight.worker_timeout_seconds,
    )
    supervision = raw.get("_worker_supervision", {})
    raw["supervisor_resource_evidence"] = supervision
    if plan.settings["device"] == "cuda":
        observed = [
            v
            for v in (
                raw.get("operational", {}).get("peak_vram_mib"),
                supervision.get("peak_vram_mb"),
            )
            if isinstance(v, (int, float))
        ]
        if observed:
            raw.setdefault("operational", {})["peak_vram_mib"] = max(observed)
    return raw


def main(argv=None):
    cli = parser(__doc__, shortlist=False, profiles=("smoke", "preflight", "locked"))
    cli.add_argument(
        "--composition",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/decisions/final-rag-locked.json",
    )
    cli.add_argument(
        "--model-lock",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/models/selected.json",
    )
    cli.add_argument("--device", choices=("cpu", "cuda", "both"))
    cli.add_argument("--inventory", type=Path, action="append")
    cli.add_argument("--stress-manifest", type=Path)
    cli.add_argument("--preflight-report", type=Path)
    cli.add_argument("--preflight-run-id")
    cli.add_argument("--judge-contract", type=Path)
    cli.add_argument("--judge-transport", type=Path)
    cli.add_argument("--judge-records", type=Path)
    cli.add_argument("--resume-evaluation", type=Path)
    cli.add_argument("--confirm-locked-test", action="store_true")
    for name, loader in LOADERS.items():
        cli.add_argument(
            "--" + name.replace("_", "-") + "-protocol",
            type=Path,
            default=Path(loader.__code__.co_filename).with_name("protocol.yaml"),
        )
    args = cli.parse_args(argv)
    if args.judge_transport and args.judge_records:
        cli.error("Use either a judge transport or retained judge replies")
    if args.resume_evaluation and args.profile == "preflight":
        cli.error("Preflight has no offline semantic evaluation to resume")
    protocols = {
        key: loader(getattr(args, key + "_protocol")) for key, loader in LOADERS.items()
    }
    final, generation, retrieval = (
        protocols[key] for key in ("final_rag", "generation", "retrieval")
    )
    metas = {key: value.meta for key, value in protocols.items()}
    composition = load_composition(args.composition, metas)
    selected_candidates(
        Path(composition.upstream["database"]["path"]),
        "locked",
        protocols["vector_database"],
    )
    protocols["chunking_embedding"].strategy(
        composition.system["chunking_embedding"].split("|", 1)[0]
    )
    retrieval.validate_composition(protocols["chunking_embedding"])
    generation.model_id(composition.system["generator"])
    source = args.manifest or manifest_path(args.profile)
    manifest = load_manifest(source)
    report_path, _ = verify_inputs(
        source, args.profile, final, inventory=args.inventory
    )
    judge = _judge(args, generation) if args.profile != "preflight" else None
    if args.resume_evaluation:
        raw, binding = load_execution(args.resume_evaluation, generation)
        if (
            raw["composition_checksum"] != composition.checksum
            or raw["schedule"]["manifest_identity"] != manifest.fingerprint
        ):
            raise ValueError(
                "Final evaluation recovery inputs do not match saved execution"
            )
        path = args.resume_evaluation.parent / f"final_evaluation-{uuid4().hex}.json"
        if judge is not None:
            judge = judge.with_journal(
                args.resume_evaluation.parent / "judge-journal",
                raw["schedule_checksum"],
            )
        try:
            result = score(raw, manifest, generation, retrieval, judge=judge)
        except CandidateExecutionError as exc:
            save_evaluation(path, error=exc, provenance=binding)
            return 2
        save_evaluation(path, result=result, provenance=binding)
        return 0
    if args.profile == "locked" and judge is None:
        raise ValueError(
            "Final locked execution requires the calibrated semantic judge"
        )
    devices = execution_devices(
        args.profile,
        args.device,
        smoke_devices=final.profile("smoke").devices,
        authoritative_device="cuda",
    )
    selected = parse_candidate(composition.system["retrieval"])
    required = {
        composition.system["chunking_embedding"].split("|", 1)[1],
        generation.model_id(composition.system["generator"]),
    }
    if selected.reranker_model:
        required.add(selected.reranker_model)
    model_lock = load_selected_model_lock(
        args.model_lock, candidates=tuple(sorted(required))
    )
    _verify_model_lineage(
        composition, model_lock, generation, selected, args.model_lock
    )
    qualification = None
    qualification_path = None
    if args.profile != "smoke":
        stress_path = args.stress_manifest or (
            source if args.profile == "preflight" else manifest_path("preflight")
        )
        verify_inputs(stress_path, "preflight", final, inventory=args.inventory)
        stress_manifest = load_manifest(stress_path)
        stress = _stress(stress_manifest)
        fingerprint, context = current_qualification_fingerprint(
            benchmark="final",
            candidates=("selected-system",),
            protocols=metas,
            revisions=model_lock_fingerprints(model_lock),
            execution={
                "device": "cuda",
                "dtype": "float16",
                "batch_size": 1,
                "composition": composition.checksum,
                "residency": composition.system["residency"],
            },
            input_envelope={
                "stress_manifest_checksum": stress_manifest.fingerprint,
                "stress_input_fingerprint": stress.fingerprint,
                "context_tokens": generation.context_tokens,
            },
        )
        if args.profile == "preflight":

            def probe(_):
                settings = {"device": "cuda", "dtype": "float16"}
                plan = BenchmarkPlan(
                    "rag",
                    "final",
                    "locked",
                    stress.name,
                    ("selected-system",),
                    final.meta.seed,
                    1,
                    0,
                    0,
                    settings,
                )
                directory = (
                    PROJECT_ROOT
                    / "artifacts/benchmarks/rag/final/preflight-evidence"
                    / uuid4().hex
                )
                raw = _worker(
                    stress,
                    model_lock,
                    plan,
                    protocols,
                    composition,
                    directory,
                    preflight=True,
                )
                _validate_returned_execution(
                    raw, stress, model_lock, plan, protocols, composition
                )
                attempts = [a for row in raw["records"] for a in row["attempts"]]
                peaks = [
                    raw.get("operational", {}).get(key)
                    for key in ("peak_ram_mib", "peak_vram_mib")
                ]
                if raw.get("resource_status") != "observed" or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v < 0
                    for v in peaks
                ):
                    raise WorkerMeasurementError(
                        "Integrated Final telemetry is unavailable", raw
                    )
                if raw.get("placement", {}).get("status") in {
                    "offload_detected",
                    "wrong_device",
                    "placement_unverifiable",
                }:
                    return {
                        "placement": raw["placement"],
                        "vram_measurement_method": raw["vram_measurement_method"],
                        "resource_samples": raw["resource_samples"],
                    }
                if raw.get("error") or any(
                    a.get("request_status") != "completed" for a in attempts
                ):
                    error = CandidateExecutionError(
                        "Integrated Final stress request failed",
                        artifacts={"final_evidence": raw},
                    )
                    error.resource_evidence = {
                        "error": raw.get("error"),
                        "attempts": attempts,
                    }
                    raise error
                return {
                    "placement": {"status": "qualified"},
                    "vram_measurement_method": raw["vram_measurement_method"],
                    "peak_vram_mb": raw["operational"]["peak_vram_mib"],
                    "resource_samples": raw["resource_samples"],
                    "tested_input_envelope": {
                        "maximum_prompt_tokens": max(
                            row["prompt_tokens"]
                            for row in raw["input_compatibility"]["questions"]
                        ),
                        "context_tokens": generation.context_tokens,
                        "maximum_query_tokens": max(
                            (
                                a.get("retrieval", {})
                                .get("query_input", {})
                                .get("native_token_count", 0)
                                for a in attempts
                            ),
                            default=0,
                        ),
                        "maximum_reranker_tokens": max(
                            (
                                count
                                for a in attempts
                                for count in a.get("retrieval", {}).get(
                                    "reranker_input_token_counts", []
                                )
                            ),
                            default=0,
                        ),
                    },
                    "prediction_directory": str(directory),
                    "placement_before": [
                        a.get("retrieval_placement_before") for a in attempts
                    ],
                    "placement_after": [
                        a.get("retrieval_placement_after") for a in attempts
                    ],
                    "generator_placement_before": [
                        a.get("generator_placement_before") for a in attempts
                    ],
                    "generator_placement_after": [
                        a.get("generator_placement_after") for a in attempts
                    ],
                }

            result = run_preflight(
                benchmark="final",
                candidates=("selected-system",),
                fingerprint=fingerprint,
                context=context,
                probe=probe,
                decision_files={"composition": composition.path},
                no_mlflow=args.no_mlflow,
            )
            return 0 if result.ready_for_development else 2
        qualification_path, qualification = resolve_preflight_report(
            benchmark="final",
            fingerprint=fingerprint,
            candidates=("selected-system",),
            explicit=args.preflight_report,
            run_id=args.preflight_run_id,
        )
        eligible_candidates(
            ("selected-system",), qualification, profile="locked", label="Final RAG"
        )
    if args.profile == "locked":
        if not args.confirm_locked_test:
            cli.error("Locked is reporting-only; explicitly pass --confirm-locked-test")
        require_locked_ci_readiness(manifest, generation, retrieval)
        reserve_locked(
            PROJECT_ROOT
            / "artifacts/benchmarks/rag/final/locked-consumption"
            / (manifest.fingerprint + ".json"),
            {
                "manifest": manifest.fingerprint,
                "composition": composition.checksum,
                "protocol": final.meta.checksum,
            },
        )
    results = []
    for device in devices:
        execution = final.profile(args.profile)
        settings = {
            "device": device,
            "dtype": execution.dtype_for(device),
            "composition_checksum": composition.checksum,
            "composition": composition.system,
        }
        if qualification:
            settings.update(
                preflight_run_id=qualification["mlflow_run_id"],
                preflight_fingerprint=qualification["qualification_fingerprint"],
                qualified_input_envelope=qualification["candidates"][0][
                    "tested_input_envelope"
                ],
            )
        plan = BenchmarkPlan(
            "rag",
            "final",
            args.profile,
            manifest.name,
            ("selected-system",),
            final.meta.seed,
            execution.repetitions,
            execution.bootstrap_resamples,
            execution.warmups,
            settings,
        )

        def evaluate(_, run_context, plan=plan):
            directory = Path(run_context["artifact_directory"]) / "retained-execution"
            raw = _worker(manifest, model_lock, plan, protocols, composition, directory)
            calibrated = (
                judge.with_journal(
                    directory / "judge-journal", raw["schedule_checksum"]
                )
                if judge
                else None
            )
            result = score(raw, manifest, generation, retrieval, judge=calibrated)
            return result

        directions = {
            **GENERATION_DIRECTIONS,
            "reliability.request_failure_rate": "min",
        }
        results.append(
            run_benchmark(
                plan,
                evaluate,
                dataset_checksum=manifest.fingerprint,
                directions=directions,
                primary_metric=PRIMARY_METRICS,
                required_metrics=tuple(directions),
                nullable_metrics=tuple(directions),
                revisions=model_revisions(model_lock),
                decision_files={"composition": composition.path},
                input_artifacts={
                    "manifest": source,
                    "data_validation_report": report_path,
                    "model_lock": args.model_lock,
                    **(
                        {"qualification": qualification_path}
                        if qualification_path
                        else {}
                    ),
                },
                protocols=metas,
                no_mlflow=args.no_mlflow,
                monitor_resources=False,
                paired_comparisons=False,
                csv_artifact_tables=("final_attempts",),
                evaluator_receives_context=True,
                run_name_prefix=f"smoke-{device}"
                if args.profile == "smoke"
                else "locked",
            )
        )
    print(
        json.dumps(
            {
                "runs": [
                    {"run_id": r.run_id, "artifacts": str(r.artifact_directory)}
                    for r in results
                ]
            },
            indent=2,
        )
    )
    return 0 if all(r.complete for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
