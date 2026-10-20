"""Generation lifecycle CLI. All dependencies are prepared local inputs."""

import json
from dataclasses import asdict
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import (
    LIFECYCLE_PROFILES,
    default_decision_path,
    execution_devices,
    parser,
)
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.data_validation import (
    manifest_path as default_manifest_path,
)
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.preflight import (
    current_qualification_fingerprint,
    eligible_candidates,
    model_lock_fingerprints,
    run_preflight,
    stress_input_identity,
)
from experiments.benchmarks.common.preflight_reports import resolve_preflight_report
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import (
    load_selected_model_lock,
    model_revisions,
)

from .evaluate import run_in_fresh_process
from .judge import load_judge_contract, replay_judge
from .metrics import BASE_KEYS, GENERATION_DIRECTIONS, OPERATIONAL_KEYS, PRIMARY_METRICS
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol
from .validation import questions_from_manifest, verify_inputs


def required_ci_policy_keys(questions):
    """Reference-only support-policy inventory, frozen before candidate outputs."""
    result = []
    slices = {
        "": questions,
        **{
            f"slice.{name}.": [q for q in questions if q["answerable"] == answerable]
            for name, answerable in (("answerable", True), ("unanswerable", False))
        },
        **{
            f"slice.{name}.": [q for q in questions if q["evidence_type"] == name]
            for name in ("text", "table", "formula", "mixed")
        },
    }
    for prefix, subset in slices.items():
        if not subset:
            continue
        answerable = any(q["answerable"] for q in subset)
        unanswerable = any(not q["answerable"] for q in subset)
        for key in (*BASE_KEYS, *OPERATIONAL_KEYS):
            if (
                key.startswith("quality.")
                or key
                in {
                    "validity.response_validity_rate",
                    "repeatability.citation_agreement",
                }
            ) and not answerable:
                continue
            if key == "validity.refusal_validity_rate" and not unanswerable:
                continue
            result.append(prefix + key)
    return tuple(result)


def _selected_candidates(
    profile,
    shortlist,
    protocol,
    *,
    qualification_fingerprint="",
    judge_contract_sha256=None,
):
    declared = tuple(protocol.models)
    if profile in {"smoke", "preflight", "development"}:
        if shortlist is not None:
            raise ValueError(
                f"{profile} must not replace its declared generation roster with a shortlist"
            )
        return declared
    decision = load_engineer_decision(
        shortlist,
        exact=1 if profile == "locked" else None,
        maximum=1 if profile == "locked" else protocol.maximum_finalists,
        expected_source=(
            "rag",
            "generation",
            "validation" if profile == "locked" else "development",
        ),
        expected_protocols={"generation": protocol.meta.checksum},
        expected_qualification=qualification_fingerprint,
    )
    if set(decision.selected_candidates) - set(declared):
        raise ValueError(
            "Generation decision contains undeclared model-mode candidates"
        )
    source = json.loads(decision.source_summary.read_text(encoding="utf-8"))
    recorded_judge = (
        source.get("provenance", {})
        .get("input_artifacts", {})
        .get("judge_contract", {})
        .get("sha256")
    )
    if judge_contract_sha256 is None or recorded_judge != judge_contract_sha256:
        raise ValueError(
            "Generation decision was scored under a different or unverified judge contract"
        )
    return decision.selected_candidates


def _qualification_identity(protocol, model_lock, manifest):
    stress = [
        row
        for row in manifest.samples
        if row.get("kind") == "question" and row.get("preflight_stress") is True
    ]
    if not stress:
        raise ValueError(
            "missing_stress_input: freeze reviewed development question(s) with preflight_stress=true"
        )
    execution = protocol.profile("development")
    return current_qualification_fingerprint(
        benchmark="generation",
        candidates=tuple(protocol.models),
        protocols={"generation": protocol.meta},
        revisions=model_lock_fingerprints(model_lock),
        execution={
            "device": execution.device,
            "dtype": execution.dtype,
            "batch_size": execution.batch_size,
            "vram_limit_mb": protocol.authoritative_peak_vram_mb,
            "preflight": asdict(protocol.preflight),
        },
        input_envelope={
            **stress_input_identity(manifest, stress),
            "context_tokens": protocol.context_tokens,
            "stress_selection": "maximum-native-rendered-prompt-among-reviewed-stress-questions",
            "output_capacity": "context-minus-complete-native-prompt",
            "truncate": False,
        },
    )


def _plan(
    protocol,
    manifest,
    candidates,
    profile,
    device,
    *,
    preflight=False,
    qualification=None,
):
    execution = protocol.preflight if preflight else protocol.profile(profile)
    return BenchmarkPlan(
        "rag",
        "generation",
        profile,
        manifest.name,
        tuple(candidates),
        protocol.meta.seed,
        execution.repetitions,
        execution.bootstrap_resamples,
        execution.warmups,
        {
            "device": device,
            "dtype": protocol.profile(profile).dtype_for(device),
            "measured_seeds": list(protocol.measured_seeds[: execution.repetitions]),
            "preflight_fingerprint": (qualification or {}).get(
                "qualification_fingerprint", ""
            ),
            "preflight_run_id": (qualification or {}).get("mlflow_run_id", ""),
            "hardware_exclusions": (qualification or {}).get("excluded_candidates", []),
            "qualified_input_envelopes": {
                row["candidate"]: row.get("tested_input_envelope")
                for row in (qualification or {}).get("candidates", [])
                if row.get("status") == "qualified"
            },
        },
    )


def main(argv=None):
    argument_parser = parser(
        "Benchmark local generation on frozen evidence", profiles=LIFECYCLE_PROFILES
    )
    argument_parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    argument_parser.add_argument("--dtype", choices=("float32", "float16"))
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    argument_parser.add_argument("--inventory", type=Path, action="append")
    argument_parser.add_argument("--preflight-report", type=Path)
    argument_parser.add_argument("--preflight-run-id")
    argument_parser.add_argument(
        "--judge-records",
        type=Path,
        help="Replay existing replies bound to the configured calibrated judge; never calls a live API",
    )
    arguments = argument_parser.parse_args(argv)
    protocol = load_protocol(arguments.protocol)
    profile = "development" if arguments.profile == "preflight" else arguments.profile
    execution = protocol.profile(profile)
    try:
        devices = execution_devices(
            arguments.profile,
            arguments.device,
            smoke_devices=protocol.profile("smoke").devices,
            authoritative_device=execution.device,
        )
        if arguments.dtype is not None and any(
            arguments.dtype != execution.dtype_for(d) for d in devices
        ):
            raise ValueError("Requested dtype must match each device's frozen profile")
    except ValueError as exc:
        argument_parser.error(str(exc))
    manifest_path = (
        arguments.manifest or default_manifest_path("generation", profile)
    ).resolve()
    manifest = load_manifest(manifest_path)
    require_manifest_split(
        manifest,
        profile,
        {"development": {"dev", "development"}, "locked": {"locked-test"}}.get(
            profile, {profile}
        ),
    )
    data_report_path, _ = verify_inputs(
        manifest_path, profile, protocol, inventory=arguments.inventory
    )
    questions = questions_from_manifest(manifest)
    judge = None
    if arguments.profile not in {"smoke", "preflight"}:
        load_judge_contract(protocol.judge_contract_path)
        missing = (
            set(required_ci_policy_keys(questions)) - protocol.minimum_ci_sources.keys()
        )
        if missing:
            raise ValueError(
                "ci_support_policy_pending: freeze per-metric/slice source support before comparison: "
                + ", ".join(sorted(missing))
            )
        if arguments.judge_records is None:
            raise ValueError(
                "judge_transport_unconfigured: provide local --judge-records or inject a calibrated judge through the Python API"
            )
    if arguments.judge_records is not None:
        if arguments.profile == "preflight":
            argument_parser.error(
                "Preflight is hardware-only and must not judge quality"
            )
        judge = replay_judge(protocol.judge_contract_path, arguments.judge_records)
    model_lock_path = PROJECT_ROOT / "data/benchmarks/models/selected.json"
    model_lock = load_selected_model_lock(
        model_lock_path, candidates=tuple(dict.fromkeys(protocol.models.values()))
    )
    fingerprint, context, qualification_path, qualification = "", {}, None, None
    if arguments.profile != "smoke":
        development_path = (
            manifest_path
            if profile == "development"
            else default_manifest_path("generation", "development")
        )
        if profile != "development":
            verify_inputs(
                development_path, "development", protocol, inventory=arguments.inventory
            )
        development = (
            manifest if profile == "development" else load_manifest(development_path)
        )
        fingerprint, context = _qualification_identity(
            protocol, model_lock, development
        )
    shortlist = arguments.shortlist or default_decision_path(
        "generation", arguments.profile
    )
    candidates = _selected_candidates(
        arguments.profile,
        shortlist,
        protocol,
        qualification_fingerprint=fingerprint,
        judge_contract_sha256=judge.contract["contract_sha256"]
        if judge is not None
        else None,
    )
    if arguments.profile == "preflight":
        if arguments.preflight_report or arguments.preflight_run_id:
            argument_parser.error(
                "Preflight generates a report; it cannot consume a previous qualification"
            )
        plan = _plan(
            protocol, manifest, candidates, "development", "cuda", preflight=True
        )
        result = run_preflight(
            benchmark="generation",
            candidates=candidates,
            fingerprint=fingerprint,
            context={**context, "data_validation_report": str(data_report_path)},
            probe=lambda candidate: run_in_fresh_process(
                candidate, manifest, model_lock, plan, protocol, preflight=True
            ),
            no_mlflow=arguments.no_mlflow,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "ready_for_development": result.ready_for_development,
                    "qualified_candidates": result.qualified_candidates,
                    "artifacts": str(result.artifact_directory),
                },
                indent=2,
            )
        )
        return 0 if result.ready_for_development else 2
    if arguments.profile != "smoke":
        qualification_path, qualification = resolve_preflight_report(
            benchmark="generation",
            fingerprint=fingerprint,
            candidates=tuple(protocol.models),
            explicit=arguments.preflight_report,
            run_id=arguments.preflight_run_id,
        )
        candidates = eligible_candidates(
            candidates, qualification, profile=profile, label="Generation"
        )
    if profile == "locked":
        marker = (
            PROJECT_ROOT
            / "artifacts/benchmarks/rag/generation/locked-consumption"
            / (manifest.fingerprint + ".json")
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "manifest": manifest.fingerprint,
                    "protocol": protocol.meta.checksum,
                    "decision": str(shortlist),
                },
                stream,
            )
    results = []
    for device in devices:
        plan = _plan(
            protocol, manifest, candidates, profile, device, qualification=qualification
        )
        inputs = {
            "manifest": manifest_path,
            "data_validation_report": data_report_path,
            "model_lock": model_lock_path,
        }
        if qualification_path:
            inputs["preflight_report"] = qualification_path
        if judge is not None:
            inputs.update(
                judge_contract=protocol.judge_contract_path,
                judge_calibration=Path(judge.contract["calibration_path"]),
                judge_records=arguments.judge_records,
            )
        results.append(
            run_benchmark(
                plan,
                lambda candidate, child, plan=plan: run_in_fresh_process(
                    candidate,
                    manifest,
                    model_lock,
                    plan,
                    protocol,
                    judge=judge,
                    evidence_directory=Path(child["artifact_directory"]),
                ),
                dataset_checksum=manifest.fingerprint,
                directions=GENERATION_DIRECTIONS,
                primary_metric=PRIMARY_METRICS,
                required_metrics=tuple(GENERATION_DIRECTIONS),
                nullable_metrics=tuple(GENERATION_DIRECTIONS),
                revisions=model_revisions(model_lock),
                decision_files={"shortlist": shortlist} if shortlist else None,
                input_artifacts=inputs,
                protocols={"generation": protocol.meta},
                no_mlflow=arguments.no_mlflow,
                monitor_resources=False,
                paired_comparisons=False,
                evaluator_receives_context=True,
                candidate_artifact_name="candidate.json",
                run_name_prefix=f"smoke-{device}" if profile == "smoke" else profile,
                operational_maximums={
                    "peak_vram_mib": protocol.authoritative_peak_vram_mb
                }
                if device == "cuda"
                else None,
            )
        )
    print(
        json.dumps(
            [
                {
                    "device": d,
                    "run_id": r.run_id,
                    "complete": r.complete,
                    "scope": "wiring_only"
                    if profile == "smoke"
                    else "authoritative_contract",
                    "artifacts": str(r.artifact_directory),
                }
                for d, r in zip(devices, results, strict=True)
            ],
            indent=2,
        )
    )
    return 0 if all(r.complete for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
