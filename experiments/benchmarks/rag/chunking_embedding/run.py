"""CLI for chunking/embedding pair qualification and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import (
    LIFECYCLE_PROFILES,
    default_decision_path,
    execution_devices,
    parser,
)
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.preflight import (
    current_qualification_fingerprint,
    eligible_candidates,
    model_lock_fingerprints,
    rag_stress_manifest,
    resolve_preflight_report,
    run_preflight,
)
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import (
    load_selected_model_lock,
    model_revisions,
)
from experiments.benchmarks.rag.chunking_embedding.benchmark import (
    directions_for,
    preflight_in_fresh_process,
    run_in_fresh_process,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser(
        "Benchmark chunking and embedding pairs", profiles=LIFECYCLE_PROFILES
    )
    argument_parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    argument_parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    argument_parser.add_argument("--preflight-report", type=Path)
    argument_parser.add_argument("--preflight-run-id")
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    arguments = argument_parser.parse_args(argv)
    protocol = load_protocol(arguments.protocol)
    execution_name = (
        "development" if arguments.profile == "preflight" else arguments.profile
    )
    execution = protocol.profile(execution_name)
    try:
        devices = execution_devices(
            arguments.profile,
            arguments.device,
            smoke_devices=protocol.profile("smoke").devices,
            authoritative_device=execution.device,
        )
    except ValueError as exc:
        argument_parser.error(str(exc))
    if arguments.profile != "smoke" and arguments.dtype not in {None, execution.dtype}:
        argument_parser.error(
            f"{arguments.profile} requires protocol dtype {execution.dtype}"
        )

    manifest_path = (arguments.manifest or _manifest_path(arguments.profile)).resolve()
    manifest = load_manifest(manifest_path)
    manifest_profile = (
        "development" if arguments.profile == "preflight" else arguments.profile
    )
    require_manifest_split(manifest, manifest_profile, _split(manifest_profile))
    declared = protocol.development_candidates
    shortlist = arguments.shortlist or default_decision_path(
        "chunking-embedding", arguments.profile
    )
    candidates = _selected_candidates(arguments.profile, shortlist, protocol)
    lock_candidates = declared if arguments.profile != "smoke" else candidates
    embedding_names = tuple(
        sorted({candidate.split("|", 1)[1] for candidate in lock_candidates})
    )
    model_lock_path = PROJECT_ROOT / "data/benchmarks/models/selected.json"
    model_lock = load_selected_model_lock(model_lock_path, candidates=embedding_names)
    fingerprint = ""
    qualification_context = {}
    if arguments.profile != "smoke":
        fingerprint, qualification_context = _qualification_identity(
            protocol, model_lock
        )

    if arguments.profile == "preflight":
        stress = rag_stress_manifest(manifest)
        plan = _plan(protocol, stress, declared, "development", "cuda", execution.dtype)
        plan = BenchmarkPlan(
            **{
                **vars(plan),
                "warmups": 0,
                "repetitions": 1,
                "bootstrap_resamples": 0,
            }
        )
        result = run_preflight(
            benchmark="chunking-embedding",
            candidates=declared,
            fingerprint=fingerprint,
            context={
                **qualification_context,
                "stress_manifest": str(manifest_path),
                "stress_manifest_checksum": manifest.fingerprint,
                "stress_manifest_fingerprint": stress.fingerprint,
            },
            probe=lambda candidate: preflight_in_fresh_process(
                candidate,
                stress,
                model_lock,
                plan,
                vram_limit_mb=protocol.authoritative_peak_vram_mb,
            ),
            no_mlflow=arguments.no_mlflow,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "ready_for_development": result.ready_for_development,
                    "qualified_candidates": result.qualified_candidates,
                    "excluded_candidates": result.excluded_candidates,
                    "blocked_candidates": result.blocked_candidates,
                    "artifacts": str(result.artifact_directory),
                },
                indent=2,
            )
        )
        return 0 if result.ready_for_development else 2

    qualification_path = None
    qualification = None
    if arguments.profile != "smoke":
        qualification_path, qualification = resolve_preflight_report(
            benchmark="chunking-embedding",
            fingerprint=fingerprint,
            candidates=declared,
            explicit=arguments.preflight_report,
            run_id=arguments.preflight_run_id,
        )
        candidates = eligible_candidates(
            candidates,
            qualification,
            profile=arguments.profile,
            label="Chunking/embedding",
        )

    results = []
    for device in devices:
        dtype = arguments.dtype or execution.dtype_for(device)
        results.append(
            _run_evaluation(
                arguments.profile,
                candidates,
                manifest,
                manifest_path,
                model_lock,
                model_lock_path,
                protocol,
                device=device,
                dtype=dtype,
                shortlist=shortlist,
                qualification_path=qualification_path,
                qualification=qualification,
                no_mlflow=arguments.no_mlflow,
            )
        )
    print(
        json.dumps(
            [
                {
                    "device": device,
                    "run_id": result.run_id,
                    "complete": result.complete,
                    "artifacts": str(result.artifact_directory),
                }
                for device, result in zip(devices, results, strict=True)
            ],
            indent=2,
        )
    )
    return 0 if all(result.complete for result in results) else 2


def _run_evaluation(
    profile,
    candidates,
    manifest,
    manifest_path,
    model_lock,
    model_lock_path,
    protocol,
    *,
    device,
    dtype,
    shortlist,
    qualification_path,
    qualification,
    no_mlflow,
):
    plan = _plan(protocol, manifest, candidates, profile, device, dtype)
    if qualification is not None:
        plan = BenchmarkPlan(
            **{
                **vars(plan),
                "settings": {
                    **plan.settings,
                    "preflight_run_id": qualification.get("mlflow_run_id"),
                    "preflight_fingerprint": qualification.get(
                        "qualification_fingerprint"
                    ),
                    "hardware_exclusions": qualification.get("excluded_candidates", []),
                },
            }
        )
    directions, required_metrics = directions_for(manifest, protocol)
    return run_benchmark(
        plan,
        lambda candidate: run_in_fresh_process(
            candidate, manifest, model_lock, plan, device=device, dtype=dtype
        ),
        dataset_checksum=manifest.checksum,
        directions=directions,
        primary_metric=tuple(
            f"quality.overall.{metric}" for metric in protocol.primary_quality_metrics
        ),
        required_metrics=required_metrics,
        paired_metrics=tuple(
            metric
            for metric in (
                *(
                    f"quality.overall.{name}"
                    for name in protocol.primary_quality_metrics
                ),
                *(f"quality.overall.{name}" for name in protocol.alpha_ndcg_metrics),
            )
            if metric in directions
        ),
        revisions=model_revisions(model_lock),
        decision_files={"shortlist": shortlist} if shortlist else None,
        input_artifacts={
            "manifest": manifest_path,
            "model_lock": model_lock_path,
            **(
                {"preflight_report": qualification_path}
                if qualification_path is not None
                else {}
            ),
        },
        protocols={"chunking_embedding": protocol.meta},
        no_mlflow=no_mlflow,
        candidate_artifact_name="candidate.json",
        sample_artifact_name="query_metrics",
        resource_artifact_name="resources",
        resource_monitor_options={
            "require_vram": device == "cuda",
            "report_zero_vram": device == "cpu",
        },
        monitor_temporary_disk=False,
        paired_group_key="document_id",
        run_name_prefix=(
            f"rag-chunking-embedding-smoke-{device}"
            if profile == "smoke"
            else f"rag-chunking-embedding-{profile}"
        ),
        operational_maximums=(
            {"peak_vram_mb": protocol.authoritative_peak_vram_mb}
            if profile != "smoke"
            else None
        ),
    )


def _plan(protocol, manifest, candidates, profile, device, dtype):
    execution = protocol.profile(profile)
    return BenchmarkPlan(
        "rag",
        "chunking-embedding",
        profile,
        manifest.name,
        tuple(candidates),
        seed=protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
        settings={
            "device": device,
            "dtype": dtype,
            "chunking_embedding_protocol": protocol.meta.worker_payload(),
        },
    )


def _selected_candidates(profile, shortlist, protocol):
    declared = protocol.development_candidates
    if profile == "smoke":
        if shortlist is not None:
            raise ValueError("Smoke chunking/embedding does not accept a shortlist")
        return (protocol.smoke_pair,)
    if profile in {"preflight", "development"}:
        if shortlist is not None:
            raise ValueError(f"{profile} runs the declared chunking/embedding matrix")
        return declared
    decision = load_engineer_decision(
        shortlist,
        exact=1 if profile == "locked" else None,
        maximum=1 if profile == "locked" else protocol.maximum_finalists,
        expected_source=(
            "rag",
            "chunking-embedding",
            "validation" if profile == "locked" else "development",
        ),
    )
    unknown = sorted(set(decision.selected_candidates) - set(declared))
    if unknown:
        raise ValueError(
            "Chunking/embedding decision contains undeclared pairs: "
            + ", ".join(unknown)
        )
    return decision.selected_candidates


def _qualification_identity(protocol, model_lock):
    execution = protocol.profile("development")
    return current_qualification_fingerprint(
        benchmark="chunking-embedding",
        candidates=protocol.development_candidates,
        protocols={"chunking_embedding": protocol.meta},
        revisions=model_lock_fingerprints(model_lock),
        execution={
            "device": execution.device,
            "dtype": execution.dtype,
            "batch_size": execution.batch_size,
            "vram_limit_mb": protocol.authoritative_peak_vram_mb,
        },
        input_envelope={
            "strategies": {
                name: dict(settings) for name, settings in protocol.strategies.items()
            },
            "reject_truncation": protocol.reject_truncation,
            "audit_depth": protocol.audit_depth,
        },
    )


def _manifest_path(profile: str) -> Path:
    name = {
        "smoke": "smoke.json",
        "preflight": "rag-selection-dev.json",
        "development": "rag-selection-dev.json",
        "validation": "rag-selection-validation.json",
        "locked": "rag-selection-locked-test.json",
    }[profile]
    return PROJECT_ROOT / "data/benchmarks/rag" / name


def _split(profile: str):
    return {
        "smoke": {"smoke"},
        "development": {"dev", "development"},
        "validation": {"validation"},
        "locked": {"locked-test"},
    }[profile]


if __name__ == "__main__":
    raise SystemExit(main())
