"""CLI for complete retrieval/reranking qualification and evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import (
    LIFECYCLE_PROFILES,
    default_decision_path,
    execution_devices,
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
from experiments.benchmarks.rag.chunking_embedding.profiles import split_candidate
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_CHUNKING_PROTOCOL_PATH,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as load_chunking_protocol,
)
from experiments.benchmarks.rag.retrieval_reranking.benchmark import (
    preflight_in_fresh_process,
    run_in_fresh_process,
)
from experiments.benchmarks.rag.retrieval_reranking.comparisons import (
    parent_artifact_builder,
)
from experiments.benchmarks.rag.retrieval_reranking.metrics import (
    directions_for,
    primary_metrics,
)
from experiments.benchmarks.rag.retrieval_reranking.profiles import (
    development_candidates,
    owner_first,
    parse_candidate,
    required_reranker_models,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark complete first-stage retrieval and reranking stacks"
    )
    parser.add_argument("--profile", choices=LIFECYCLE_PROFILES, default="smoke")
    parser.add_argument(
        "--chunking-protocol", type=Path, default=DEFAULT_CHUNKING_PROTOCOL_PATH
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--embedding-selection", type=Path)
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--preflight-run-id")
    parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--compare-finalists", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args(argv)

    protocol = load_protocol(arguments.protocol)
    chunking_protocol = load_chunking_protocol(arguments.chunking_protocol)
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
        parser.error(str(exc))
    if arguments.profile != "smoke" and arguments.dtype not in {None, execution.dtype}:
        parser.error(f"{arguments.profile} requires protocol dtype {execution.dtype}")
    if arguments.compare_finalists and arguments.profile != "validation":
        parser.error("--compare-finalists is valid only for validation")

    manifest_path = (arguments.manifest or _manifest_path(arguments.profile)).resolve()
    manifest = load_manifest(manifest_path)
    manifest_profile = (
        "development" if arguments.profile == "preflight" else arguments.profile
    )
    require_manifest_split(manifest, manifest_profile, _split(manifest_profile))
    declared = owner_first(development_candidates())
    for candidate in declared:
        parse_candidate(candidate)

    embedding_selection = arguments.embedding_selection
    if arguments.profile != "smoke" and embedding_selection is None:
        embedding_selection = default_decision_path("chunking-embedding", "locked")
    shortlist = arguments.shortlist or default_decision_path(
        "retrieval-reranking", arguments.profile
    )
    selected_pair, embedding_decision = _selected_embedding(
        arguments.profile, embedding_selection, chunking_protocol
    )
    candidates = _selected_candidates(
        arguments.profile, shortlist, declared, protocol.maximum_finalists
    )
    _, embedding_name = split_candidate(selected_pair)
    lock_models = (
        (embedding_name, *required_reranker_models(declared))
        if arguments.profile != "smoke"
        else (embedding_name, *required_reranker_models(candidates))
    )
    model_lock_path = PROJECT_ROOT / "data/benchmarks/models/selected.json"
    model_lock = load_selected_model_lock(model_lock_path, candidates=lock_models)
    fingerprint = ""
    qualification_context = {}
    if arguments.profile != "smoke":
        fingerprint, qualification_context = _qualification_identity(
            protocol,
            chunking_protocol,
            declared,
            selected_pair,
            model_lock,
        )

    if arguments.profile == "preflight":
        stress = rag_stress_manifest(manifest)
        plan = _plan(
            protocol,
            chunking_protocol,
            stress,
            declared,
            selected_pair,
            embedding_decision,
            "development",
            "cuda",
            execution.dtype,
            warmups=0,
            repetitions=1,
            bootstrap_resamples=0,
        )
        result = run_preflight(
            benchmark="retrieval-reranking",
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
            decision_files={"chunking_embedding": embedding_selection},
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
            benchmark="retrieval-reranking",
            fingerprint=fingerprint,
            candidates=declared,
            explicit=arguments.preflight_report,
            run_id=arguments.preflight_run_id,
        )
        candidates = eligible_candidates(
            candidates,
            qualification,
            profile=arguments.profile,
            label="Retrieval/reranking",
        )

    results = []
    for device in devices:
        dtype = arguments.dtype or execution.dtype_for(device)
        results.append(
            _run_evaluation(
                protocol,
                chunking_protocol,
                manifest,
                manifest_path,
                candidates,
                selected_pair,
                embedding_decision,
                model_lock,
                model_lock_path,
                arguments.profile,
                device,
                dtype,
                shortlist=shortlist,
                embedding_selection=embedding_selection,
                qualification_path=qualification_path,
                qualification=qualification,
                compare_finalists=arguments.compare_finalists,
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
    protocol,
    chunking_protocol,
    manifest,
    manifest_path,
    candidates,
    selected_pair,
    embedding_decision,
    model_lock,
    model_lock_path,
    profile,
    device,
    dtype,
    *,
    shortlist,
    embedding_selection,
    qualification_path,
    qualification,
    compare_finalists,
    no_mlflow,
):
    plan = _plan(
        protocol,
        chunking_protocol,
        manifest,
        candidates,
        selected_pair,
        embedding_decision,
        profile,
        device,
        dtype,
    )
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
    pools: dict[str, dict[str, object]] = {}

    def store_pool(owner: str, evaluated, owner_run_id: str) -> None:
        artifacts = dict(evaluated[5])
        rows = artifacts.get("candidate_pool")
        index_build = artifacts.get("index_build")
        if not isinstance(rows, list) or not isinstance(index_build, Mapping):
            raise RuntimeError(f"{owner} did not return its owned pool")
        checksum = stable_hash(rows)
        if dict(evaluated[3]).get("pool_checksum") != checksum:
            raise RuntimeError(f"{owner} returned an inconsistent pool hash")
        pools[owner] = {
            "owner_candidate": owner,
            "owner_run_id": owner_run_id,
            "pool_checksum": checksum,
            "index_checksum": index_build.get("index_sha256"),
            "rows": rows,
        }

    def evaluate(candidate_name: str, context: Mapping[str, object]):
        parsed = parse_candidate(candidate_name)
        if parsed.reranker != "none" and parsed.owner_identifier not in pools:
            owner_result = run_in_fresh_process(
                parsed.owner_identifier,
                manifest,
                model_lock,
                plan,
                device=device,
                dtype=dtype,
                frozen_pool=None,
                child_run_id=f"{context['mlflow_run_id']}:pool-input",
            )
            store_pool(
                parsed.owner_identifier,
                owner_result,
                f"{context['mlflow_run_id']}:pool-input",
            )
        evaluated = run_in_fresh_process(
            candidate_name,
            manifest,
            model_lock,
            plan,
            device=device,
            dtype=dtype,
            frozen_pool=pools.get(parsed.owner_identifier),
            child_run_id=str(context["mlflow_run_id"]),
        )
        if parsed.reranker == "none":
            store_pool(candidate_name, evaluated, str(context["mlflow_run_id"]))
        return evaluated

    decision_files = {
        name: path
        for name, path in {
            "chunking_embedding": embedding_selection,
            "retrieval_reranking": shortlist,
        }.items()
        if path is not None
    }
    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.checksum,
        directions=directions,
        primary_metric=tuple(
            metric for metric in primary_metrics(protocol) if metric in directions
        ),
        required_metrics=required_metrics,
        paired_metrics=(),
        revisions=model_revisions(model_lock),
        decision_files=decision_files or None,
        input_artifacts={
            "manifest": manifest_path,
            "model_lock": model_lock_path,
            **(
                {"preflight_report": qualification_path}
                if qualification_path is not None
                else {}
            ),
        },
        protocols={
            "retrieval": protocol.meta,
            "chunking_embedding": chunking_protocol.meta,
        },
        no_mlflow=no_mlflow,
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
        sample_artifact_name="query_metrics",
        resource_artifact_name="resources",
        resource_monitor_options=lambda candidate: {
            "require_vram": device == "cuda"
            and parse_candidate(candidate).model_backed,
            "report_zero_vram": device == "cpu"
            or not parse_candidate(candidate).model_backed,
            "zero_vram_measurement_method": (
                "cpu-zero" if device == "cpu" else "not-applicable-zero"
            ),
        },
        monitor_temporary_disk=False,
        run_name_prefix=(
            f"rag-retrieval-reranking-smoke-{device}"
            if profile == "smoke"
            else f"rag-retrieval-reranking-{profile}"
        ),
        shuffle_candidates=False,
        evaluator_receives_context=True,
        parent_artifact_builder=parent_artifact_builder(
            pools,
            directions,
            compare_finalists=compare_finalists,
        ),
        operational_maximums=(
            {"peak_vram_mb": protocol.authoritative_peak_vram_mb}
            if profile != "smoke"
            else None
        ),
    )


def _plan(
    protocol,
    chunking_protocol,
    manifest,
    candidates,
    selected_pair,
    embedding_decision,
    profile,
    device,
    dtype,
    *,
    warmups=None,
    repetitions=None,
    bootstrap_resamples=None,
):
    execution = protocol.profile(profile)
    return BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        profile,
        manifest.name,
        tuple(candidates),
        seed=protocol.meta.seed,
        repetitions=execution.repetitions if repetitions is None else repetitions,
        bootstrap_resamples=(
            execution.bootstrap_resamples
            if bootstrap_resamples is None
            else bootstrap_resamples
        ),
        warmups=execution.warmups if warmups is None else warmups,
        settings={
            "retrieval_protocol": protocol.meta.worker_payload(),
            "chunking_embedding_protocol": chunking_protocol.meta.worker_payload(),
            "chunker_embedding": selected_pair,
            "chunker_embedding_decision_fingerprint": (
                stable_hash(
                    {
                        "source_run_id": embedding_decision.source_run_id,
                        "selected_candidates": embedding_decision.selected_candidates,
                    }
                )
                if embedding_decision is not None
                else "smoke-control"
            ),
            "device": device,
            "dtype": dtype,
        },
    )


def _selected_embedding(profile, path, chunking_protocol):
    if profile == "smoke":
        if path is not None:
            raise ValueError("Smoke retrieval/reranking does not accept selections")
        return chunking_protocol.smoke_pair, None
    decision = load_engineer_decision(
        path,
        exact=1,
        expected_source=("rag", "chunking-embedding", "validation"),
    )
    selected = decision.selected_candidates[0]
    if selected not in chunking_protocol.development_candidates:
        raise ValueError("Selected chunking/embedding pair is not declared")
    return selected, decision


def _selected_candidates(profile, path, declared, maximum_finalists):
    if profile == "smoke":
        if path is not None:
            raise ValueError("Smoke retrieval/reranking does not accept a shortlist")
        return declared
    if profile in {"preflight", "development"}:
        if path is not None:
            raise ValueError(f"{profile} runs the declared retrieval matrix")
        return declared
    decision = load_engineer_decision(
        path,
        exact=1 if profile == "locked" else None,
        maximum=1 if profile == "locked" else maximum_finalists,
        expected_source=(
            "rag",
            "retrieval-reranking",
            "validation" if profile == "locked" else "development",
        ),
    )
    unknown = sorted(set(decision.selected_candidates) - set(declared))
    if unknown:
        raise ValueError(
            "Retrieval decision contains undeclared candidates: " + ", ".join(unknown)
        )
    return owner_first(decision.selected_candidates)


def _qualification_identity(
    protocol, chunking_protocol, declared, selected_pair, model_lock
):
    execution = protocol.profile("development")
    return current_qualification_fingerprint(
        benchmark="retrieval-reranking",
        candidates=declared,
        protocols={
            "retrieval": protocol.meta,
            "chunking_embedding": chunking_protocol.meta,
        },
        revisions=model_lock_fingerprints(model_lock),
        execution={
            "device": execution.device,
            "dtype": execution.dtype,
            "batch_size": execution.batch_size,
            "vram_limit_mb": protocol.authoritative_peak_vram_mb,
            "chunker_embedding": selected_pair,
        },
        input_envelope={
            "pool_size": protocol.pool_size,
            "reranker_maximum_tokens": dict(protocol.reranker_maximum_tokens),
            "reject_truncation": protocol.reject_truncation,
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
