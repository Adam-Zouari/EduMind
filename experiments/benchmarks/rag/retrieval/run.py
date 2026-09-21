"""CLI for the complete retrieval/reranking benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import (
    load_selected_model_lock,
    model_revisions,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import split_candidate
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_CHUNKING_PROTOCOL_PATH,
    load_protocol as load_chunking_protocol,
)
from experiments.benchmarks.rag.retrieval.benchmark import run_in_fresh_process
from experiments.benchmarks.rag.retrieval.comparisons import parent_artifact_builder
from experiments.benchmarks.rag.retrieval.metrics import (
    directions_for,
    primary_metrics,
)
from experiments.benchmarks.rag.retrieval.profiles import (
    development_candidates,
    owner_first,
    parse_candidate,
    required_reranker_models,
    validation_candidates,
)
from experiments.benchmarks.rag.retrieval.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark complete first-stage retrieval and reranking stacks"
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "development", "validation"),
        default="smoke",
    )
    parser.add_argument(
        "--chunking-protocol",
        type=Path,
        default=DEFAULT_CHUNKING_PROTOCOL_PATH,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="strict versioned retrieval/reranking protocol",
    )
    parser.add_argument(
        "--embedding-selection",
        type=Path,
        help="decision selecting exactly one validated chunker/embedding pair",
    )
    parser.add_argument(
        "--shortlist",
        type=Path,
        help="development retrieval decision selecting up to three finalist stacks",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--compare-finalists",
        action="store_true",
        help="write explicit cross-stack finalist comparisons in validation",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args(argv)

    protocol = load_protocol(arguments.protocol)
    chunking_protocol = load_chunking_protocol(arguments.chunking_protocol)
    execution = protocol.profile(arguments.profile)
    device = arguments.device or execution.device
    dtype = arguments.dtype or execution.dtype
    if execution.hardware_required and (device, dtype) != (
        execution.device,
        execution.dtype,
    ):
        parser.error(
            "authoritative retrieval/reranking requires "
            f"--device {execution.device} --dtype {execution.dtype}"
        )
    if arguments.compare_finalists and arguments.profile != "validation":
        parser.error("--compare-finalists is valid only for validation")

    manifest_path = arguments.manifest or PROJECT_ROOT / {
        "smoke": "data/benchmarks/rag/smoke.json",
        "development": "data/benchmarks/rag/rag-selection-dev.json",
        "validation": "data/benchmarks/rag/rag-selection-validation.json",
    }[arguments.profile]
    manifest = load_manifest(manifest_path)
    require_manifest_split(
        manifest,
        arguments.profile,
        {
            "smoke": {"smoke"},
            "development": {"dev", "development"},
            "validation": {"validation"},
        }[arguments.profile],
    )

    declared = owner_first(development_candidates())
    for candidate in declared:
        parse_candidate(candidate)

    decision_files: dict[str, Path] = {}
    embedding_decision = None
    finalists: tuple[str, ...] = ()
    if arguments.profile == "smoke":
        if arguments.embedding_selection or arguments.shortlist:
            parser.error("smoke does not accept selection decisions")
        selected_pair = chunking_protocol.smoke_pair
        candidates = declared
    else:
        if arguments.embedding_selection is None:
            parser.error(
                "development and validation require --embedding-selection DECISION_JSON"
            )
        embedding_decision = load_engineer_decision(
            arguments.embedding_selection,
            exact=1,
            expected_source=("rag", "chunking-embedding", "validation"),
        )
        selected_pair = embedding_decision.selected_candidates[0]
        decision_files["chunking_embedding"] = arguments.embedding_selection
        if arguments.profile == "development":
            if arguments.shortlist is not None:
                parser.error("development runs the complete 15-candidate matrix")
            candidates = declared
        else:
            if arguments.shortlist is None:
                parser.error("validation requires --shortlist DECISION_JSON")
            retrieval_decision = load_engineer_decision(
                arguments.shortlist,
                maximum=protocol.maximum_finalists,
                expected_source=("rag", "retrieval-reranking", "development"),
            )
            unknown = sorted(
                set(retrieval_decision.selected_candidates) - set(declared)
            )
            if unknown:
                raise ValueError(
                    "Retrieval shortlist contains undeclared candidates: "
                    + ", ".join(unknown)
                )
            finalists = retrieval_decision.selected_candidates
            candidates = validation_candidates(finalists, declared)
            decision_files["retrieval_reranking"] = arguments.shortlist

    if selected_pair not in chunking_protocol.development_candidates:
        raise ValueError(
            "Selected chunking/embedding pair is not declared by the supplied chunking protocol"
        )
    _, embedding_name = split_candidate(selected_pair)
    model_lock_path = PROJECT_ROOT / "data/benchmarks/models/selected.json"
    required_models = (embedding_name, *required_reranker_models(candidates))
    model_lock = load_selected_model_lock(
        model_lock_path,
        candidates=required_models,
    )

    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        arguments.profile,
        manifest.name,
        candidates,
        seed=protocol.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
        settings={
            "retrieval_protocol": protocol.metadata(arguments.protocol).worker_payload(),
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
            "requested_finalists": list(finalists),
        },
    )
    directions, required_metrics = directions_for(manifest, protocol)
    pools: dict[str, dict[str, object]] = {}

    def evaluate(candidate_name: str, context: Mapping[str, object]):
        parsed = parse_candidate(candidate_name)
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
            artifacts = dict(evaluated[5])
            rows = artifacts.get("candidate_pool")
            index_build = artifacts.get("index_build")
            if not isinstance(rows, list) or not isinstance(index_build, Mapping):
                raise RuntimeError(f"{candidate_name} did not return its owned pool")
            pool_checksum = stable_hash(rows)
            parameters = dict(evaluated[3])
            if parameters.get("pool_checksum") != pool_checksum:
                raise RuntimeError(f"{candidate_name} returned an inconsistent pool hash")
            pools[candidate_name] = {
                "owner_candidate": candidate_name,
                "owner_run_id": str(context["mlflow_run_id"]),
                "pool_checksum": pool_checksum,
                "index_checksum": index_build.get("index_sha256"),
                "rows": rows,
            }
        return evaluated

    result = run_benchmark(
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
        },
        protocols={
            "retrieval": protocol.metadata(arguments.protocol),
            "chunking_embedding": chunking_protocol.meta,
        },
        no_mlflow=arguments.no_mlflow,
        paired_comparisons=False,
        candidate_artifact_name="candidate.json",
        sample_artifact_name="query_metrics",
        resource_artifact_name="resources",
        resource_monitor_options=lambda candidate: {
            "require_vram": (
                device == "cuda" and parse_candidate(candidate).model_backed
            ),
            "report_zero_vram": (
                device == "cpu" or not parse_candidate(candidate).model_backed
            ),
            "zero_vram_measurement_method": (
                "cpu-zero" if device == "cpu" else "not-applicable-zero"
            ),
        },
        monitor_temporary_disk=False,
        run_name_prefix=f"rag-retrieval-reranking-{arguments.profile}",
        shuffle_candidates=False,
        evaluator_receives_context=True,
        parent_artifact_builder=parent_artifact_builder(
            pools,
            directions,
            compare_finalists=arguments.compare_finalists,
        ),
        operational_maximums=(
            {"peak_vram_mb": protocol.authoritative_peak_vram_mb}
            if execution.hardware_required
            else None
        ),
    )
    print(
        json.dumps(
            {"run_id": result.run_id, "artifacts": str(result.artifact_directory)},
            indent=2,
        )
    )
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
