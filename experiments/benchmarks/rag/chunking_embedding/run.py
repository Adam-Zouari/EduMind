from pathlib import Path
import json

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import parser
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.chunking_embedding.benchmark import (
    directions_for,
    run_in_fresh_process,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)

if __name__ == "__main__":
    argument_parser = parser("Benchmark chunking and embedding pairs")
    argument_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    argument_parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    arguments = argument_parser.parse_args()
    protocol = load_protocol(arguments.protocol)
    execution = protocol.profile(arguments.profile)
    if execution.hardware_required and (arguments.device, arguments.dtype) != (
        execution.device,
        execution.dtype,
    ):
        argument_parser.error(
            f"{arguments.profile} requires --device {execution.device} "
            f"--dtype {execution.dtype}"
        )
    manifest_path = arguments.manifest or PROJECT_ROOT / (
        "data/benchmarks/rag/smoke.json"
        if arguments.profile == "smoke"
        else f"data/benchmarks/rag/rag-selection-{'dev' if arguments.profile == 'development' else 'validation'}.json"
    )
    manifest = load_manifest(manifest_path)
    require_manifest_split(manifest, arguments.profile, {
        "smoke": {"smoke"},
        "development": {"dev", "development"},
        "validation": {"validation"},
    }[arguments.profile])
    declared = protocol.development_candidates
    if arguments.profile == "development":
        if arguments.shortlist is not None:
            raise ValueError(
                "Development chunking/embedding must run the complete declared matrix"
            )
        candidates = declared
    elif arguments.profile == "validation":
        if arguments.shortlist is None:
            raise ValueError(
                "Validation chunking/embedding requires --shortlist DECISION_JSON"
            )
        decision = load_engineer_decision(
            arguments.shortlist,
            maximum=protocol.maximum_finalists,
            expected_source=("rag", "chunking-embedding", "development"),
        )
        candidates = decision.selected_candidates
        unknown = sorted(set(candidates) - set(declared))
        if unknown:
            raise ValueError(
                "Chunking/embedding shortlist contains undeclared pairs: "
                + ", ".join(unknown)
            )
    else:
        if arguments.shortlist is not None:
            raise ValueError("Smoke chunking/embedding does not accept a shortlist")
        candidates = (protocol.smoke_pair,)
    embedding_names = tuple(sorted({candidate.split("|", 1)[1] for candidate in candidates}))
    model_lock_path = PROJECT_ROOT / "data/benchmarks/models/selected.json"
    model_lock = load_selected_model_lock(
        model_lock_path,
        candidates=embedding_names,
    )
    revisions = model_revisions(model_lock)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        arguments.profile,
        manifest.name,
        candidates,
        seed=protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
        settings={
            "device": arguments.device,
            "dtype": arguments.dtype,
            "chunking_embedding_protocol": protocol.meta.worker_payload(),
        },
    )
    directions, required_metrics = directions_for(manifest, protocol)

    result = run_benchmark(
        plan,
        lambda candidate: run_in_fresh_process(
            candidate,
            manifest,
            model_lock,
            plan,
            device=arguments.device,
            dtype=arguments.dtype,
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
                *(f"quality.overall.{name}" for name in protocol.primary_quality_metrics),
                *(f"quality.overall.{name}" for name in protocol.alpha_ndcg_metrics),
            )
            if metric in directions
        ),
        revisions=revisions,
        decision_files={"shortlist": arguments.shortlist} if arguments.shortlist else None,
        input_artifacts={"manifest": manifest_path, "model_lock": model_lock_path},
        protocols={"chunking_embedding": protocol.meta},
        no_mlflow=arguments.no_mlflow,
        candidate_artifact_name="candidate.json",
        sample_artifact_name="query_metrics",
        resource_artifact_name="resources",
        resource_monitor_options={
            "require_vram": arguments.device == "cuda",
            "report_zero_vram": arguments.device == "cpu",
        },
        monitor_temporary_disk=False,
        paired_group_key="document_id",
        run_name_prefix={
            "smoke": "rag-chunking-embedding-smoke",
            "development": "rag-chunking-embedding-development",
            "validation": "rag-chunking-embedding-validation",
        }[arguments.profile],
        operational_maximums=(
            {"peak_vram_mb": protocol.authoritative_peak_vram_mb}
            if execution.hardware_required
            else None
        ),
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    raise SystemExit(0 if result.complete else 2)
