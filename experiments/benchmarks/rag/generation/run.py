import json
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import (
    load_selected_model_lock,
    model_revisions,
)
from experiments.benchmarks.rag.generation.evaluate import (
    GENERATION_DIRECTIONS,
    evaluate_candidate,
)
from experiments.benchmarks.rag.generation.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)

if __name__ == "__main__":
    argument_parser = parser(
        "Benchmark direct Hugging Face generation on frozen contexts"
    )
    argument_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Whole-model device shared by every candidate",
    )
    argument_parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16", "auto")
    )
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    arguments = argument_parser.parse_args()
    protocol = load_protocol(arguments.protocol)
    execution = protocol.profile(arguments.profile)
    if arguments.profile in {"development", "validation"} and arguments.device is None:
        argument_parser.error(
            "development/validation generation requires explicit --device cpu|cuda"
        )
    device = arguments.device or execution.device
    dtype = arguments.dtype or execution.dtype
    if execution.hardware_required and (device, dtype) != (
        execution.device,
        execution.dtype,
    ):
        argument_parser.error(
            f"{arguments.profile} generation requires --device {execution.device} "
            f"--dtype {execution.dtype}"
        )
    manifest_path = arguments.manifest or PROJECT_ROOT / (
        "data/benchmarks/rag/smoke.json"
        if arguments.profile == "smoke"
        else f"data/benchmarks/rag/rag-selection-{'dev' if arguments.profile == 'development' else 'validation'}.json"
    )
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
    candidates = resolved_candidates(
        tuple(protocol.models),
        arguments.profile,
        arguments.shortlist,
        expected_source=("rag", "generation", "development"),
        maximum=protocol.maximum_finalists,
    )
    required_models = tuple(
        dict.fromkeys(
            [
                *(protocol.model_id(candidate) for candidate in candidates),
                protocol.faithfulness_model,
            ]
        )
    )
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=required_models,
    )
    revisions = model_revisions(model_lock)
    plan = BenchmarkPlan(
        "rag",
        "generation",
        arguments.profile,
        manifest.name,
        candidates,
        seed=protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
    )
    result = run_benchmark(
        plan,
        lambda candidate: evaluate_candidate(
            candidate,
            manifest,
            model_lock,
            final_index=None,
            retrieval_method="frozen",
            top_k=None,
            repetitions=plan.repetitions,
            device=device,
            dtype=dtype,
            bootstrap_resamples=plan.bootstrap_resamples,
            bootstrap_seed=plan.seed,
            protocol=protocol,
            retrieval_protocol=None,
            warmups=plan.warmups,
            question_scope=(
                "development-screen" if arguments.profile == "development" else "all"
            ),
        ),
        dataset_checksum=manifest.fingerprint,
        directions=GENERATION_DIRECTIONS,
        primary_metric="citation_f1",
        revisions=revisions,
        decision_files={"shortlist": arguments.shortlist}
        if arguments.shortlist
        else None,
        input_artifacts={"manifest": manifest_path},
        protocols={"generation": protocol.meta},
        no_mlflow=arguments.no_mlflow,
        operational_maximums=(
            {"model_peak_vram_mb": protocol.authoritative_peak_vram_mb}
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
    raise SystemExit(0 if result.complete else 2)
