"""Run direct generation benchmarks on frozen retrieval contexts."""

from __future__ import annotations

import json
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import (
    default_decision_path,
    execution_devices,
    parser,
    resolved_candidates,
)
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


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser(
        "Benchmark direct Hugging Face generation on frozen contexts"
    )
    argument_parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    argument_parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16", "auto")
    )
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    arguments = argument_parser.parse_args(argv)
    protocol = load_protocol(arguments.protocol)
    execution = protocol.profile(arguments.profile)
    try:
        devices = execution_devices(
            arguments.profile,
            arguments.device,
            smoke_devices=protocol.profile("smoke").devices,
            authoritative_device=execution.device,
            allow_preflight=False,
        )
    except ValueError as exc:
        argument_parser.error(str(exc))
    if arguments.profile != "smoke" and arguments.dtype not in {None, execution.dtype}:
        argument_parser.error(
            f"{arguments.profile} generation requires dtype {execution.dtype}"
        )
    manifest_path = (arguments.manifest or _manifest(arguments.profile)).resolve()
    manifest = load_manifest(manifest_path)
    require_manifest_split(manifest, arguments.profile, _split(arguments.profile))
    shortlist = arguments.shortlist or default_decision_path(
        "generation", arguments.profile
    )
    candidates = resolved_candidates(
        tuple(protocol.models),
        arguments.profile,
        shortlist,
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
    results = []
    for device in devices:
        dtype = arguments.dtype or execution.dtype_for(device)
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
            settings={"device": device, "dtype": dtype},
        )

        def evaluate(candidate, *, plan=plan, device=device, dtype=dtype):
            return evaluate_candidate(
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
                    "development-screen"
                    if arguments.profile == "development"
                    else "all"
                ),
            )

        results.append(
            run_benchmark(
                plan,
                evaluate,
                dataset_checksum=manifest.fingerprint,
                directions=GENERATION_DIRECTIONS,
                primary_metric="citation_f1",
                revisions=model_revisions(model_lock),
                decision_files={"shortlist": shortlist} if shortlist else None,
                input_artifacts={"manifest": manifest_path},
                protocols={"generation": protocol.meta},
                no_mlflow=arguments.no_mlflow,
                resource_monitor_options={
                    "require_vram": device == "cuda",
                    "report_zero_vram": device == "cpu",
                },
                operational_maximums=(
                    {
                        "model_peak_vram_mb": protocol.authoritative_peak_vram_mb,
                        "peak_vram_mb": protocol.authoritative_peak_vram_mb,
                    }
                    if arguments.profile != "smoke"
                    else None
                ),
                run_name_prefix=(
                    f"rag-generation-smoke-{device}"
                    if arguments.profile == "smoke"
                    else f"rag-generation-{arguments.profile}"
                ),
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


def _manifest(profile: str) -> Path:
    suffix = {
        "smoke": "smoke",
        "development": "rag-selection-dev",
        "validation": "rag-selection-validation",
    }[profile]
    return PROJECT_ROOT / "data/benchmarks/rag" / f"{suffix}.json"


def _split(profile: str):
    return {
        "smoke": {"smoke"},
        "development": {"dev", "development"},
        "validation": {"validation"},
    }[profile]


if __name__ == "__main__":
    raise SystemExit(main())
