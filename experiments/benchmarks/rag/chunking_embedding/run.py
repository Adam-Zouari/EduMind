from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import load_candidates, parser
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.chunking_embedding.benchmark import (
    PAIRED_METRICS,
    PRIMARY_METRICS,
    directions_for,
    run_in_fresh_process,
)

directory = Path(__file__).parent
argument_parser = parser("Benchmark chunking and embedding pairs")
argument_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
argument_parser.add_argument(
    "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
)
arguments = argument_parser.parse_args()
manifest_path = arguments.manifest or PROJECT_ROOT / (
    "data/benchmarks/rag/smoke.json"
    if arguments.profile == "smoke"
    else f"data/benchmarks/rag/rag-selection-{'dev' if arguments.profile == 'standard' else 'validation'}.json"
)
manifest = load_manifest(manifest_path)
require_manifest_split(manifest, arguments.profile, {
    "smoke": {"smoke"},
    "standard": {"dev", "development"},
    "full": {"validation"},
}[arguments.profile])
candidate_path = directory / "candidates.yaml"
declared = load_candidates(candidate_path, "standard")
if arguments.profile == "standard":
    if arguments.shortlist is not None:
        raise ValueError("Standard chunking/embedding must run the complete declared matrix")
    candidates = declared
elif arguments.profile == "full":
    if arguments.shortlist is None:
        raise ValueError("Full chunking/embedding requires --shortlist DECISION_JSON")
    decision = load_engineer_decision(
        arguments.shortlist,
        maximum=3,
        expected_source=("rag", "chunking-embedding", "standard"),
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
    candidates = load_candidates(candidate_path, "smoke")
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
    repetitions=1 if arguments.profile == "smoke" else 3,
    bootstrap_resamples=0 if arguments.profile == "smoke" else 10_000,
    warmups=1 if arguments.profile == "smoke" else 2,
    settings={
        "device": arguments.device,
        "dtype": arguments.dtype,
        "evaluation_tokenizer": "tiktoken:cl100k_base",
        "top_k": 5,
        "artifact_top_k": 20,
        "alpha": 0.5,
    },
)
directions, required_metrics = directions_for(manifest)

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
    primary_metric=PRIMARY_METRICS,
    required_metrics=required_metrics,
    paired_metrics=PAIRED_METRICS,
    revisions=revisions,
    decision_files={"shortlist": arguments.shortlist} if arguments.shortlist else None,
    input_artifacts={"manifest": manifest_path, "model_lock": model_lock_path},
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
        "standard": "rag-chunking-embedding-standard-development",
        "full": "rag-chunking-embedding-full-validation",
    }[arguments.profile],
)
print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
raise SystemExit(0 if result.complete else 2)
