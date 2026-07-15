from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.prepare import load_model_lock
from experiments.benchmarks.rag.generation.evaluate import evaluate_candidate

directory = Path(__file__).parent
arguments = parser("Benchmark Ollama generation on frozen oracle contexts").parse_args()
manifest_path = arguments.manifest or PROJECT_ROOT / (
    "data/benchmarks/rag/smoke.json"
    if arguments.profile == "smoke"
    else f"data/benchmarks/rag/qasper-{'dev' if arguments.profile == 'standard' else 'validation'}.json"
)
manifest = load_manifest(manifest_path)
candidates = resolved_candidates(directory / "candidates.yaml", arguments.profile, arguments.shortlist)
digests = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/ollama.json")
revisions = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/huggingface.json")
plan = BenchmarkPlan(
    "rag",
    "generation",
    arguments.profile,
    manifest.name,
    candidates,
    repetitions=1 if arguments.profile == "smoke" else 3,
    bootstrap_resamples=500 if arguments.profile == "smoke" else 10_000,
    warmups=2,
)
result = run_benchmark(
    plan,
    lambda candidate: evaluate_candidate(
        candidate, manifest, digests, revisions, repetitions=plan.repetitions
    ),
    dataset_checksum=manifest.fingerprint,
    directions={
        "citation_f1": "max",
        "answerability_balanced_accuracy": "max",
        "nli_faithfulness": "max",
        "operational.p95_latency_seconds": "min",
        "operational.combined_process_ollama_memory_gb": "min",
    },
    gates={
        "malformed_output_rate": ("min", 0.0),
        "operational.p95_latency_seconds": ("min", 30.0),
        "operational.combined_process_ollama_memory_gb": ("min", 28.0),
    },
    revisions={**revisions, **digests},
    no_mlflow=arguments.no_mlflow,
)
print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
raise SystemExit(0 if all(row.status == "success" for row in result.candidates) else 2)
