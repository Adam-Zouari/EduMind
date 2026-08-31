from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.generation.evaluate import GENERATION_DIRECTIONS, evaluate_candidate

directory = Path(__file__).parent
argument_parser = parser("Benchmark direct Hugging Face generation on frozen contexts")
argument_parser.add_argument(
    "--device", choices=("cpu", "cuda"), help="Whole-model device shared by every candidate"
)
arguments = argument_parser.parse_args()
if arguments.profile in {"standard", "full"} and arguments.device is None:
    argument_parser.error("standard/full generation requires explicit --device cpu|cuda")
device = arguments.device or "cpu"
manifest_path = arguments.manifest or PROJECT_ROOT / (
    "data/benchmarks/rag/smoke.json"
    if arguments.profile == "smoke"
    else f"data/benchmarks/rag/rag-selection-{'dev' if arguments.profile == 'standard' else 'validation'}.json"
)
manifest = load_manifest(manifest_path)
candidates = resolved_candidates(directory / "candidates.yaml", arguments.profile, arguments.shortlist)
model_lock = load_selected_model_lock(
    PROJECT_ROOT / "data/benchmarks/models/selected.json"
)
revisions = model_revisions(model_lock)
plan = BenchmarkPlan(
    "rag",
    "generation",
    arguments.profile,
    manifest.name,
    candidates,
    repetitions=1 if arguments.profile == "smoke" else 3,
    bootstrap_resamples=0 if arguments.profile == "smoke" else 10_000,
    warmups=2,
)
result = run_benchmark(
    plan,
    lambda candidate: evaluate_candidate(
        candidate,
        manifest,
        model_lock,
        repetitions=plan.repetitions,
        device=device,
        bootstrap_resamples=plan.bootstrap_resamples,
        bootstrap_seed=plan.seed,
    ),
    dataset_checksum=manifest.fingerprint,
    directions=GENERATION_DIRECTIONS,
    primary_metric="citation_f1",
    revisions=revisions,
    decision_files={"shortlist": arguments.shortlist} if arguments.shortlist else None,
    no_mlflow=arguments.no_mlflow,
)
print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
raise SystemExit(0 if result.complete else 2)
