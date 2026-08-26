from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import load_candidates, parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.evaluation import RETRIEVAL_DIRECTIONS, evaluate


directory = Path(__file__).parent
argument_parser = parser("Benchmark retrieval and reranking strategies")
argument_parser.add_argument(
    "--embedding-selection",
    type=Path,
    help="engineer decision selecting up to three chunking/embedding pairs",
)
arguments = argument_parser.parse_args()
if arguments.profile == "standard" and arguments.embedding_selection is None:
    argument_parser.error("standard retrieval requires --embedding-selection DECISION_JSON")
manifest_path = arguments.manifest or PROJECT_ROOT / (
    "data/benchmarks/rag/smoke.json"
    if arguments.profile == "smoke"
    else f"data/benchmarks/rag/rag-selection-{'dev' if arguments.profile == 'standard' else 'validation'}.json"
)
manifest = load_manifest(manifest_path)
if arguments.shortlist:
    candidates = resolved_candidates(
        directory / "candidates.yaml", arguments.profile, arguments.shortlist
    )
else:
    methods = load_candidates(directory / "candidates.yaml", arguments.profile)
    pairs = (
        load_engineer_decision(
            arguments.embedding_selection, maximum=3
        ).selected_candidates
        if arguments.embedding_selection
        else ("token-256-32|sentence-transformers/all-MiniLM-L6-v2",)
    )
    candidates = tuple(
        f"{pair.replace('|', '@@', 1)}@@{method}" for pair in pairs for method in methods
    )
model_lock = load_selected_model_lock(
    PROJECT_ROOT / "data/benchmarks/models/selected.json"
)
revisions = model_revisions(model_lock)
plan = BenchmarkPlan(
    "rag",
    "retrieval",
    arguments.profile,
    manifest.name,
    candidates,
    repetitions=1 if arguments.profile == "smoke" else 3,
    bootstrap_resamples=500 if arguments.profile == "smoke" else 10_000,
)

result = run_benchmark(
    plan,
    lambda candidate: evaluate(
        manifest,
        *candidate.split("@@", 2),
        model_lock,
        plan.repetitions,
    ),
    dataset_checksum=manifest.fingerprint,
    directions=RETRIEVAL_DIRECTIONS,
    primary_metric="ndcg_at_5",
    revisions=revisions,
    decision_files={
        name: path
        for name, path in {
            "shortlist": arguments.shortlist,
            "embedding": arguments.embedding_selection,
        }.items()
        if path is not None
    },
    no_mlflow=arguments.no_mlflow,
)
print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
raise SystemExit(0 if result.complete else 2)
