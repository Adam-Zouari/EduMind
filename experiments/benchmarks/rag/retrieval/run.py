from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import load_candidates, parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.evaluation import RETRIEVAL_DIRECTIONS, evaluate


def _summary_selections(path: Path, *, maximum: int) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("pareto_candidates")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} has no Pareto candidates")
    if len(values) > maximum:
        raise ValueError(
            f"{path} has {len(values)} candidates; explicitly approve at most {maximum} first"
        )
    return tuple(str(value) for value in values)


directory = Path(__file__).parent
argument_parser = parser("Benchmark retrieval and reranking strategies")
argument_parser.add_argument(
    "--embedding-summary",
    type=Path,
    help="chunking/embedding summary.json; its Pareto pairs are crossed with retrieval methods",
)
arguments = argument_parser.parse_args()
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
        _summary_selections(arguments.embedding_summary, maximum=3)
        if arguments.embedding_summary
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
    revisions=revisions,
    no_mlflow=arguments.no_mlflow,
)
print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
raise SystemExit(0 if all(row.status == "success" for row in result.candidates) else 2)
