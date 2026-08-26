from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from edumind.common.artifacts import atomic_write_json
from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.evaluation import RETRIEVAL_QUALITY_DIRECTIONS, build_index
from experiments.benchmarks.rag.generation.evaluate import GENERATION_DIRECTIONS, evaluate_candidate

directory = Path(__file__).parent
argument_parser = parser("Benchmark shortlisted complete RAG systems")
argument_parser.add_argument("--retrieval-selection", type=Path)
argument_parser.add_argument("--generation-selection", type=Path)
argument_parser.add_argument("--review-results", type=Path)
argument_parser.add_argument("--confirm-locked-test", action="store_true")
argument_parser.add_argument(
    "--device", choices=("cpu", "cuda"), help="Whole-model generator device"
)
arguments = argument_parser.parse_args()
if (
    arguments.profile == "standard"
    and arguments.shortlist is None
    and (arguments.retrieval_selection is None or arguments.generation_selection is None)
):
    argument_parser.error(
        "standard final RAG requires --retrieval-selection and --generation-selection"
    )
if arguments.profile in {"standard", "full"} and arguments.device is None:
    argument_parser.error("standard/full final RAG requires explicit --device cpu|cuda")
device = arguments.device or "cpu"
manifest_path = arguments.manifest or PROJECT_ROOT / (
    "data/benchmarks/rag/smoke.json"
    if arguments.profile == "smoke"
    else f"data/benchmarks/rag/rag-selection-{'validation' if arguments.profile == 'standard' else 'locked-test'}.json"
)
manifest = load_manifest(manifest_path)
candidates = resolved_candidates(directory / "candidates.yaml", arguments.profile, arguments.shortlist)
if arguments.shortlist is None and (arguments.retrieval_selection or arguments.generation_selection):
    if not arguments.retrieval_selection or not arguments.generation_selection:
        raise ValueError("Provide both --retrieval-selection and --generation-selection")
    retrievals = load_engineer_decision(
        arguments.retrieval_selection, maximum=3
    ).selected_candidates
    generators = load_engineer_decision(
        arguments.generation_selection, maximum=3
    ).selected_candidates
    candidates = tuple(
        f"{retrieval}@@{generator}@@top_k={top_k}"
        for retrieval in retrievals
        for generator in generators
        for top_k in (3, 5)
    )
if arguments.profile == "full" and len(candidates) != 1:
    raise ValueError("Locked-test full evaluation requires exactly one approved final candidate")
locked_marker = PROJECT_ROOT / "artifacts/benchmarks/rag/final/locked-test-v1.json"
if arguments.profile == "full":
    if not arguments.confirm_locked_test or not arguments.review_results:
        raise ValueError(
            "Full is the one-time locked test. Provide --review-results and "
            "--confirm-locked-test after blinded review."
        )
    review = json.loads(arguments.review_results.read_text(encoding="utf-8"))
    if not review.get("complete") or int(review.get("judgment_count", 0)) != 60:
        raise ValueError("Locked test requires a complete imported 60-judgment review")
    if candidates[0] not in review.get("candidates", {}):
        raise ValueError("The locked-test candidate was not one of the reviewed systems")
    if locked_marker.exists():
        raise ValueError(
            f"Locked test v1 was already consumed; see {locked_marker}. "
            "Create a new benchmark version before another test evaluation."
        )
model_lock = load_selected_model_lock(
    PROJECT_ROOT / "data/benchmarks/models/selected.json"
)
revisions = model_revisions(model_lock)
indexes = {}
plan = BenchmarkPlan(
    "rag",
    "final",
    arguments.profile,
    manifest.name,
    candidates,
    repetitions=1 if arguments.profile == "smoke" else 3,
    bootstrap_resamples=500 if arguments.profile == "smoke" else 10_000,
    warmups=2,
)

def evaluate(candidate):
    if "@@" in candidate:
        chunker, embedding, retrieval, generator, top_k_value = candidate.split("@@", 4)
    else:
        retrieval, generator, top_k_value = candidate.split("|", 2)
        chunker, embedding = "token-256-32", "sentence-transformers/all-MiniLM-L6-v2"
    pair = (chunker, embedding)
    if pair not in indexes:
        indexes[pair] = build_index(
            manifest, chunker, embedding, model_lock, with_bm25=True
        )
    return evaluate_candidate(
        generator,
        manifest,
        model_lock,
        final_index=indexes[pair],
        retrieval_method=retrieval,
        top_k=int(top_k_value.removeprefix("top_k=")),
        repetitions=plan.repetitions,
        device=device,
        bootstrap_resamples=plan.bootstrap_resamples,
        bootstrap_seed=plan.seed,
    )

result = run_benchmark(
    plan,
    evaluate,
    dataset_checksum=manifest.fingerprint,
    directions={**GENERATION_DIRECTIONS, **RETRIEVAL_QUALITY_DIRECTIONS},
    primary_metric="citation_f1",
    revisions=revisions,
    decision_files={
        name: path
        for name, path in {
            "shortlist": arguments.shortlist,
            "retrieval": arguments.retrieval_selection,
            "generation": arguments.generation_selection,
        }.items()
        if path is not None
    },
    no_mlflow=arguments.no_mlflow,
)
print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
if arguments.profile == "full" and result.complete:
    atomic_write_json(
        locked_marker,
        {
            "run_id": result.run_id,
            "candidate": candidates[0],
            "review_results": str(arguments.review_results),
            "artifact_directory": str(result.artifact_directory),
        },
    )
raise SystemExit(0 if result.complete else 2)
