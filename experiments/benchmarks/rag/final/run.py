from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.paths import PROJECT_ROOT
from edumind.common.artifacts import atomic_write_json
from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.prepare import load_model_lock
from experiments.benchmarks.rag.evaluation import build_index
from experiments.benchmarks.rag.generation.evaluate import evaluate_candidate

def _selections(path: Path, maximum: int) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("pareto_candidates")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} has no Pareto candidates")
    if len(values) > maximum:
        raise ValueError(f"{path} must be explicitly reduced to at most {maximum} candidates")
    return tuple(str(value) for value in values)


directory = Path(__file__).parent
argument_parser = parser("Benchmark shortlisted complete RAG systems")
argument_parser.add_argument("--retrieval-summary", type=Path)
argument_parser.add_argument("--generation-summary", type=Path)
argument_parser.add_argument("--review-results", type=Path)
argument_parser.add_argument("--confirm-locked-test", action="store_true")
arguments = argument_parser.parse_args()
manifest_path = arguments.manifest or PROJECT_ROOT / (
    "data/benchmarks/rag/smoke.json"
    if arguments.profile == "smoke"
    else f"data/benchmarks/rag/qasper-{'validation' if arguments.profile == 'standard' else 'locked-test'}.json"
)
manifest = load_manifest(manifest_path)
candidates = resolved_candidates(directory / "candidates.yaml", arguments.profile, arguments.shortlist)
if arguments.shortlist is None and (arguments.retrieval_summary or arguments.generation_summary):
    if not arguments.retrieval_summary or not arguments.generation_summary:
        raise ValueError("Provide both --retrieval-summary and --generation-summary")
    retrievals = _selections(arguments.retrieval_summary, 3)
    generators = _selections(arguments.generation_summary, 3)
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
digests = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/ollama.json")
revisions = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/huggingface.json")
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
            manifest, chunker, embedding, revisions, with_bm25=True
        )
    return evaluate_candidate(
        generator,
        manifest,
        digests,
        revisions,
        final_index=indexes[pair],
        retrieval_method=retrieval,
        top_k=int(top_k_value.removeprefix("top_k=")),
        repetitions=plan.repetitions,
    )

result = run_benchmark(
    plan,
    evaluate,
    dataset_checksum=manifest.fingerprint,
    directions={
        "ndcg_at_3": "max",
        "ndcg_at_5": "max",
        "context_recall_at_3": "max",
        "context_recall_at_5": "max",
        "context_precision_at_3": "max",
        "context_precision_at_5": "max",
        "context_recall_at_2048_tokens": "max",
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
if arguments.profile == "full" and all(row.status == "success" for row in result.candidates):
    atomic_write_json(
        locked_marker,
        {
            "run_id": result.run_id,
            "candidate": candidates[0],
            "review_results": str(arguments.review_results),
            "artifact_directory": str(result.artifact_directory),
        },
    )
raise SystemExit(0 if all(row.status == "success" for row in result.candidates) else 2)
