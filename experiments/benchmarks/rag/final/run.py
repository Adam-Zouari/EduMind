from pathlib import Path
import json
from edumind.common.paths import PROJECT_ROOT
from edumind.common.artifacts import atomic_write_json
from experiments.benchmarks.common.arguments import parser
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_CHUNKING_PROTOCOL_PATH,
    load_protocol as load_chunking_protocol,
)
from experiments.benchmarks.rag.evaluation import build_index, retrieval_quality_directions
from experiments.benchmarks.rag.generation.evaluate import GENERATION_DIRECTIONS, evaluate_candidate
from experiments.benchmarks.rag.generation.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_GENERATION_PROTOCOL_PATH,
    load_protocol as load_generation_protocol,
)
from experiments.benchmarks.rag.retrieval.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_RETRIEVAL_PROTOCOL_PATH,
    load_protocol as load_retrieval_protocol,
)
from experiments.benchmarks.rag.retrieval.profiles import parse_candidate
from experiments.benchmarks.rag.final.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_FINAL_PROTOCOL_PATH,
    load_protocol as load_final_protocol,
)


if __name__ == "__main__":
    argument_parser = parser(
        "Benchmark shortlisted complete RAG systems",
        profiles=("smoke", "development", "validation", "locked"),
    )
    argument_parser.add_argument("--retrieval-selection", type=Path)
    argument_parser.add_argument("--generation-selection", type=Path)
    argument_parser.add_argument("--review-results", type=Path)
    argument_parser.add_argument("--confirm-locked-test", action="store_true")
    argument_parser.add_argument(
        "--device", choices=("cpu", "cuda"), help="Whole-model generator device"
    )
    argument_parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16", "auto")
    )
    argument_parser.add_argument("--final-protocol", type=Path, default=DEFAULT_FINAL_PROTOCOL_PATH)
    argument_parser.add_argument("--generation-protocol", type=Path, default=DEFAULT_GENERATION_PROTOCOL_PATH)
    argument_parser.add_argument("--retrieval-protocol", type=Path, default=DEFAULT_RETRIEVAL_PROTOCOL_PATH)
    argument_parser.add_argument("--chunking-protocol", type=Path, default=DEFAULT_CHUNKING_PROTOCOL_PATH)
    arguments = argument_parser.parse_args()
    final_protocol = load_final_protocol(arguments.final_protocol)
    generation_protocol = load_generation_protocol(arguments.generation_protocol)
    retrieval_protocol = load_retrieval_protocol(arguments.retrieval_protocol)
    chunking_protocol = load_chunking_protocol(arguments.chunking_protocol)
    execution = final_protocol.profile(arguments.profile)
    if arguments.profile in {"smoke", "development"} and arguments.shortlist is not None:
        argument_parser.error(f"{arguments.profile} final RAG does not accept --shortlist")
    if arguments.profile in {"validation", "locked"} and arguments.shortlist is None:
        argument_parser.error(
            f"{arguments.profile} final RAG requires --shortlist DECISION_JSON"
        )
    if arguments.profile == "development" and (
        arguments.retrieval_selection is None or arguments.generation_selection is None
    ):
        argument_parser.error(
            "development final RAG requires --retrieval-selection and "
            "--generation-selection"
        )
    if arguments.profile != "development" and (
        arguments.retrieval_selection is not None
        or arguments.generation_selection is not None
    ):
        argument_parser.error(
            "retrieval and generation selections apply only to development final RAG"
        )
    if arguments.profile != "locked" and (
        arguments.review_results is not None or arguments.confirm_locked_test
    ):
        argument_parser.error(
            "review results and locked-test confirmation apply only to --profile locked"
        )
    if arguments.profile != "smoke" and arguments.device is None:
        argument_parser.error(
            "development/validation/locked final RAG requires explicit --device cpu|cuda"
        )
    device = arguments.device or execution.device
    dtype = arguments.dtype or execution.dtype
    if execution.hardware_required and (device, dtype) != (execution.device, execution.dtype):
        argument_parser.error(
            f"{arguments.profile} final RAG requires --device {execution.device} "
            f"--dtype {execution.dtype}"
        )
    manifest_path = arguments.manifest or PROJECT_ROOT / (
        "data/benchmarks/rag/smoke.json"
        if arguments.profile == "smoke"
        else f"data/benchmarks/rag/rag-selection-{ {'development': 'dev', 'validation': 'validation', 'locked': 'locked-test'}[arguments.profile] }.json"
    )
    manifest = load_manifest(manifest_path)
    require_manifest_split(
        manifest,
        arguments.profile,
        {
            "smoke": "smoke",
            "development": "dev",
            "validation": "validation",
            "locked": "locked-test",
        }[arguments.profile],
    )
    if arguments.profile in {"validation", "locked"}:
        expected_profile = (
            "development" if arguments.profile == "validation" else "validation"
        )
        candidates = load_engineer_decision(
            arguments.shortlist,
            expected_source=("rag", "final", expected_profile),
            exact=(
                1
                if arguments.profile == "locked"
                else final_protocol.human_review_system_count
            ),
            maximum=(
                final_protocol.human_review_system_count
                if arguments.profile == "validation"
                else None
            ),
        ).selected_candidates
    elif arguments.profile == "smoke":
        candidates = final_protocol.smoke_candidates()
    elif arguments.profile == "development":
        candidates = ()
    else:
        raise AssertionError(f"Unhandled Final RAG profile: {arguments.profile}")
    if arguments.profile == "development":
        if not arguments.retrieval_selection or not arguments.generation_selection:
            raise ValueError("Provide both --retrieval-selection and --generation-selection")
        retrieval_decision = load_engineer_decision(
            arguments.retrieval_selection,
            maximum=final_protocol.maximum_retrieval_finalists,
            expected_source=("rag", "retrieval-reranking", "validation"),
        )
        retrievals = retrieval_decision.selected_candidates
        retrieval_summary = json.loads(
            retrieval_decision.source_summary.read_text(encoding="utf-8")
        )
        selected_pair = str(
            retrieval_summary.get("plan", {})
            .get("settings", {})
            .get("chunker_embedding", "")
        )
        if selected_pair not in chunking_protocol.development_candidates:
            raise ValueError("Final RAG chunking/embedding selection is not in the supplied protocol")
        generators = load_engineer_decision(
            arguments.generation_selection,
            maximum=final_protocol.maximum_generation_finalists,
            expected_source=("rag", "generation", "validation"),
        ).selected_candidates
        candidates = tuple(
            f"{selected_pair.replace('|', '@@', 1)}@@{retrieval}@@{generator}@@top_k={top_k}"
            for retrieval in retrievals
            for generator in generators
            for top_k in final_protocol.top_k
        )
    if arguments.profile == "locked" and len(candidates) != 1:
        raise ValueError("Locked-test evaluation requires exactly one approved final candidate")
    if len(candidates) > final_protocol.maximum_final_systems:
        raise ValueError(
            "Final RAG candidate matrix exceeds protocol maximum_final_systems"
        )
    locked_marker = (
        PROJECT_ROOT
        / "artifacts/benchmarks/rag/final"
        / f"{final_protocol.locked_marker_version}.json"
    )
    if arguments.profile == "locked":
        if not arguments.confirm_locked_test or not arguments.review_results:
            raise ValueError(
                "Locked is the one-time test. Provide --review-results and "
                "--confirm-locked-test after blinded review."
            )
        review = json.loads(arguments.review_results.read_text(encoding="utf-8"))
        if (
            not review.get("complete")
            or int(review.get("judgment_count", 0))
            != final_protocol.required_judgment_count
        ):
            raise ValueError(
                "Locked test requires a complete imported "
                f"{final_protocol.required_judgment_count}-judgment review"
            )
        if candidates[0] not in review.get("candidates", {}):
            raise ValueError("The locked-test candidate was not one of the reviewed systems")
        if locked_marker.exists():
            raise ValueError(
                f"Locked test {final_protocol.locked_marker_version} was already consumed; "
                f"see {locked_marker}. "
                "Create a new benchmark version before another test evaluation."
            )
    required_models = {generation_protocol.faithfulness_model}
    for candidate in candidates:
        chunker, embedding, retrieval, generator, top_k_value = candidate.split("@@", 4)
        chunking_protocol.strategy(chunker)
        if embedding not in chunking_protocol.embedding_models:
            raise ValueError(f"Final RAG embedding model is not in the supplied protocol: {embedding}")
        parsed_retrieval = parse_candidate(retrieval)
        generator_model = generation_protocol.model_id(generator)
        if int(top_k_value.removeprefix("top_k=")) not in final_protocol.top_k:
            raise ValueError(f"Final system top_k is outside {final_protocol.top_k}")
        required_models.add(embedding)
        required_models.add(generator_model)
        if parsed_retrieval.reranker_model is not None:
            required_models.add(parsed_retrieval.reranker_model)
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=tuple(sorted(required_models)),
    )
    revisions = model_revisions(model_lock)
    indexes = {}
    plan = BenchmarkPlan(
        "rag",
        "final",
        arguments.profile,
        manifest.name,
        candidates,
        seed=final_protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
    )

    def evaluate(candidate):
        chunker, embedding, retrieval, generator, top_k_value = candidate.split("@@", 4)
        pair = (chunker, embedding)
        if pair not in indexes:
            indexes[pair] = build_index(
                manifest,
                chunker,
                embedding,
                model_lock,
                with_bm25=True,
                device=device,
                dtype=dtype,
                chunking_protocol=chunking_protocol,
                retrieval_protocol=retrieval_protocol,
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
            dtype=dtype,
            bootstrap_resamples=plan.bootstrap_resamples,
            bootstrap_seed=plan.seed,
            protocol=generation_protocol,
            retrieval_protocol=retrieval_protocol,
            warmups=plan.warmups,
            question_scope="all",
        )

    result = run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions={
            **GENERATION_DIRECTIONS,
            **retrieval_quality_directions(retrieval_protocol.quality_cutoffs),
        },
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
        input_artifacts={"manifest": manifest_path},
        protocols={
            "chunking_embedding": chunking_protocol.meta,
            "retrieval": retrieval_protocol.meta,
            "generation": generation_protocol.meta,
            "final_rag": final_protocol.meta,
        },
        no_mlflow=arguments.no_mlflow,
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    if arguments.profile == "locked" and result.complete:
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
