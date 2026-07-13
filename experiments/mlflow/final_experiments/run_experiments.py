"""Stage 6 final full-stack bakeoff for the English-only benchmark."""

from __future__ import annotations

import argparse
import logging
import time
from statistics import mean

from edumind.rag.errors import OllamaConnectionError, OllamaRequestError
from edumind.rag.llm_generator import OllamaGenerator
from edumind.rag.types import LLMSettings
from experiments.mlflow.benchmark import BenchmarkDataset, load_benchmark_dataset
from experiments.mlflow.harness import (
    StageCandidateResult,
    append_stage_score,
    build_stage_summary,
    collect_hardware_info,
    load_cached_candidate_result,
    load_stage_best_candidates,
    save_cached_candidate_result,
    save_stage_outputs,
    stage_results_to_artifact_payload,
)
from experiments.mlflow.mlflow_config import configure_mlflow
from experiments.mlflow.stage_specs import CHUNKING_CANDIDATES, EMBEDDING_CANDIDATES, FullStackCandidate, RetrievalStackCandidate, VECTOR_BACKEND_CANDIDATES
from experiments.mlflow.stage_utils import build_backend, build_chunk_records, build_embedder, evaluate_llm_answers
from experiments.mlflow.utils.metrics_logger import MLflowExperiment
from experiments.mlflow.vector_backends import VectorBackendUnavailableError

logger = logging.getLogger(__name__)

BASELINE_CHUNK_EMBEDDING = next(
    candidate for candidate in EMBEDDING_CANDIDATES if candidate.model_name == "BAAI/bge-base-en-v1.5"
)
DEFAULT_FINAL_STACK_LIMIT = 5


def run_all_experiments(
    *,
    dataset_name: str = "student_benchmark",
    split: str | None = "holdout",
    resume: bool = False,
    force: bool = False,
    stage_limit: int | None = None,
    top_n: int = 5,
    test_mode: bool = False,
) -> int:
    """Run the Stage 6 full-stack confirmation bakeoff."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if test_mode or dataset_name == "synthetic_regression":
        dataset_name = "synthetic_regression"
        split = "default"
        challenge_name = "synthetic_regression"
    else:
        challenge_name = "challenge_benchmark"

    holdout_dataset = load_benchmark_dataset(dataset_name, split=split)
    challenge_dataset = load_benchmark_dataset(challenge_name, split="default")
    full_stacks = _load_stage_five_stacks(holdout_dataset, count=stage_limit or DEFAULT_FINAL_STACK_LIMIT)
    if not full_stacks:
        logger.warning("No full stacks available for final evaluation.")
        return 1

    chunk_candidates = {candidate.name: candidate for candidate in CHUNKING_CANDIDATES}
    embedding_candidates = {candidate.model_name: candidate for candidate in EMBEDDING_CANDIDATES}
    backend_candidates = {candidate.name: candidate for candidate in VECTOR_BACKEND_CANDIDATES}
    baseline_chunk_embedder = build_embedder(BASELINE_CHUNK_EMBEDDING)
    holdout_chunks = {
        stack.retrieval_stack.chunker_name: build_chunk_records(
            holdout_dataset,
            chunk_candidates[stack.retrieval_stack.chunker_name],
            embedder=baseline_chunk_embedder,
        )
        for stack in full_stacks
    }
    challenge_chunks = {
        stack.retrieval_stack.chunker_name: build_chunk_records(
            challenge_dataset,
            chunk_candidates[stack.retrieval_stack.chunker_name],
            embedder=baseline_chunk_embedder,
        )
        for stack in full_stacks
    }
    results: list[StageCandidateResult] = []

    with MLflowExperiment(
        "final_experiments",
        run_name=f"final_{holdout_dataset.name}_{holdout_dataset.split}",
        tags={
            "stage": "final",
            "dataset": holdout_dataset.name,
            "split": holdout_dataset.split,
            "language": "en",
        },
    ) as parent_run:
        parent_run.log_artifact(
            "stage_inputs.json",
            {
                "holdout_dataset": {"name": holdout_dataset.name, "split": holdout_dataset.split},
                "challenge_dataset": {"name": challenge_dataset.name, "split": challenge_dataset.split},
                "full_stacks": [stack.to_dict() for stack in full_stacks],
            },
        )

        for stack in full_stacks[:top_n]:
            candidate_config = stack.to_dict()
            if resume and not force:
                cached_result = load_cached_candidate_result(
                    stage="final",
                    dataset_name=holdout_dataset.name,
                    dataset_version=holdout_dataset.version,
                    split=holdout_dataset.split,
                    candidate_config=candidate_config,
                )
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            with MLflowExperiment(
                "final_experiments",
                run_name=stack.name,
                nested=True,
                tags={"stage": "final", "candidate": stack.name},
            ) as child_run:
                child_run.log_params(
                    {
                        "chunker_name": stack.retrieval_stack.chunker_name,
                        "embedding_model": stack.retrieval_stack.embedding_model,
                        "vector_backend": stack.retrieval_stack.vector_backend,
                        "retrieval_name": stack.retrieval_stack.retrieval_name,
                        "bm25_alpha": stack.retrieval_stack.bm25_alpha,
                        "llm_model": stack.llm_model,
                        "dataset_name": holdout_dataset.name,
                        "dataset_version": holdout_dataset.version,
                        "split": holdout_dataset.split,
                    }
                )
                child_run.log_artifact("run_config.json", _build_run_config(holdout_dataset, candidate_config))

                try:
                    evaluation = _evaluate_full_stack(
                        stack=stack,
                        holdout_dataset=holdout_dataset,
                        challenge_dataset=challenge_dataset,
                        holdout_chunks=holdout_chunks[stack.retrieval_stack.chunker_name],
                        challenge_chunks=challenge_chunks[stack.retrieval_stack.chunker_name],
                        embedding_candidates=embedding_candidates,
                        backend_candidates=backend_candidates,
                    )
                except VectorBackendUnavailableError as exc:
                    skipped = StageCandidateResult(
                        stage="final",
                        dataset_name=holdout_dataset.name,
                        dataset_version=holdout_dataset.version,
                        split=holdout_dataset.split,
                        candidate_name=stack.name,
                        candidate_config=candidate_config,
                        status="skipped",
                        skip_reason=str(exc),
                    )
                    save_cached_candidate_result(skipped)
                    results.append(skipped)
                    child_run.log_artifact("skip_reason.txt", str(exc))
                    continue

                child_run.log_metrics(evaluation["metrics"])
                for filename, content in evaluation["artifacts"].items():
                    child_run.log_artifact(filename, content)
                result = StageCandidateResult(
                    stage="final",
                    dataset_name=holdout_dataset.name,
                    dataset_version=holdout_dataset.version,
                    split=holdout_dataset.split,
                    candidate_name=stack.name,
                    candidate_config=candidate_config,
                    metrics=evaluation["metrics"],
                    artifacts=evaluation["artifacts"],
                )
                save_cached_candidate_result(result)
                results.append(result)

        leaderboard = _build_leaderboard(results)
        recommendations = _build_recommendations(leaderboard)
        summary = build_stage_summary(
            stage="final",
            dataset_name=holdout_dataset.name,
            split=holdout_dataset.split,
            leaderboard=leaderboard,
            notes=["The final stage confirms promoted full stacks on holdout plus challenge data."],
        )
        save_stage_outputs(
            stage="final",
            dataset_name=holdout_dataset.name,
            split=holdout_dataset.split,
            leaderboard=leaderboard,
            best_candidates={"recommendations": recommendations},
            summary_markdown=summary,
        )
        parent_run.log_artifact("candidate_results.json", stage_results_to_artifact_payload(results))
        parent_run.log_artifact("leaderboard.json", leaderboard)
        parent_run.log_artifact("recommendations.json", recommendations)
        parent_run.log_artifact("stage_summary.md", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 6 CLI parser."""
    parser = argparse.ArgumentParser(description="Run the English-only final bakeoff.")
    parser.add_argument("--dataset", default="student_benchmark")
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Stage 6 final runner."""
    args = build_parser().parse_args(argv)
    return run_all_experiments(
        dataset_name=args.dataset,
        split=args.split,
        resume=args.resume,
        force=args.force,
        stage_limit=args.stage_limit,
        top_n=args.top_n,
        test_mode=args.test_mode,
    )


def _load_stage_five_stacks(dataset: BenchmarkDataset, count: int) -> list[FullStackCandidate]:
    dataset_name = "student_benchmark"
    split = "dev"
    if dataset.name == "synthetic_regression":
        dataset_name = dataset.name
        split = dataset.split
    previous = load_stage_best_candidates(
        stage="llm",
        dataset_name=dataset_name,
        split=split,
    )
    if previous is None:
        return []
    rows = previous.get("top_full_stacks", [])
    if not isinstance(rows, list):
        return []

    stacks: list[FullStackCandidate] = []
    for row in rows[:count]:
        if not isinstance(row, dict):
            continue
        retrieval_stack = RetrievalStackCandidate(
            chunker_name=str(row.get("chunker_name", "")),
            embedding_model=str(row.get("embedding_model", "")),
            vector_backend=str(row.get("vector_backend", "")),
            retrieval_name=str(row.get("retrieval_name", "")),
            bm25_alpha=float(row.get("bm25_alpha", 0.0)),
        )
        stacks.append(
            FullStackCandidate(
                retrieval_stack=retrieval_stack,
                llm_model=str(row.get("llm_model", "")),
            )
        )
    return stacks


def _evaluate_full_stack(
    *,
    stack: FullStackCandidate,
    holdout_dataset: BenchmarkDataset,
    challenge_dataset: BenchmarkDataset,
    holdout_chunks,
    challenge_chunks,
    embedding_candidates,
    backend_candidates,
) -> dict[str, object]:
    from experiments.mlflow.stage_utils import evaluate_retrieval_stack

    embedding_candidate = embedding_candidates.get(stack.retrieval_stack.embedding_model)
    backend_candidate = backend_candidates.get(stack.retrieval_stack.vector_backend)
    if embedding_candidate is None or backend_candidate is None:
        raise VectorBackendUnavailableError("Full stack references an unavailable embedding or backend candidate.")

    embedder = build_embedder(embedding_candidate)
    holdout_backend = build_backend(
        backend_spec=backend_candidate,
        stage="final_holdout",
        dataset_name=holdout_dataset.name,
        split=holdout_dataset.split,
        candidate_suffix=stack.name,
        bm25_alpha=stack.retrieval_stack.bm25_alpha,
    )
    challenge_backend = build_backend(
        backend_spec=backend_candidate,
        stage="final_challenge",
        dataset_name=challenge_dataset.name,
        split=challenge_dataset.split,
        candidate_suffix=stack.name,
        bm25_alpha=stack.retrieval_stack.bm25_alpha,
    )
    mode = _resolve_retrieval_mode(stack.retrieval_stack.retrieval_name)

    holdout_retrieval, holdout_payloads = evaluate_retrieval_stack(
        dataset=holdout_dataset,
        chunk_records=holdout_chunks,
        embedder=embedder,
        backend=holdout_backend,
        retrieval_mode=mode,
        bm25_alpha=stack.retrieval_stack.bm25_alpha,
        top_k=5,
        include_filters=True,
    )
    challenge_retrieval, challenge_payloads = evaluate_retrieval_stack(
        dataset=challenge_dataset,
        chunk_records=challenge_chunks,
        embedder=embedder,
        backend=challenge_backend,
        retrieval_mode=mode,
        bm25_alpha=stack.retrieval_stack.bm25_alpha,
        top_k=5,
        include_filters=True,
    )
    generator = OllamaGenerator(
        settings=LLMSettings(
            model_name=stack.llm_model,
            base_url="http://localhost:11434",
            temperature=0.3,
            max_tokens=256,
            request_timeout=120,
        )
    )
    if not generator.health_check():
        raise VectorBackendUnavailableError("Ollama is unavailable for final full-stack evaluation.")
    if stack.llm_model not in set(generator.list_models()):
        raise VectorBackendUnavailableError(f"Model '{stack.llm_model}' is not installed in Ollama.")

    holdout_answers, holdout_latencies = _generate_answers(generator, holdout_payloads)
    challenge_answers, challenge_latencies = _generate_answers(generator, challenge_payloads)
    holdout_llm, holdout_rows = evaluate_llm_answers(
        dataset=holdout_dataset,
        query_payloads=holdout_payloads,
        answers=holdout_answers,
    )
    challenge_llm, challenge_rows = evaluate_llm_answers(
        dataset=challenge_dataset,
        query_payloads=challenge_payloads,
        answers=challenge_answers,
    )

    retrieval_score = _retrieval_subscore(holdout_retrieval)
    llm_score = _llm_subscore(holdout_llm, mean(holdout_latencies) if holdout_latencies else 0.0)
    challenge_score = 0.5 * _retrieval_subscore(challenge_retrieval) + 0.5 * _llm_subscore(
        challenge_llm,
        mean(challenge_latencies) if challenge_latencies else 0.0,
    )
    metrics = {
        "retrieval_score": retrieval_score,
        "llm_score": llm_score,
        "challenge_score": challenge_score,
        "storage_size_mb": float(mean([holdout_retrieval["storage_size_mb"], challenge_retrieval["storage_size_mb"]])),
        "holdout_chunk_recall_at_5": holdout_retrieval["chunk_recall_at_5"],
        "holdout_chunk_mrr": holdout_retrieval["chunk_mrr"],
        "holdout_chunk_ndcg_at_5": holdout_retrieval["chunk_ndcg_at_5"],
        "holdout_filter_success_rate": holdout_retrieval["filter_success_rate"],
        "holdout_query_latency_ms": holdout_retrieval["query_latency_ms"],
        "holdout_faithfulness": holdout_llm["faithfulness"],
        "holdout_correctness": holdout_llm["correctness"],
        "holdout_completeness": holdout_llm["completeness"],
        "holdout_context_precision": holdout_llm["context_precision"],
        "holdout_answer_latency_ms": float(mean(holdout_latencies)) if holdout_latencies else 0.0,
        "challenge_chunk_recall_at_5": challenge_retrieval["chunk_recall_at_5"],
        "challenge_faithfulness": challenge_llm["faithfulness"],
        "challenge_correctness": challenge_llm["correctness"],
        "challenge_answer_latency_ms": float(mean(challenge_latencies)) if challenge_latencies else 0.0,
    }
    artifacts = {
        "holdout_retrieval_samples.json": holdout_payloads[:20],
        "holdout_answer_samples.json": holdout_rows[:20],
        "challenge_retrieval_samples.json": challenge_payloads[:20],
        "challenge_answer_samples.json": challenge_rows[:20],
    }
    return {"metrics": metrics, "artifacts": artifacts}


def _generate_answers(
    generator: OllamaGenerator,
    payloads: list[dict[str, object]],
) -> tuple[dict[str, str], list[float]]:
    answers: dict[str, str] = {}
    latencies: list[float] = []
    for payload in payloads:
        question_id = payload.get("question_id")
        query = payload.get("query")
        context = payload.get("context")
        if not isinstance(question_id, str) or not isinstance(query, str) or not isinstance(context, str):
            continue
        start_time = time.perf_counter()
        try:
            answers[question_id] = generator.generate(query, context)
        except (OllamaConnectionError, OllamaRequestError) as exc:
            raise VectorBackendUnavailableError(f"Ollama generation failed: {exc}") from exc
        latencies.append((time.perf_counter() - start_time) * 1000)
    return answers, latencies


def _resolve_retrieval_mode(retrieval_name: str) -> str:
    if retrieval_name == "dense_only":
        return "dense_only"
    if retrieval_name == "bm25_only":
        return "bm25_only"
    return "hybrid"


def _retrieval_subscore(metrics: dict[str, float]) -> float:
    latency_score = 1.0 / (1.0 + max(0.0, metrics.get("query_latency_ms", 0.0)) / 1000)
    return (
        0.40 * metrics.get("chunk_recall_at_5", 0.0)
        + 0.25 * metrics.get("chunk_mrr", 0.0)
        + 0.20 * metrics.get("chunk_ndcg_at_5", 0.0)
        + 0.10 * metrics.get("filter_success_rate", 0.0)
        + 0.05 * latency_score
    )


def _llm_subscore(metrics: dict[str, float], latency_ms: float) -> float:
    latency_score = 1.0 / (1.0 + max(0.0, latency_ms) / 1000)
    return (
        0.35 * metrics.get("faithfulness", 0.0)
        + 0.25 * metrics.get("correctness", 0.0)
        + 0.20 * metrics.get("completeness", 0.0)
        + 0.10 * metrics.get("context_precision", 0.0)
        + 0.10 * latency_score
    )


def _build_leaderboard(results: list[StageCandidateResult]) -> list[dict[str, object]]:
    completed = [result for result in results if result.status == "completed"]
    storage_values = [result.metrics.get("storage_size_mb", 0.0) for result in completed]
    maximum = max(storage_values) if storage_values else 0.0
    minimum = min(storage_values) if storage_values else 0.0

    rows: list[dict[str, object]] = []
    for result in results:
        storage_size = result.metrics.get("storage_size_mb", 0.0)
        if maximum == minimum:
            runtime_footprint_score = 1.0
        else:
            runtime_footprint_score = 1.0 - ((storage_size - minimum) / (maximum - minimum))
        final_score = (
            0.40 * result.metrics.get("retrieval_score", 0.0)
            + 0.35 * result.metrics.get("llm_score", 0.0)
            + 0.15 * result.metrics.get("challenge_score", 0.0)
            + 0.10 * runtime_footprint_score
        )
        rows.append(
            {
                "candidate_name": result.candidate_name,
                "status": result.status,
                "final_score": final_score,
                "runtime_footprint_score": runtime_footprint_score,
                **result.metrics,
                "skip_reason": result.skip_reason,
            }
        )
    return append_stage_score(rows, score_key="final_score")


def _build_recommendations(leaderboard: list[dict[str, object]]) -> dict[str, object]:
    completed = [row for row in leaderboard if row.get("status") == "completed"]
    if not completed:
        return {}
    best_overall = completed[0]
    low_resource = sorted(
        completed,
        key=lambda row: (
            float(row.get("storage_size_mb", 0.0)),
            float(row.get("holdout_answer_latency_ms", 0.0)),
            -float(row.get("final_score", 0.0)),
        ),
    )[0]
    latency_focused = sorted(
        completed,
        key=lambda row: (
            float(row.get("holdout_answer_latency_ms", 0.0)),
            -float(row.get("final_score", 0.0)),
        ),
    )[0]
    return {
        "best_overall": best_overall,
        "best_low_resource": low_resource,
        "best_latency_focused": latency_focused,
    }


def _build_run_config(dataset: BenchmarkDataset, candidate_config: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "final",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "split": dataset.split,
        "candidate_config": candidate_config,
        "hardware": collect_hardware_info(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
