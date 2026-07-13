"""Stage 5 local-language-model experiments for the English-only benchmark."""

from __future__ import annotations

import argparse
import logging
import time
from statistics import mean

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
from experiments.mlflow.stage_specs import (
    CHUNKING_CANDIDATES,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TOP_K,
    EMBEDDING_CANDIDATES,
    FullStackCandidate,
    RetrievalStackCandidate,
    SLM_CANDIDATES,
    VECTOR_BACKEND_CANDIDATES,
)
from experiments.mlflow.stage_utils import (
    build_backend,
    build_chunk_records,
    build_context_from_hits,
    build_embedder,
    evaluate_llm_answers,
    slice_candidates,
)
from experiments.mlflow.utils.metrics_logger import MLflowExperiment
from experiments.mlflow.vector_backends import VectorBackendUnavailableError
from edumind.rag.errors import OllamaConnectionError, OllamaRequestError
from edumind.rag.llm_generator import OllamaGenerator
from edumind.rag.types import LLMSettings

logger = logging.getLogger(__name__)

BASELINE_CHUNK_EMBEDDING = next(
    candidate for candidate in EMBEDDING_CANDIDATES if candidate.model_name == "BAAI/bge-base-en-v1.5"
)
DEFAULT_LLM_QUESTION_LIMIT = 40


def run_all_experiments(
    *,
    dataset_name: str = "student_benchmark",
    split: str | None = None,
    resume: bool = False,
    force: bool = False,
    stage_limit: int | None = None,
    top_n: int = 5,
    test_mode: bool = False,
) -> int:
    """Run the Stage 5 SLM sweep."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if test_mode:
        dataset_name = "synthetic_regression"
        split = split or "default"
        stage_limit = stage_limit or 1

    dataset = load_benchmark_dataset(dataset_name, split=split)
    llm_dataset = _subset_questions(dataset, count=8 if test_mode else DEFAULT_LLM_QUESTION_LIMIT)
    retrieval_stacks = _load_stage_four_stacks(dataset, count=3)
    if not retrieval_stacks:
        logger.warning("No retrieval stacks available for LLM evaluation.")
        return 1

    chunk_candidates = {candidate.name: candidate for candidate in CHUNKING_CANDIDATES}
    embedding_candidates = {candidate.model_name: candidate for candidate in EMBEDDING_CANDIDATES}
    backend_candidates = {candidate.name: candidate for candidate in VECTOR_BACKEND_CANDIDATES}
    baseline_chunk_embedder = build_embedder(BASELINE_CHUNK_EMBEDDING)
    chunk_records_by_name = {
        name: build_chunk_records(llm_dataset, chunk_candidates[name], embedder=baseline_chunk_embedder)
        for name in {stack.chunker_name for stack in retrieval_stacks}
    }
    retrieval_payloads = _prepare_retrieval_payloads(
        dataset=llm_dataset,
        retrieval_stacks=retrieval_stacks,
        chunk_records_by_name=chunk_records_by_name,
        embedding_candidates=embedding_candidates,
        backend_candidates=backend_candidates,
    )
    llm_candidates = slice_candidates(SLM_CANDIDATES, stage_limit)
    results: list[StageCandidateResult] = []

    with MLflowExperiment(
        "llm_experiments",
        run_name=f"llm_{dataset.name}_{dataset.split}",
        tags={
            "stage": "llm",
            "dataset": dataset.name,
            "split": dataset.split,
            "language": "en",
        },
    ) as parent_run:
        parent_run.log_artifact(
            "stage_inputs.json",
            {
                "retrieval_stacks": [stack.to_dict() for stack in retrieval_stacks],
                "llm_question_limit": len(llm_dataset.questions),
            },
        )

        for llm_candidate in llm_candidates:
            candidate_config = {
                "model_name": llm_candidate.model_name,
                "retrieval_stacks": [stack.to_dict() for stack in retrieval_stacks],
                "llm_question_limit": len(llm_dataset.questions),
            }
            if resume and not force:
                cached_result = load_cached_candidate_result(
                    stage="llm",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_config=candidate_config,
                )
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            with MLflowExperiment(
                "llm_experiments",
                run_name=llm_candidate.name,
                nested=True,
                tags={"stage": "llm", "candidate": llm_candidate.name},
            ) as child_run:
                child_run.log_params(
                    {
                        "model_name": llm_candidate.model_name,
                        "dataset_name": dataset.name,
                        "dataset_version": dataset.version,
                        "split": dataset.split,
                        "seed": DEFAULT_RANDOM_SEED,
                        "top_k": DEFAULT_TOP_K,
                        "num_questions": len(llm_dataset.questions),
                    }
                )
                child_run.log_artifact("run_config.json", _build_run_config(dataset, candidate_config))

                generator = OllamaGenerator(
                    settings=LLMSettings(
                        model_name=llm_candidate.model_name,
                        base_url="http://localhost:11434",
                        temperature=0.3,
                        max_tokens=256,
                        request_timeout=120,
                    )
                )
                if not generator.health_check():
                    skipped = StageCandidateResult(
                        stage="llm",
                        dataset_name=dataset.name,
                        dataset_version=dataset.version,
                        split=dataset.split,
                        candidate_name=llm_candidate.name,
                        candidate_config=candidate_config,
                        status="skipped",
                        skip_reason="Ollama service is unavailable.",
                    )
                    save_cached_candidate_result(skipped)
                    results.append(skipped)
                    child_run.log_artifact("skip_reason.txt", skipped.skip_reason or "Unavailable")
                    continue

                available_models = set(generator.list_models())
                if llm_candidate.model_name not in available_models:
                    skipped = StageCandidateResult(
                        stage="llm",
                        dataset_name=dataset.name,
                        dataset_version=dataset.version,
                        split=dataset.split,
                        candidate_name=llm_candidate.name,
                        candidate_config=candidate_config,
                        status="skipped",
                        skip_reason=f"Model '{llm_candidate.model_name}' is not installed in Ollama.",
                    )
                    save_cached_candidate_result(skipped)
                    results.append(skipped)
                    child_run.log_artifact("skip_reason.txt", skipped.skip_reason or "Unavailable")
                    continue

                stack_rows, aggregate_metrics, sample_rows = _evaluate_llm_candidate(
                    llm_dataset=llm_dataset,
                    retrieval_stacks=retrieval_stacks,
                    retrieval_payloads=retrieval_payloads,
                    generator=generator,
                )
                artifacts = {
                    "stack_metrics.json": stack_rows,
                    "manual_review.json": sample_rows,
                }
                child_run.log_metrics(aggregate_metrics)
                for filename, content in artifacts.items():
                    child_run.log_artifact(filename, content)

                result = StageCandidateResult(
                    stage="llm",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_name=llm_candidate.name,
                    candidate_config=candidate_config,
                    metrics=aggregate_metrics,
                    artifacts=artifacts,
                )
                save_cached_candidate_result(result)
                results.append(result)

        leaderboard, full_stack_leaderboard = _build_leaderboards(results)
        best_candidates = {"top_full_stacks": full_stack_leaderboard[:top_n]}
        summary = build_stage_summary(
            stage="llm",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            notes=["The LLM stage evaluates a capped English question subset to keep local runs tractable."],
        )
        save_stage_outputs(
            stage="llm",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            best_candidates=best_candidates,
            summary_markdown=summary,
        )
        parent_run.log_artifact("candidate_results.json", stage_results_to_artifact_payload(results))
        parent_run.log_artifact("leaderboard.json", leaderboard)
        parent_run.log_artifact("full_stack_leaderboard.json", full_stack_leaderboard)
        parent_run.log_artifact("best_candidates.json", best_candidates)
        parent_run.log_artifact("stage_summary.md", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 5 CLI parser."""
    parser = argparse.ArgumentParser(description="Run English-only LLM experiments.")
    parser.add_argument("--dataset", default="student_benchmark")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Stage 5 LLM runner."""
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


def _subset_questions(dataset: BenchmarkDataset, *, count: int) -> BenchmarkDataset:
    return BenchmarkDataset(
        name=dataset.name,
        version=dataset.version,
        split=dataset.split,
        assets=dataset.assets,
        snapshots=dataset.snapshots,
        questions=dataset.questions[:count],
        metadata=dict(dataset.metadata),
    )


def _load_stage_four_stacks(dataset: BenchmarkDataset, count: int) -> list[RetrievalStackCandidate]:
    previous = load_stage_best_candidates(
        stage="retrieval",
        dataset_name=dataset.name,
        split=dataset.split,
    )
    if previous is None:
        return []
    raw_stacks = previous.get("top_retrieval_stacks", [])
    if not isinstance(raw_stacks, list):
        return []

    stacks: list[RetrievalStackCandidate] = []
    for row in raw_stacks[:count]:
        if not isinstance(row, dict):
            continue
        stacks.append(
            RetrievalStackCandidate(
                chunker_name=str(row.get("chunker_name", "")),
                embedding_model=str(row.get("embedding_model", "")),
                vector_backend=str(row.get("vector_backend", "")),
                retrieval_name=str(row.get("retrieval_name", "")),
                bm25_alpha=float(row.get("bm25_alpha", 0.0)),
            )
        )
    return stacks


def _prepare_retrieval_payloads(
    *,
    dataset: BenchmarkDataset,
    retrieval_stacks: list[RetrievalStackCandidate],
    chunk_records_by_name,
    embedding_candidates,
    backend_candidates,
) -> dict[str, list[dict[str, object]]]:
    from experiments.mlflow.stage_utils import evaluate_retrieval_stack

    payloads: dict[str, list[dict[str, object]]] = {}
    for stack in retrieval_stacks:
        embedding_candidate = embedding_candidates.get(stack.embedding_model)
        backend_candidate = backend_candidates.get(stack.vector_backend)
        if embedding_candidate is None or backend_candidate is None:
            continue

        chunk_records = chunk_records_by_name.get(stack.chunker_name)
        if chunk_records is None:
            continue
        embedder = build_embedder(embedding_candidate)
        backend = build_backend(
            backend_spec=backend_candidate,
            stage="llm",
            dataset_name=dataset.name,
            split=dataset.split,
            candidate_suffix=stack.name,
            bm25_alpha=stack.bm25_alpha,
        )
        mode = "hybrid"
        if stack.retrieval_name == "dense_only":
            mode = "dense_only"
        elif stack.retrieval_name == "bm25_only":
            mode = "bm25_only"
        metrics, query_results = evaluate_retrieval_stack(
            dataset=dataset,
            chunk_records=chunk_records,
            embedder=embedder,
            backend=backend,
            retrieval_mode=mode,
            bm25_alpha=stack.bm25_alpha,
            top_k=DEFAULT_TOP_K,
            include_filters=True,
        )
        logger.info("Prepared retrieval payloads for %s with recall@5 %.4f", stack.name, metrics["chunk_recall_at_5"])
        backend.reset()
        payloads[stack.name] = query_results
    return payloads


def _evaluate_llm_candidate(
    *,
    llm_dataset: BenchmarkDataset,
    retrieval_stacks: list[RetrievalStackCandidate],
    retrieval_payloads: dict[str, list[dict[str, object]]],
    generator: OllamaGenerator,
) -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]]]:
    stack_rows: list[dict[str, object]] = []
    manual_review_rows: list[dict[str, object]] = []

    for stack in retrieval_stacks:
        payloads = retrieval_payloads.get(stack.name, [])
        answers: dict[str, str] = {}
        latencies_ms: list[float] = []
        throughput_scores: list[float] = []
        for payload in payloads:
            question_id = payload.get("question_id")
            query_text = payload.get("query")
            context = payload.get("context")
            if not isinstance(question_id, str) or not isinstance(query_text, str) or not isinstance(context, str):
                continue

            start_time = time.perf_counter()
            try:
                answer = generator.generate(query_text, context)
            except (OllamaConnectionError, OllamaRequestError) as exc:
                raise VectorBackendUnavailableError(f"Ollama generation failed: {exc}") from exc
            latency_ms = (time.perf_counter() - start_time) * 1000
            answers[question_id] = answer
            latencies_ms.append(latency_ms)
            token_count = max(1, len(answer.split()))
            throughput_scores.append(token_count / (latency_ms / 1000) if latency_ms > 0 else 0.0)

        answer_metrics, per_question_rows = evaluate_llm_answers(
            dataset=llm_dataset,
            query_payloads=payloads,
            answers=answers,
        )
        stack_rows.append(
            {
                "chunker_name": stack.chunker_name,
                "embedding_model": stack.embedding_model,
                "vector_backend": stack.vector_backend,
                "retrieval_name": stack.retrieval_name,
                "bm25_alpha": stack.bm25_alpha,
                **answer_metrics,
                "latency_ms": float(mean(latencies_ms)) if latencies_ms else 0.0,
                "tokens_per_sec": float(mean(throughput_scores)) if throughput_scores else 0.0,
            }
        )
        for row in per_question_rows[:20]:
            manual_review_rows.append(
                {
                    "stack_name": stack.name,
                    **row,
                }
            )

    aggregate_metrics = {
        "faithfulness": float(mean(float(row["faithfulness"]) for row in stack_rows)) if stack_rows else 0.0,
        "correctness": float(mean(float(row["correctness"]) for row in stack_rows)) if stack_rows else 0.0,
        "completeness": float(mean(float(row["completeness"]) for row in stack_rows)) if stack_rows else 0.0,
        "context_precision": float(mean(float(row["context_precision"]) for row in stack_rows)) if stack_rows else 0.0,
        "latency_ms": float(mean(float(row["latency_ms"]) for row in stack_rows)) if stack_rows else 0.0,
        "tokens_per_sec": float(mean(float(row["tokens_per_sec"]) for row in stack_rows)) if stack_rows else 0.0,
        "num_stacks": float(len(stack_rows)),
    }
    return stack_rows, aggregate_metrics, manual_review_rows[:20]


def _build_leaderboards(
    results: list[StageCandidateResult],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed = [result for result in results if result.status == "completed"]
    latency_lookup = _normalized_scores(completed)
    leaderboard_rows: list[dict[str, object]] = []
    full_stack_rows: list[dict[str, object]] = []
    for result in results:
        latency_score = latency_lookup.get(result.candidate_name, 0.0)
        stage_score = (
            0.35 * result.metrics.get("faithfulness", 0.0)
            + 0.25 * result.metrics.get("correctness", 0.0)
            + 0.20 * result.metrics.get("completeness", 0.0)
            + 0.10 * result.metrics.get("context_precision", 0.0)
            + 0.10 * latency_score
        )
        leaderboard_rows.append(
            {
                "candidate_name": result.candidate_name,
                "status": result.status,
                "stage_score": stage_score,
                "latency_score": latency_score,
                **result.metrics,
                "skip_reason": result.skip_reason,
            }
        )
        for row in result.artifacts.get("stack_metrics.json", []):
            if not isinstance(row, dict):
                continue
            full_stack_rows.append(
                {
                    **row,
                    "llm_model": result.candidate_config.get("model_name", ""),
                }
            )

    ranked_full_stacks = _rank_full_stack_rows(full_stack_rows)
    return append_stage_score(leaderboard_rows, score_key="stage_score"), ranked_full_stacks


def _rank_full_stack_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows]
    maximum = max(latencies)
    minimum = min(latencies)
    ranked: list[dict[str, object]] = []
    for row in rows:
        latency_value = float(row.get("latency_ms", 0.0))
        latency_score = 1.0 if maximum == minimum else 1.0 - ((latency_value - minimum) / (maximum - minimum))
        stack = RetrievalStackCandidate(
            chunker_name=str(row.get("chunker_name", "")),
            embedding_model=str(row.get("embedding_model", "")),
            vector_backend=str(row.get("vector_backend", "")),
            retrieval_name=str(row.get("retrieval_name", "")),
            bm25_alpha=float(row.get("bm25_alpha", 0.0)),
        )
        full_stack = FullStackCandidate(
            retrieval_stack=stack,
            llm_model=str(row.get("llm_model", "")),
        )
        stage_score = (
            0.35 * float(row.get("faithfulness", 0.0))
            + 0.25 * float(row.get("correctness", 0.0))
            + 0.20 * float(row.get("completeness", 0.0))
            + 0.10 * float(row.get("context_precision", 0.0))
            + 0.10 * latency_score
        )
        ranked.append(
            {
                **row,
                "candidate_name": full_stack.name,
                "stage_score": stage_score,
                "latency_score": latency_score,
            }
        )
    return append_stage_score(ranked, score_key="stage_score")


def _normalized_scores(results: list[StageCandidateResult]) -> dict[str, float]:
    if not results:
        return {}
    values = [result.metrics.get("latency_ms", 0.0) for result in results]
    maximum = max(values)
    minimum = min(values)
    if maximum == minimum:
        return {result.candidate_name: 1.0 for result in results}
    return {
        result.candidate_name: 1.0 - ((result.metrics.get("latency_ms", 0.0) - minimum) / (maximum - minimum))
        for result in results
    }


def _build_run_config(dataset: BenchmarkDataset, candidate_config: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "llm",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "split": dataset.split,
        "seed": DEFAULT_RANDOM_SEED,
        "top_k": DEFAULT_TOP_K,
        "candidate_config": candidate_config,
        "hardware": collect_hardware_info(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
