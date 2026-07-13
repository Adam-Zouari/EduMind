"""Stage 4 retrieval-strategy experiments for the English-only benchmark."""

from __future__ import annotations

import argparse
import logging

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
    RETRIEVAL_STRATEGY_CANDIDATES,
    RetrievalStackCandidate,
    VECTOR_BACKEND_CANDIDATES,
)
from experiments.mlflow.stage_utils import build_backend, build_chunk_records, build_embedder, slice_candidates
from experiments.mlflow.utils.metrics_logger import MLflowExperiment
from experiments.mlflow.vector_backends import VectorBackendUnavailableError

logger = logging.getLogger(__name__)

BASELINE_CHUNK_EMBEDDING = next(
    candidate for candidate in EMBEDDING_CANDIDATES if candidate.model_name == "BAAI/bge-base-en-v1.5"
)


def run_all_experiments(
    *,
    dataset_name: str = "student_benchmark",
    split: str | None = None,
    resume: bool = False,
    force: bool = False,
    stage_limit: int | None = None,
    top_n: int = 3,
    test_mode: bool = False,
) -> int:
    """Run the Stage 4 retrieval sweep."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if test_mode:
        dataset_name = "synthetic_regression"
        split = split or "default"
        stage_limit = stage_limit or 2

    dataset = load_benchmark_dataset(dataset_name, split=split)
    selected_chunkers, selected_embeddings, selected_backends = _load_stage_three_candidates(dataset)
    baseline_chunk_embedder = build_embedder(BASELINE_CHUNK_EMBEDDING)
    chunk_records_by_name = {
        candidate.name: build_chunk_records(dataset, candidate, embedder=baseline_chunk_embedder)
        for candidate in CHUNKING_CANDIDATES
        if candidate.name in selected_chunkers
    }
    embedding_candidates = [
        candidate for candidate in EMBEDDING_CANDIDATES if candidate.model_name in selected_embeddings
    ]
    backend_candidates = [
        candidate for candidate in VECTOR_BACKEND_CANDIDATES if candidate.name in selected_backends
    ]
    strategy_candidates = slice_candidates(RETRIEVAL_STRATEGY_CANDIDATES, stage_limit)
    results: list[StageCandidateResult] = []

    with MLflowExperiment(
        "retrieval_experiments",
        run_name=f"retrieval_{dataset.name}_{dataset.split}",
        tags={
            "stage": "retrieval",
            "dataset": dataset.name,
            "split": dataset.split,
            "language": "en",
        },
    ) as parent_run:
        parent_run.log_artifact(
            "stage_inputs.json",
            {
                "chunkers": selected_chunkers,
                "embeddings": selected_embeddings,
                "vector_backends": selected_backends,
            },
        )

        for strategy in strategy_candidates:
            candidate_config = {
                "strategy_name": strategy.name,
                "mode": strategy.mode,
                "bm25_alpha": strategy.bm25_alpha,
                "chunkers": selected_chunkers,
                "embeddings": selected_embeddings,
                "vector_backends": selected_backends,
            }
            if resume and not force:
                cached_result = load_cached_candidate_result(
                    stage="retrieval",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_config=candidate_config,
                )
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            with MLflowExperiment(
                "retrieval_experiments",
                run_name=strategy.name,
                nested=True,
                tags={"stage": "retrieval", "candidate": strategy.name},
            ) as child_run:
                child_run.log_params(
                    {
                        **candidate_config,
                        "dataset_name": dataset.name,
                        "dataset_version": dataset.version,
                        "split": dataset.split,
                        "seed": DEFAULT_RANDOM_SEED,
                        "top_k": DEFAULT_TOP_K,
                    }
                )
                child_run.log_artifact("run_config.json", _build_run_config(dataset, candidate_config))

                try:
                    stack_rows = _evaluate_strategy_candidate(
                        dataset=dataset,
                        strategy=strategy,
                        backend_candidates=backend_candidates,
                        embedding_candidates=embedding_candidates,
                        selected_chunkers=selected_chunkers,
                        chunk_records_by_name=chunk_records_by_name,
                    )
                except VectorBackendUnavailableError as exc:
                    skipped = StageCandidateResult(
                        stage="retrieval",
                        dataset_name=dataset.name,
                        dataset_version=dataset.version,
                        split=dataset.split,
                        candidate_name=strategy.name,
                        candidate_config=candidate_config,
                        status="skipped",
                        skip_reason=str(exc),
                    )
                    save_cached_candidate_result(skipped)
                    results.append(skipped)
                    child_run.log_artifact("skip_reason.txt", str(exc))
                    continue

                aggregate_metrics = _aggregate_strategy_rows(stack_rows)
                artifacts = {"stack_metrics.json": stack_rows}
                child_run.log_metrics(aggregate_metrics)
                for filename, content in artifacts.items():
                    child_run.log_artifact(filename, content)

                result = StageCandidateResult(
                    stage="retrieval",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_name=strategy.name,
                    candidate_config=candidate_config,
                    metrics=aggregate_metrics,
                    artifacts=artifacts,
                )
                save_cached_candidate_result(result)
                results.append(result)

        leaderboard, stack_leaderboard = _build_leaderboards(results)
        best_candidates = {
            "top_retrieval_stacks": stack_leaderboard[:top_n],
        }
        summary = build_stage_summary(
            stage="retrieval",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            notes=["Stage 4 ranks full retrieval stacks rather than retrieval modes in isolation."],
        )
        save_stage_outputs(
            stage="retrieval",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            best_candidates=best_candidates,
            summary_markdown=summary,
        )
        parent_run.log_artifact("candidate_results.json", stage_results_to_artifact_payload(results))
        parent_run.log_artifact("leaderboard.json", leaderboard)
        parent_run.log_artifact("stack_leaderboard.json", stack_leaderboard)
        parent_run.log_artifact("best_candidates.json", best_candidates)
        parent_run.log_artifact("stage_summary.md", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 4 CLI parser."""
    parser = argparse.ArgumentParser(description="Run English-only retrieval experiments.")
    parser.add_argument("--dataset", default="student_benchmark")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Stage 4 retrieval runner."""
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


def _load_stage_three_candidates(
    dataset: BenchmarkDataset,
) -> tuple[list[str], list[str], list[str]]:
    embedding_stage = load_stage_best_candidates(
        stage="embedding",
        dataset_name=dataset.name,
        split=dataset.split,
    ) or {}
    vectordb_stage = load_stage_best_candidates(
        stage="vectordb",
        dataset_name=dataset.name,
        split=dataset.split,
    ) or {}

    chunkers = [str(value) for value in embedding_stage.get("top_chunkers", [])]
    embeddings = [str(value) for value in embedding_stage.get("top_embeddings", [])]
    backends = [str(value) for value in vectordb_stage.get("top_vector_backends", [])]
    if not chunkers:
        chunkers = [candidate.name for candidate in CHUNKING_CANDIDATES[:2]]
    if not embeddings:
        embeddings = [candidate.model_name for candidate in EMBEDDING_CANDIDATES[:2]]
    if not backends:
        backends = [candidate.name for candidate in VECTOR_BACKEND_CANDIDATES[:1]]
    return chunkers[:2], embeddings[:2], backends[:2]


def _evaluate_strategy_candidate(
    *,
    dataset: BenchmarkDataset,
    strategy,
    backend_candidates,
    embedding_candidates,
    selected_chunkers: list[str],
    chunk_records_by_name,
) -> list[dict[str, object]]:
    from experiments.mlflow.stage_utils import evaluate_retrieval_stack

    stack_rows: list[dict[str, object]] = []
    for backend_spec in backend_candidates:
        for chunker_name in selected_chunkers:
            chunk_records = chunk_records_by_name[chunker_name]
            for embedding_candidate in embedding_candidates:
                embedder = build_embedder(embedding_candidate)
                backend = build_backend(
                    backend_spec=backend_spec,
                    stage="retrieval",
                    dataset_name=dataset.name,
                    split=dataset.split,
                    candidate_suffix=f"{strategy.name}_{backend_spec.name}_{chunker_name}_{embedding_candidate.name}",
                    bm25_alpha=strategy.bm25_alpha,
                )
                metrics, query_results = evaluate_retrieval_stack(
                    dataset=dataset,
                    chunk_records=chunk_records,
                    embedder=embedder,
                    backend=backend,
                    retrieval_mode=strategy.mode,
                    bm25_alpha=strategy.bm25_alpha,
                    top_k=DEFAULT_TOP_K,
                    include_filters=True,
                )
                backend.reset()
                stack_rows.append(
                    {
                        "chunker_name": chunker_name,
                        "embedding_model": embedding_candidate.model_name,
                        "vector_backend": backend_spec.name,
                        "retrieval_name": strategy.name,
                        "bm25_alpha": strategy.bm25_alpha,
                        "query_results": query_results[:25],
                        **metrics,
                    }
                )
    if not stack_rows:
        raise VectorBackendUnavailableError("No retrieval stacks completed successfully.")
    return stack_rows


def _aggregate_strategy_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "chunk_recall_at_5": sum(float(row["chunk_recall_at_5"]) for row in rows) / len(rows),
        "chunk_mrr": sum(float(row["chunk_mrr"]) for row in rows) / len(rows),
        "chunk_ndcg_at_5": sum(float(row["chunk_ndcg_at_5"]) for row in rows) / len(rows),
        "source_recall_at_5": sum(float(row["source_recall_at_5"]) for row in rows) / len(rows),
        "source_hit_rate_at_5": sum(float(row["source_hit_rate_at_5"]) for row in rows) / len(rows),
        "query_latency_ms": sum(float(row["query_latency_ms"]) for row in rows) / len(rows),
        "filter_success_rate": sum(float(row["filter_success_rate"]) for row in rows) / len(rows),
        "num_stacks": float(len(rows)),
    }


def _build_leaderboards(
    results: list[StageCandidateResult],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed = [result for result in results if result.status == "completed"]
    latency_lookup = _normalized_scores(completed, key="query_latency_ms", lower_is_better=True)
    leaderboard_rows: list[dict[str, object]] = []
    stack_rows: list[dict[str, object]] = []
    for result in results:
        latency_score = latency_lookup.get(result.candidate_name, 0.0)
        stage_score = (
            0.40 * result.metrics.get("chunk_recall_at_5", 0.0)
            + 0.25 * result.metrics.get("chunk_mrr", 0.0)
            + 0.20 * result.metrics.get("chunk_ndcg_at_5", 0.0)
            + 0.10 * result.metrics.get("filter_success_rate", 0.0)
            + 0.05 * latency_score
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
            if isinstance(row, dict):
                stack_rows.append(dict(row))
    stack_leaderboard = _rank_stack_rows(stack_rows)
    return append_stage_score(leaderboard_rows, score_key="stage_score"), stack_leaderboard


def _rank_stack_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    latencies = [float(row.get("query_latency_ms", 0.0)) for row in rows]
    maximum = max(latencies)
    minimum = min(latencies)
    ranked: list[dict[str, object]] = []
    for row in rows:
        latency_value = float(row.get("query_latency_ms", 0.0))
        latency_score = 1.0 if maximum == minimum else 1.0 - ((latency_value - minimum) / (maximum - minimum))
        stage_score = (
            0.40 * float(row.get("chunk_recall_at_5", 0.0))
            + 0.25 * float(row.get("chunk_mrr", 0.0))
            + 0.20 * float(row.get("chunk_ndcg_at_5", 0.0))
            + 0.10 * float(row.get("filter_success_rate", 0.0))
            + 0.05 * latency_score
        )
        ranked.append(
            {
                **row,
                "candidate_name": RetrievalStackCandidate(
                    chunker_name=str(row["chunker_name"]),
                    embedding_model=str(row["embedding_model"]),
                    vector_backend=str(row["vector_backend"]),
                    retrieval_name=str(row["retrieval_name"]),
                    bm25_alpha=float(row["bm25_alpha"]),
                ).name,
                "stage_score": stage_score,
                "latency_score": latency_score,
            }
        )
    return append_stage_score(ranked, score_key="stage_score")


def _normalized_scores(
    results: list[StageCandidateResult],
    *,
    key: str,
    lower_is_better: bool,
) -> dict[str, float]:
    if not results:
        return {}
    values = [result.metrics.get(key, 0.0) for result in results]
    maximum = max(values)
    minimum = min(values)
    if maximum == minimum:
        return {result.candidate_name: 1.0 for result in results}
    if lower_is_better:
        return {
            result.candidate_name: 1.0 - ((result.metrics.get(key, 0.0) - minimum) / (maximum - minimum))
            for result in results
        }
    return {
        result.candidate_name: (result.metrics.get(key, 0.0) - minimum) / (maximum - minimum)
        for result in results
    }


def _build_run_config(dataset: BenchmarkDataset, candidate_config: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "retrieval",
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
