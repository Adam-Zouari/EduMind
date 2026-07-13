"""Stage 2 embedding experiments for the English-only benchmark."""

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
)
from experiments.mlflow.stage_utils import build_backend, build_chunk_records, build_embedder, slice_candidates
from experiments.mlflow.utils.gpu_utils import get_gpu_memory_usage
from experiments.mlflow.utils.metrics_logger import MLflowExperiment

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
    top_n: int = 2,
    test_mode: bool = False,
) -> int:
    """Run the Stage 2 embedding sweep."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if test_mode:
        dataset_name = "synthetic_regression"
        split = split or "default"
        stage_limit = stage_limit or 2

    dataset = load_benchmark_dataset(dataset_name, split=split)
    selected_chunkers = _load_stage_one_chunkers(dataset, default_count=3)
    baseline_chunk_embedder = build_embedder(BASELINE_CHUNK_EMBEDDING)
    chunk_records_by_name = {
        candidate.name: build_chunk_records(dataset, candidate, embedder=baseline_chunk_embedder)
        for candidate in CHUNKING_CANDIDATES
        if candidate.name in selected_chunkers
    }
    candidates = slice_candidates(EMBEDDING_CANDIDATES, stage_limit)
    results: list[StageCandidateResult] = []

    with MLflowExperiment(
        "embedding_experiments",
        run_name=f"embedding_{dataset.name}_{dataset.split}",
        tags={
            "stage": "embedding",
            "dataset": dataset.name,
            "split": dataset.split,
            "language": "en",
        },
    ) as parent_run:
        parent_run.log_artifact(
            "chunker_inputs.json",
            {"selected_chunkers": selected_chunkers, "baseline_chunk_embedding": BASELINE_CHUNK_EMBEDDING.model_name},
        )

        for candidate in candidates:
            candidate_config = {
                "model_name": candidate.model_name,
                "embedding_dim": candidate.embedding_dim,
                "description": candidate.description,
                "chunkers": selected_chunkers,
            }
            if resume and not force:
                cached_result = load_cached_candidate_result(
                    stage="embedding",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_config=candidate_config,
                )
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            with MLflowExperiment(
                "embedding_experiments",
                run_name=candidate.name,
                nested=True,
                tags={"stage": "embedding", "candidate": candidate.name},
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

                embedder = build_embedder(candidate)
                gpu_metrics = get_gpu_memory_usage()
                pair_rows: list[dict[str, object]] = []
                per_chunker_queries: dict[str, object] = {}
                total_elapsed = 0.0
                total_chunks = 0

                for chunker_name in selected_chunkers:
                    chunk_records = chunk_records_by_name[chunker_name]
                    total_chunks += len(chunk_records)
                    backend = build_backend(
                        backend_spec=_baseline_backend(),
                        stage="embedding",
                        dataset_name=dataset.name,
                        split=dataset.split,
                        candidate_suffix=f"{candidate.name}_{chunker_name}",
                        bm25_alpha=0.0,
                    )
                    start_time = time.perf_counter()
                    try:
                        from experiments.mlflow.stage_utils import evaluate_retrieval_stack

                        metrics, per_query_results = evaluate_retrieval_stack(
                            dataset=dataset,
                            chunk_records=chunk_records,
                            embedder=embedder,
                            backend=backend,
                            retrieval_mode="dense_only",
                            bm25_alpha=0.0,
                            top_k=DEFAULT_TOP_K,
                            include_filters=False,
                        )
                    finally:
                        backend.reset()
                    elapsed = time.perf_counter() - start_time
                    total_elapsed += elapsed
                    pair_rows.append(
                        {
                            "chunker_name": chunker_name,
                            "embedding_model": candidate.model_name,
                            **metrics,
                            "embedding_stage_elapsed_sec": elapsed,
                        }
                    )
                    per_chunker_queries[chunker_name] = per_query_results[:25]

                aggregate_metrics = _aggregate_embedding_metrics(
                    pair_rows,
                    total_elapsed=total_elapsed,
                    total_chunks=total_chunks,
                    gpu_metrics=gpu_metrics,
                )
                artifacts = {
                    "pair_metrics.json": pair_rows,
                    "query_samples.json": per_chunker_queries,
                }
                child_run.log_metrics(aggregate_metrics)
                for filename, content in artifacts.items():
                    child_run.log_artifact(filename, content)

                result = StageCandidateResult(
                    stage="embedding",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_name=candidate.name,
                    candidate_config=candidate_config,
                    metrics=aggregate_metrics,
                    artifacts=artifacts,
                )
                save_cached_candidate_result(result)
                results.append(result)

        leaderboard, pair_leaderboard = _build_leaderboards(results)
        best_candidates = {
            "top_embeddings": [
                row["candidate_name"]
                for row in leaderboard
                if row.get("status") == "completed"
            ][:top_n],
            "top_chunkers": _top_chunkers_from_pairs(pair_leaderboard, count=top_n),
        }
        summary = build_stage_summary(
            stage="embedding",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            notes=["Stage 2 keeps chunking fixed and varies only the embedding model."],
        )
        save_stage_outputs(
            stage="embedding",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            best_candidates=best_candidates,
            summary_markdown=summary,
        )
        parent_run.log_artifact("candidate_results.json", stage_results_to_artifact_payload(results))
        parent_run.log_artifact("leaderboard.json", leaderboard)
        parent_run.log_artifact("pair_leaderboard.json", pair_leaderboard)
        parent_run.log_artifact("best_candidates.json", best_candidates)
        parent_run.log_artifact("stage_summary.md", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 2 CLI parser."""
    parser = argparse.ArgumentParser(description="Run English-only embedding experiments.")
    parser.add_argument("--dataset", default="student_benchmark")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Stage 2 embedding runner."""
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


def _baseline_backend():
    from experiments.mlflow.stage_specs import VECTOR_BACKEND_CANDIDATES

    return next(candidate for candidate in VECTOR_BACKEND_CANDIDATES if candidate.name == "chroma")


def _load_stage_one_chunkers(dataset: BenchmarkDataset, default_count: int) -> list[str]:
    previous = load_stage_best_candidates(
        stage="chunking",
        dataset_name=dataset.name,
        split=dataset.split,
    )
    if previous and isinstance(previous.get("top_chunkers"), list):
        values = [str(value) for value in previous["top_chunkers"]]
        if values:
            return values[:default_count]
    return [candidate.name for candidate in CHUNKING_CANDIDATES[:default_count]]


def _aggregate_embedding_metrics(
    pair_rows: list[dict[str, object]],
    *,
    total_elapsed: float,
    total_chunks: int,
    gpu_metrics: dict[str, float],
) -> dict[str, float]:
    if not pair_rows:
        return {}
    return {
        "chunk_recall_at_5": float(mean(float(row["chunk_recall_at_5"]) for row in pair_rows)),
        "chunk_mrr": float(mean(float(row["chunk_mrr"]) for row in pair_rows)),
        "chunk_ndcg_at_5": float(mean(float(row["chunk_ndcg_at_5"]) for row in pair_rows)),
        "source_recall_at_5": float(mean(float(row["source_recall_at_5"]) for row in pair_rows)),
        "query_latency_ms": float(mean(float(row["query_latency_ms"]) for row in pair_rows)),
        "embedding_latency_sec": total_elapsed,
        "indexing_throughput_chunks_per_sec": float(total_chunks / total_elapsed) if total_elapsed > 0 else 0.0,
        "gpu_memory_mb": float(gpu_metrics.get("allocated_mb", 0.0)),
        "num_pairs": float(len(pair_rows)),
    }


def _build_leaderboards(
    results: list[StageCandidateResult],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed = [result for result in results if result.status == "completed"]
    latency_lookup = _normalized_latency_scores(completed)

    leaderboard_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for result in results:
        latency_score = latency_lookup.get(result.candidate_name, 0.0)
        stage_score = (
            0.45 * result.metrics.get("chunk_recall_at_5", 0.0)
            + 0.25 * result.metrics.get("chunk_mrr", 0.0)
            + 0.15 * result.metrics.get("chunk_ndcg_at_5", 0.0)
            + 0.15 * latency_score
        )
        leaderboard_rows.append(
            {
                "candidate_name": result.candidate_name,
                "status": result.status,
                "stage_score": stage_score,
                "latency_score": latency_score,
                **result.metrics,
            }
        )
        for row in result.artifacts.get("pair_metrics.json", []):
            if not isinstance(row, dict):
                continue
            pair_rows.append(dict(row))

    pair_leaderboard = _rank_pair_rows(pair_rows)
    return append_stage_score(leaderboard_rows, score_key="stage_score"), pair_leaderboard


def _rank_pair_rows(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not pair_rows:
        return []
    latencies = [float(row.get("query_latency_ms", 0.0)) for row in pair_rows]
    maximum = max(latencies)
    minimum = min(latencies)
    ranked: list[dict[str, object]] = []
    for row in pair_rows:
        latency_value = float(row.get("query_latency_ms", 0.0))
        latency_score = 1.0 if maximum == minimum else 1.0 - ((latency_value - minimum) / (maximum - minimum))
        stage_score = (
            0.45 * float(row.get("chunk_recall_at_5", 0.0))
            + 0.25 * float(row.get("chunk_mrr", 0.0))
            + 0.15 * float(row.get("chunk_ndcg_at_5", 0.0))
            + 0.15 * latency_score
        )
        ranked.append({**row, "stage_score": stage_score, "latency_score": latency_score})
    return append_stage_score(ranked, score_key="stage_score")


def _normalized_latency_scores(results: list[StageCandidateResult]) -> dict[str, float]:
    if not results:
        return {}
    latencies = [result.metrics.get("query_latency_ms", 0.0) for result in results]
    maximum = max(latencies)
    minimum = min(latencies)
    if maximum == minimum:
        return {result.candidate_name: 1.0 for result in results}
    return {
        result.candidate_name: 1.0 - ((result.metrics.get("query_latency_ms", 0.0) - minimum) / (maximum - minimum))
        for result in results
    }


def _top_chunkers_from_pairs(pair_leaderboard: list[dict[str, object]], count: int) -> list[str]:
    aggregated: dict[str, list[float]] = {}
    for row in pair_leaderboard:
        chunker_name = row.get("chunker_name")
        if not isinstance(chunker_name, str):
            continue
        aggregated.setdefault(chunker_name, []).append(float(row.get("stage_score", 0.0)))
    ranked = sorted(
        (
            {"chunker_name": name, "score": float(mean(scores))}
            for name, scores in aggregated.items()
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return [item["chunker_name"] for item in ranked[:count]]


def _build_run_config(dataset: BenchmarkDataset, candidate_config: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "embedding",
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
