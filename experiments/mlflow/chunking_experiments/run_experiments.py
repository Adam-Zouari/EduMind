"""Stage 1 chunking experiments for the English-only benchmark."""

from __future__ import annotations

import argparse
import logging
import time
from statistics import mean

from experiments.mlflow.benchmark import BenchmarkDataset, load_benchmark_dataset, prepare_benchmark_dataset
from experiments.mlflow.harness import (
    StageCandidateResult,
    append_stage_score,
    build_stage_summary,
    collect_hardware_info,
    load_cached_candidate_result,
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
from experiments.mlflow.utils.metrics_logger import MLflowExperiment

logger = logging.getLogger(__name__)

BASELINE_EMBEDDING = next(
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
    """Run the Stage 1 chunking sweep on the selected benchmark split."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if test_mode:
        dataset_name = "synthetic_regression"
        split = split or "default"
        stage_limit = stage_limit or 2

    dataset = load_benchmark_dataset(dataset_name, split=split)
    preparation_report = prepare_benchmark_dataset(dataset.name, split=dataset.split)
    baseline_embedder = build_embedder(BASELINE_EMBEDDING)
    candidates = slice_candidates(CHUNKING_CANDIDATES, stage_limit)
    results: list[StageCandidateResult] = []

    with MLflowExperiment(
        "chunking_experiments",
        run_name=f"chunking_{dataset.name}_{dataset.split}",
        tags={
            "stage": "chunking",
            "dataset": dataset.name,
            "split": dataset.split,
            "language": "en",
        },
    ) as parent_run:
        parent_run.log_artifact("dataset_preparation.json", preparation_report)

        for candidate in candidates:
            candidate_config = {
                "name": candidate.name,
                "description": candidate.description,
                "kind": candidate.kind,
                "chunk_size": candidate.chunk_size,
                "chunk_overlap": candidate.chunk_overlap,
                "child_size": candidate.child_size or 0,
            }
            if resume and not force:
                cached_result = load_cached_candidate_result(
                    stage="chunking",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_config=candidate_config,
                )
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            with MLflowExperiment(
                "chunking_experiments",
                run_name=candidate.name,
                nested=True,
                tags={"stage": "chunking", "candidate": candidate.name},
            ) as child_run:
                child_run.log_params(
                    {
                        **candidate_config,
                        "dataset_name": dataset.name,
                        "dataset_version": dataset.version,
                        "split": dataset.split,
                        "seed": DEFAULT_RANDOM_SEED,
                        "top_k": DEFAULT_TOP_K,
                        "baseline_embedding": BASELINE_EMBEDDING.model_name,
                    }
                )
                child_run.log_artifact("run_config.json", _build_run_config(dataset, candidate_config))

                total_start = time.perf_counter()
                chunk_records = build_chunk_records(dataset, candidate, embedder=baseline_embedder)
                backend = build_backend(
                    backend_spec=_baseline_backend(),
                    stage="chunking",
                    dataset_name=dataset.name,
                    split=dataset.split,
                    candidate_suffix=candidate.name,
                    bm25_alpha=0.0,
                )
                try:
                    metrics, per_query_results = _evaluate_candidate(
                        dataset=dataset,
                        chunk_records=chunk_records,
                        baseline_embedder=baseline_embedder,
                        candidate_name=candidate.name,
                        backend=backend,
                    )
                finally:
                    backend.reset()

                total_elapsed = time.perf_counter() - total_start
                chunk_lengths = [len(chunk.text.split()) for chunk in chunk_records]
                metrics.update(
                    {
                        "avg_chunks_per_source": _safe_divide(
                            len(chunk_records),
                            len(dataset.snapshots),
                        ),
                        "avg_chunk_tokens": float(mean(chunk_lengths)) if chunk_lengths else 0.0,
                        "ingest_time_per_source_sec": _safe_divide(
                            total_elapsed,
                            len(dataset.snapshots),
                        ),
                        "total_runtime_sec": total_elapsed,
                    }
                )
                artifacts = {
                    "query_results.json": per_query_results,
                    "sample_chunks.json": [
                        {
                            "chunk_id": chunk.id,
                            "source_id": chunk.source_id,
                            "text": chunk.text[:240],
                            "metadata": chunk.metadata,
                        }
                        for chunk in chunk_records[:10]
                    ],
                }
                child_run.log_metrics(metrics)
                for filename, content in artifacts.items():
                    child_run.log_artifact(filename, content)

                result = StageCandidateResult(
                    stage="chunking",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_name=candidate.name,
                    candidate_config=candidate_config,
                    metrics=metrics,
                    artifacts=artifacts,
                    notes=[],
                )
                save_cached_candidate_result(result)
                results.append(result)

        leaderboard = _build_leaderboard(results)
        best_candidates = {
            "top_chunkers": [
                row["candidate_name"]
                for row in leaderboard
                if row.get("status") == "completed"
            ][:top_n]
        }
        summary = build_stage_summary(
            stage="chunking",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            notes=["English-only Stage 1 baseline uses BAAI/bge-base-en-v1.5 and Chroma dense retrieval."],
        )
        save_stage_outputs(
            stage="chunking",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            best_candidates=best_candidates,
            summary_markdown=summary,
        )
        parent_run.log_artifact("candidate_results.json", stage_results_to_artifact_payload(results))
        parent_run.log_artifact("leaderboard.json", leaderboard)
        parent_run.log_artifact("best_candidates.json", best_candidates)
        parent_run.log_artifact("stage_summary.md", summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Stage 1 CLI parser."""
    parser = argparse.ArgumentParser(description="Run English-only chunking experiments.")
    parser.add_argument("--dataset", default="student_benchmark")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Stage 1 chunking runner."""
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


def _evaluate_candidate(
    *,
    dataset: BenchmarkDataset,
    chunk_records,
    baseline_embedder,
    candidate_name: str,
    backend,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    from experiments.mlflow.stage_utils import evaluate_retrieval_stack

    logger.info("Evaluating chunker %s on %s/%s", candidate_name, dataset.name, dataset.split)
    metrics, per_query_results = evaluate_retrieval_stack(
        dataset=dataset,
        chunk_records=chunk_records,
        embedder=baseline_embedder,
        backend=backend,
        retrieval_mode="dense_only",
        bm25_alpha=0.0,
        top_k=DEFAULT_TOP_K,
        include_filters=False,
    )
    return metrics, per_query_results


def _build_leaderboard(results: list[StageCandidateResult]) -> list[dict[str, object]]:
    completed = [result for result in results if result.status == "completed"]
    latency_lookup = _normalized_latency_scores(completed)
    rows = []
    for result in results:
        latency_score = latency_lookup.get(result.candidate_name, 0.0)
        stage_score = (
            0.40 * result.metrics.get("chunk_recall_at_5", 0.0)
            + 0.25 * result.metrics.get("chunk_mrr", 0.0)
            + 0.20 * result.metrics.get("source_recall_at_5", 0.0)
            + 0.15 * latency_score
        )
        rows.append(
            {
                "candidate_name": result.candidate_name,
                "status": result.status,
                "stage_score": stage_score,
                **result.metrics,
                "latency_score": latency_score,
            }
        )
    return append_stage_score(rows, score_key="stage_score")


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


def _build_run_config(dataset: BenchmarkDataset, candidate_config: dict[str, object]) -> dict[str, object]:
    return {
        "stage": "chunking",
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "split": dataset.split,
        "seed": DEFAULT_RANDOM_SEED,
        "top_k": DEFAULT_TOP_K,
        "candidate_config": candidate_config,
        "hardware": collect_hardware_info(),
    }


def _safe_divide(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    raise SystemExit(main())
