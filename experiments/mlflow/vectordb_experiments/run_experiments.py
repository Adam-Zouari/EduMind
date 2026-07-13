"""Stage 3 vector-database experiments for the English-only benchmark."""

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
    top_n: int = 2,
    test_mode: bool = False,
) -> int:
    """Run the Stage 3 vector DB sweep."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if test_mode:
        dataset_name = "synthetic_regression"
        split = split or "default"
        stage_limit = stage_limit or 2

    dataset = load_benchmark_dataset(dataset_name, split=split)
    selected_chunkers, selected_embeddings = _load_stage_two_candidates(dataset)
    baseline_chunk_embedder = build_embedder(BASELINE_CHUNK_EMBEDDING)
    chunk_records_by_name = {
        candidate.name: build_chunk_records(dataset, candidate, embedder=baseline_chunk_embedder)
        for candidate in CHUNKING_CANDIDATES
        if candidate.name in selected_chunkers
    }
    embedding_candidates = [candidate for candidate in EMBEDDING_CANDIDATES if candidate.model_name in selected_embeddings]
    backend_candidates = slice_candidates(VECTOR_BACKEND_CANDIDATES, stage_limit)
    results: list[StageCandidateResult] = []

    with MLflowExperiment(
        "vectordb_experiments",
        run_name=f"vectordb_{dataset.name}_{dataset.split}",
        tags={
            "stage": "vectordb",
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
            },
        )

        for backend_spec in backend_candidates:
            candidate_config = {
                "backend_name": backend_spec.name,
                "chunkers": selected_chunkers,
                "embeddings": selected_embeddings,
            }
            if resume and not force:
                cached_result = load_cached_candidate_result(
                    stage="vectordb",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_config=candidate_config,
                )
                if cached_result is not None:
                    results.append(cached_result)
                    continue

            with MLflowExperiment(
                "vectordb_experiments",
                run_name=backend_spec.name,
                nested=True,
                tags={"stage": "vectordb", "candidate": backend_spec.name},
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
                    pair_rows, aggregate_metrics = _evaluate_backend_candidate(
                        dataset=dataset,
                        backend_name=backend_spec.name,
                        backend_spec=backend_spec,
                        selected_chunkers=selected_chunkers,
                        embedding_candidates=embedding_candidates,
                        chunk_records_by_name=chunk_records_by_name,
                    )
                except VectorBackendUnavailableError as exc:
                    skipped = StageCandidateResult(
                        stage="vectordb",
                        dataset_name=dataset.name,
                        dataset_version=dataset.version,
                        split=dataset.split,
                        candidate_name=backend_spec.name,
                        candidate_config=candidate_config,
                        status="skipped",
                        skip_reason=str(exc),
                    )
                    save_cached_candidate_result(skipped)
                    results.append(skipped)
                    child_run.log_artifact("skip_reason.txt", str(exc))
                    continue

                artifacts = {"pair_metrics.json": pair_rows}
                child_run.log_metrics(aggregate_metrics)
                for filename, content in artifacts.items():
                    child_run.log_artifact(filename, content)

                result = StageCandidateResult(
                    stage="vectordb",
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    split=dataset.split,
                    candidate_name=backend_spec.name,
                    candidate_config=candidate_config,
                    metrics=aggregate_metrics,
                    artifacts=artifacts,
                )
                save_cached_candidate_result(result)
                results.append(result)

        leaderboard, pair_leaderboard = _build_leaderboards(results)
        best_candidates = {
            "top_vector_backends": [
                row["candidate_name"]
                for row in leaderboard
                if row.get("status") == "completed"
            ][:top_n]
        }
        summary = build_stage_summary(
            stage="vectordb",
            dataset_name=dataset.name,
            split=dataset.split,
            leaderboard=leaderboard,
            notes=["Unavailable backends are recorded as skipped instead of failing the suite."],
        )
        save_stage_outputs(
            stage="vectordb",
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
    """Build the Stage 3 CLI parser."""
    parser = argparse.ArgumentParser(description="Run English-only vector DB experiments.")
    parser.add_argument("--dataset", default="student_benchmark")
    parser.add_argument("--split", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--test-mode", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Stage 3 vector DB runner."""
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


def _load_stage_two_candidates(dataset: BenchmarkDataset) -> tuple[list[str], list[str]]:
    previous = load_stage_best_candidates(
        stage="embedding",
        dataset_name=dataset.name,
        split=dataset.split,
    )
    if previous is None:
        return (
            [candidate.name for candidate in CHUNKING_CANDIDATES[:2]],
            [candidate.model_name for candidate in EMBEDDING_CANDIDATES[:2]],
        )

    chunkers = [str(value) for value in previous.get("top_chunkers", [])]
    embeddings = [str(value) for value in previous.get("top_embeddings", [])]
    if not chunkers:
        chunkers = [candidate.name for candidate in CHUNKING_CANDIDATES[:2]]
    if not embeddings:
        embeddings = [candidate.model_name for candidate in EMBEDDING_CANDIDATES[:2]]
    return chunkers[:2], embeddings[:2]


def _evaluate_backend_candidate(
    *,
    dataset: BenchmarkDataset,
    backend_name: str,
    backend_spec,
    selected_chunkers: list[str],
    embedding_candidates,
    chunk_records_by_name,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    from experiments.mlflow.stage_utils import evaluate_retrieval_stack

    pair_rows: list[dict[str, object]] = []
    total_elapsed = 0.0
    ingest_latencies: list[float] = []
    reset_latencies: list[float] = []
    for chunker_name in selected_chunkers:
        chunk_records = chunk_records_by_name[chunker_name]
        for embedding_candidate in embedding_candidates:
            embedder = build_embedder(embedding_candidate)
            backend = build_backend(
                backend_spec=backend_spec,
                stage="vectordb",
                dataset_name=dataset.name,
                split=dataset.split,
                candidate_suffix=f"{backend_name}_{chunker_name}_{embedding_candidate.name}",
                bm25_alpha=0.0,
            )
            start_time = time.perf_counter()
            metrics, _ = evaluate_retrieval_stack(
                dataset=dataset,
                chunk_records=chunk_records,
                embedder=embedder,
                backend=backend,
                retrieval_mode="dense_only",
                bm25_alpha=0.0,
                top_k=DEFAULT_TOP_K,
                include_filters=False,
            )
            elapsed = time.perf_counter() - start_time
            total_elapsed += elapsed
            ingest_latencies.append(elapsed)

            reset_start = time.perf_counter()
            backend.reset()
            reset_latencies.append((time.perf_counter() - reset_start) * 1000)
            pair_rows.append(
                {
                    "backend_name": backend_name,
                    "chunker_name": chunker_name,
                    "embedding_model": embedding_candidate.model_name,
                    **metrics,
                    "ingest_eval_elapsed_sec": elapsed,
                    "reset_time_ms": reset_latencies[-1],
                }
            )

    if not pair_rows:
        raise VectorBackendUnavailableError(f"No evaluation pairs were completed for backend '{backend_name}'.")

    aggregate_metrics = {
        "chunk_recall_at_5": float(mean(float(row["chunk_recall_at_5"]) for row in pair_rows)),
        "chunk_mrr": float(mean(float(row["chunk_mrr"]) for row in pair_rows)),
        "chunk_ndcg_at_5": float(mean(float(row["chunk_ndcg_at_5"]) for row in pair_rows)),
        "source_recall_at_5": float(mean(float(row["source_recall_at_5"]) for row in pair_rows)),
        "query_latency_ms": float(mean(float(row["query_latency_ms"]) for row in pair_rows)),
        "ingest_time_sec": float(mean(ingest_latencies)) if ingest_latencies else 0.0,
        "reset_time_ms": float(mean(reset_latencies)) if reset_latencies else 0.0,
        "storage_size_mb": float(mean(float(row["storage_size_mb"]) for row in pair_rows)),
        "num_pairs": float(len(pair_rows)),
        "total_runtime_sec": total_elapsed,
    }
    return pair_rows, aggregate_metrics


def _build_leaderboards(
    results: list[StageCandidateResult],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed = [result for result in results if result.status == "completed"]
    latency_lookup = _normalized_scores(completed, key="query_latency_ms", lower_is_better=True)
    ingest_lookup = _normalized_scores(completed, key="ingest_time_sec", lower_is_better=True)

    leaderboard_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for result in results:
        latency_score = latency_lookup.get(result.candidate_name, 0.0)
        ingest_score = ingest_lookup.get(result.candidate_name, 0.0)
        stage_score = (
            0.35 * result.metrics.get("chunk_recall_at_5", 0.0)
            + 0.20 * result.metrics.get("chunk_mrr", 0.0)
            + 0.15 * result.metrics.get("chunk_ndcg_at_5", 0.0)
            + 0.15 * latency_score
            + 0.15 * ingest_score
        )
        leaderboard_rows.append(
            {
                "candidate_name": result.candidate_name,
                "status": result.status,
                "stage_score": stage_score,
                "latency_score": latency_score,
                "ingest_score": ingest_score,
                **result.metrics,
                "skip_reason": result.skip_reason,
            }
        )
        for row in result.artifacts.get("pair_metrics.json", []):
            if isinstance(row, dict):
                pair_rows.append(dict(row))

    return append_stage_score(leaderboard_rows, score_key="stage_score"), append_stage_score(
        pair_rows,
        score_key="chunk_recall_at_5",
    )


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
        "stage": "vectordb",
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
