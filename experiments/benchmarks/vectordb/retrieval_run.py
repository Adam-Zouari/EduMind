"""Complete retrieval comparison for dense-benchmark finalists plus Chroma."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.evaluation import (
    RETRIEVAL_QUALITY_DIRECTIONS,
    build_index,
    reranker_for,
    retrieval_metrics,
)
from experiments.benchmarks.rag.methods import BM25, reciprocal_rank_fusion
from experiments.benchmarks.vectordb.adapters import Config, Record, create
from experiments.benchmarks.vectordb.conformance import _finish_index
from experiments.benchmarks.vectordb.docker_metrics import image_lock, verify_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare complete retrieval on DB finalists")
    parser.add_argument("--database-selection", type=Path, required=True)
    parser.add_argument("--embedding-selection", type=Path, required=True)
    parser.add_argument("--retrieval-selection", type=Path, required=True)
    parser.add_argument("--profile", choices=("standard", "full"), default="standard")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    database_decision = load_engineer_decision(
        arguments.database_selection, minimum=2, maximum=3
    )
    database_payload = _payload(database_decision.source_summary)
    embedding = _single_selection(arguments.embedding_selection)
    retrieval = _single_selection(arguments.retrieval_selection)
    candidates = database_decision.selected_candidates
    chunker_name, embedding_name = embedding.split("|", 1)
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/rag-selection-validation.json")
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json"
    )
    revisions = model_revisions(model_lock)
    vector_revisions = image_lock()
    index = build_index(manifest, chunker_name, embedding_name, model_lock, with_bm25=True)
    bm25 = BM25([chunk.text for chunk in index.chunks])
    by_id = {chunk.identifier: position for position, chunk in enumerate(index.chunks)}
    plan = BenchmarkPlan(
        "vectordb-server-v4",
        "complete-retrieval",
        arguments.profile,
        manifest.name,
        candidates,
        repetitions=3,
        bootstrap_resamples=10_000,
    )

    def evaluate(candidate):
        verify_image(candidate, vector_revisions[f"image:{candidate}"])
        reranker = reranker_for(retrieval, model_lock)
        config = _config(database_payload, candidate, index.vectors.shape[1])
        adapter = create(candidate, config)
        try:
            adapter.reset()
            adapter.upsert(
                [
                    Record(
                        chunk.identifier,
                        index.vectors[position],
                        chunk.text,
                        {
                            "source_id": chunk.document_id,
                            "document_id": chunk.document_id,
                            "start": chunk.start,
                            "end": chunk.end,
                            "token_count": chunk.tokens,
                            "chunking_fingerprint": chunker_name,
                            "embedding_fingerprint": embedding_name,
                        },
                    )
                    for position, chunk in enumerate(index.chunks)
                ]
            )
            _finish_index(adapter)
            adapter.index_info()
            samples, latencies = [], []
            questions = list(
                row
                for row in manifest.samples
                if row.get("kind") == "question" and row.get("answerable") and row.get("evidence")
            )

            def retrieve(question):
                query = str(question["question"])
                started = time.perf_counter()
                query_vector = index.embedder.embed_query(query)
                dense = [by_id[hit.identifier] for hit in adapter.search(query_vector, 20)]
                lexical = [position for position, _ in bm25.rank(query, 20)]
                if retrieval == "dense":
                    order = dense
                elif retrieval == "bm25":
                    order = lexical
                else:
                    order = reciprocal_rank_fusion([dense, lexical], 20)
                    if reranker is not None:
                        local = reranker.rank(query, [index.chunks[position].text for position in order])
                        order = [order[position] for position in local]
                return order, time.perf_counter() - started

            for question in questions[:2]:
                retrieve(question)
            for question in questions:
                measured = [retrieve(question) for _ in range(plan.repetitions)]
                orders = [value[0] for value in measured]
                item_latencies = [value[1] for value in measured]
                order = orders[0]
                selected = [index.chunks[position] for position in order[:10]]
                latency = float(np.median(item_latencies))
                latencies.extend(item_latencies)
                metrics, tokens = retrieval_metrics(question, selected, index.chunks, index.tokenizer)
                metrics["determinism"] = float(all(value == order for value in orders))
                samples.append(
                    SampleResult(str(question["id"]), metrics, latency, {"retrieved_tokens": tokens})
                )
            operational = {
                "p50_latency_seconds": float(np.median(latencies)),
                "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
            }
            for concurrency in (1, 8, 32, 64):
                started = time.perf_counter()
                errors = 0
                concurrent_latencies = []
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(retrieve, question)
                        for _ in range(plan.repetitions)
                        for question in questions
                    ]
                    for future in as_completed(futures):
                        try:
                            concurrent_latencies.append(future.result()[1])
                        except Exception:
                            errors += 1
                elapsed = time.perf_counter() - started
                count = len(questions) * plan.repetitions
                operational[f"throughput_concurrency_{concurrency}_qps"] = count / max(elapsed, 1e-9)
                operational[f"error_rate_concurrency_{concurrency}"] = errors / count
                if concurrent_latencies:
                    operational[f"p95_concurrency_{concurrency}_seconds"] = float(
                        np.quantile(concurrent_latencies, 0.95)
                    )
                    operational[f"p99_concurrency_{concurrency}_seconds"] = float(
                        np.quantile(concurrent_latencies, 0.99)
                    )
            return samples, operational, {
                "target_concurrency_success": float(
                    operational["error_rate_concurrency_64"] == 0.0
                )
            }
        finally:
            adapter.close()

    result = run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions={
            **RETRIEVAL_QUALITY_DIRECTIONS,
            "operational.p50_latency_seconds": "min",
            "operational.p95_latency_seconds": "min",
            "target_concurrency_success": "max",
        },
        primary_metric="ndcg_at_5",
        revisions={**revisions, **vector_revisions},
        decision_files={
            "database": arguments.database_selection,
            "embedding": arguments.embedding_selection,
            "retrieval": arguments.retrieval_selection,
        },
        no_mlflow=arguments.no_mlflow,
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if result.complete else 2


def _payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _single_selection(path: Path) -> str:
    return load_engineer_decision(path, exact=1).selected_candidates[0]


def _config(payload, candidate, dimension):
    row = next(
        value for value in payload["candidates"] if value.get("candidate") == candidate
    )
    operational = row.get("operational", {})
    prefixes = [
        key.removesuffix(".selected_m")
        for key in operational
        if key.endswith(".selected_m")
        and ("full-real-selected" in key or f"d{dimension}" in key)
    ]
    if not prefixes:
        raise ValueError(f"No selected HNSW configuration for dimension {dimension}")
    prefix = next((value for value in prefixes if "full-real-selected" in value), prefixes[0])
    selected = {
        key.removeprefix(prefix + "."): int(value)
        for key, value in operational.items()
        if key.startswith(prefix + ".selected_")
    }
    return Config(
        dimension,
        selected.get("selected_m", 16),
        selected.get("selected_ef_construction", 100),
        selected.get("selected_ef_search", 64),
    )


if __name__ == "__main__":
    raise SystemExit(main())
