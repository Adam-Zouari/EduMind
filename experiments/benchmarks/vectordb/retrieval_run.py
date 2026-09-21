"""Complete retrieval comparison for dense-benchmark finalists plus Chroma."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.preparation.models import load_selected_model_lock, model_revisions
from experiments.benchmarks.rag.evaluation import (
    build_index,
    reranker_for,
    retrieval_metrics,
    retrieval_quality_directions,
)
from experiments.benchmarks.rag.methods import reciprocal_rank_fusion
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_CHUNKING_PROTOCOL_PATH,
    load_protocol as load_chunking_protocol,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_RETRIEVAL_PROTOCOL_PATH,
    load_protocol as load_retrieval_protocol,
)
from experiments.benchmarks.vectordb.adapters import Config, Record, create
from experiments.benchmarks.vectordb.conformance import _finish_index
from experiments.benchmarks.vectordb.docker_metrics import image_lock, verify_image
from experiments.benchmarks.vectordb.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_VECTOR_PROTOCOL_PATH,
    VectorDatabaseProtocol,
    load_protocol as load_vector_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare complete retrieval on DB finalists")
    parser.add_argument("--database-selection", type=Path, required=True)
    parser.add_argument("--embedding-selection", type=Path, required=True)
    parser.add_argument("--retrieval-selection", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("development", "validation"),
        default="development",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_VECTOR_PROTOCOL_PATH)
    parser.add_argument("--retrieval-protocol", type=Path, default=DEFAULT_RETRIEVAL_PROTOCOL_PATH)
    parser.add_argument("--chunking-protocol", type=Path, default=DEFAULT_CHUNKING_PROTOCOL_PATH)
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    vector_protocol = load_vector_protocol(arguments.protocol)
    retrieval_protocol = load_retrieval_protocol(arguments.retrieval_protocol)
    chunking_protocol = load_chunking_protocol(arguments.chunking_protocol)
    vector_execution = vector_protocol.profile(arguments.profile)
    retrieval_execution = retrieval_protocol.profile(arguments.profile)
    device = arguments.device or retrieval_execution.device
    dtype = arguments.dtype or retrieval_execution.dtype
    if retrieval_execution.hardware_required and (device, dtype) != (
        retrieval_execution.device,
        retrieval_execution.dtype,
    ):
        parser.error(
            f"{arguments.profile} complete retrieval requires "
            f"--device {retrieval_execution.device} --dtype {retrieval_execution.dtype}"
        )
    database_decision = load_engineer_decision(
        arguments.database_selection,
        minimum=2,
        maximum=vector_protocol.shortlist_limit,
        expected_source=("vectordb-server-v4", "dense-ann", "validation"),
    )
    database_payload = _payload(database_decision.source_summary)
    embedding = _single_selection(
        arguments.embedding_selection, "chunking-embedding"
    )
    retrieval = _single_selection(
        arguments.retrieval_selection, "retrieval-reranking"
    )
    candidates = database_decision.selected_candidates
    chunker_name, embedding_name = embedding.split("|", 1)
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/rag-selection-validation.json")
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json"
    )
    revisions = model_revisions(model_lock)
    vector_revisions = image_lock()
    index = build_index(
        manifest,
        chunker_name,
        embedding_name,
        model_lock,
        with_bm25=True,
        device=device,
        dtype=dtype,
        chunking_protocol=chunking_protocol,
        retrieval_protocol=retrieval_protocol,
    )
    if index.bm25 is None:
        raise RuntimeError("Complete retrieval requires the BM25 index")
    bm25 = index.bm25
    by_id = {chunk.identifier: position for position, chunk in enumerate(index.chunks)}
    plan = BenchmarkPlan(
        "vectordb-server-v4",
        "complete-retrieval",
        arguments.profile,
        manifest.name,
        candidates,
        seed=vector_protocol.meta.seed,
        repetitions=vector_execution.repetitions,
        bootstrap_resamples=vector_execution.bootstrap_resamples,
        warmups=vector_execution.warmups,
        settings={
            "device": device,
            "dtype": dtype,
        },
    )

    def evaluate(candidate):
        verify_image(
            candidate,
            vector_revisions[f"image:{candidate}"],
            timeout_seconds=vector_protocol.docker_storage_timeout_seconds,
        )
        reranker = reranker_for(
            retrieval,
            model_lock,
            device=device,
            dtype=dtype,
            retrieval_protocol=retrieval_protocol,
        )
        config = _config(
            database_payload,
            candidate,
            index.vectors.shape[1],
            vector_protocol,
        )
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
                dense = [
                    by_id[hit.identifier]
                    for hit in adapter.search(query_vector, retrieval_protocol.pool_size)
                ]
                lexical = [
                    position
                    for position, _ in bm25.rank(query, retrieval_protocol.pool_size)
                ]
                first_stage = retrieval.split("|", 1)[0]
                if first_stage == "dense":
                    order = dense
                elif first_stage == "bm25":
                    order = lexical
                elif first_stage == "rrf":
                    order = reciprocal_rank_fusion(
                        [dense, lexical],
                        retrieval_protocol.pool_size,
                        retrieval_protocol.rrf_constant,
                        weights=retrieval_protocol.rrf_weights,
                        tie_keys={
                            position: chunk.identifier
                            for position, chunk in enumerate(index.chunks)
                        },
                    )
                else:
                    raise ValueError(f"Unsupported retrieval stack: {retrieval}")
                if reranker is not None:
                    local = reranker.rank(
                        query, [index.chunks[position].text for position in order]
                    )
                    order = [order[position] for position in local]
                return order, time.perf_counter() - started

            questions = questions[: int(vector_protocol.workloads["validation"]["selected_real_query_limit"])]
            for question in questions[: plan.warmups]:
                retrieve(question)
            for question in questions:
                measured = [retrieve(question) for _ in range(plan.repetitions)]
                orders = [value[0] for value in measured]
                item_latencies = [value[1] for value in measured]
                order = orders[0]
                selected = [
                    index.chunks[position]
                    for position in order[: max(retrieval_protocol.quality_cutoffs)]
                ]
                latency = float(np.median(item_latencies))
                latencies.extend(item_latencies)
                metrics, tokens = retrieval_metrics(
                    question,
                    selected,
                    index.chunks,
                    index.tokenizer,
                    cutoffs=retrieval_protocol.quality_cutoffs,
                )
                metrics["determinism"] = float(all(value == order for value in orders))
                samples.append(
                    SampleResult(str(question["id"]), metrics, latency, {"retrieved_tokens": tokens})
                )
            operational = {
                "p50_latency_seconds": float(np.median(latencies)),
                "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
            }
            concurrency_values = vector_protocol.concurrency_levels(arguments.profile)
            for concurrency in concurrency_values:
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
                        operational[f"error_rate_concurrency_{concurrency_values[-1]}"] == 0.0
                )
            }
        finally:
            adapter.close()

    result = run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions={
            **retrieval_quality_directions(retrieval_protocol.quality_cutoffs),
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
        protocols={
            "vector_database": vector_protocol.meta,
            "retrieval": retrieval_protocol.meta,
            "chunking_embedding": chunking_protocol.meta,
        },
        no_mlflow=arguments.no_mlflow,
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if result.complete else 2


def _payload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _single_selection(path: Path, stage: str) -> str:
    return load_engineer_decision(
        path,
        exact=1,
        expected_source=("rag", stage, "validation"),
    ).selected_candidates[0]


def _config(
    payload,
    candidate,
    dimension,
    protocol: VectorDatabaseProtocol,
):
    row = next(
        value for value in payload["candidates"] if value.get("candidate") == candidate
    )
    operational = row.get("operational", {})
    prefixes = [
        key.removesuffix(".selected_m")
        for key in operational
        if key.endswith(".selected_m")
        and ("validation-real-selected" in key or f"d{dimension}" in key)
    ]
    if not prefixes:
        raise ValueError(f"No selected HNSW configuration for dimension {dimension}")
    prefix = next(
        (value for value in prefixes if "validation-real-selected" in value),
        prefixes[0],
    )
    selected = {
        key.removeprefix(prefix + "."): int(value)
        for key, value in operational.items()
        if key.startswith(prefix + ".selected_")
    }
    return Config(
        dimension,
        selected["selected_m"],
        selected["selected_ef_construction"],
        selected["selected_ef_search"],
        upsert_batch_size=protocol.upsert_batch_sizes[candidate],
        request_timeout_seconds=protocol.request_timeout_seconds,
        connection_pool_limit=protocol.connection_pool_limit,
        index_readiness_timeout_seconds=protocol.index_readiness_timeout_seconds,
        index_readiness_poll_seconds=protocol.index_readiness_poll_seconds,
        index_verification_limit=protocol.index_verification_limit,
        qdrant_full_scan_threshold=protocol.qdrant_full_scan_threshold,
        qdrant_indexing_threshold=protocol.qdrant_indexing_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
