from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from edumind.common.paths import PROJECT_ROOT
from edumind.common.artifacts import sha256_file, stable_hash
from experiments.benchmarks.common.arguments import resolved_candidates
from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.vectordb.adapters import Config, InvalidIndexState, Record, create
from experiments.benchmarks.vectordb.conformance import _finish_index, check
from experiments.benchmarks.vectordb.docker_metrics import DockerMonitor, image_lock, verify_image
from experiments.benchmarks.vectordb.workload import clustered, exact_ids, recall, records
from experiments.benchmarks.vectordb.workload import Corpus

DIRECTORY = Path(__file__).parent
COMPOSE = DIRECTORY / "compose.yml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark four real vector database servers")
    parser.add_argument("--profile", choices=("smoke", "standard", "full"), default="smoke")
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument(
        "--embedding-selection",
        type=Path,
        help="required for full: engineer-selected chunking/embedding pair",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    candidates = resolved_candidates(DIRECTORY / "candidates.yaml", arguments.profile, arguments.shortlist)
    if arguments.profile == "full" and len(candidates) > 3:
        raise ValueError("Select at most three vector-database finalists before full")
    revisions = image_lock()
    workloads = _workloads(arguments.profile, arguments.embedding_selection)
    dataset_name = "+".join(name for name, _ in workloads)
    if arguments.embedding_selection is not None:
        dataset_name += f"@{sha256_file(arguments.embedding_selection)[:12]}"
    plan = BenchmarkPlan(
        "vectordb-server-v4",
        "dense-ann",
        arguments.profile,
        dataset_name,
        candidates,
        repetitions=1 if arguments.profile == "smoke" else 3,
        bootstrap_resamples=0 if arguments.profile == "smoke" else 10_000,
    )

    def evaluate(candidate: str):
        verify_image(candidate, revisions[f"image:{candidate}"])
        samples: list[SampleResult] = []
        operational: dict[str, float] = {}
        correctness: dict[str, float] = {}
        selected_configs: dict[str, dict[str, int]] = {}
        with DockerMonitor(candidate) as docker:
            for workload_number, (name, corpus) in enumerate(workloads):
                size, dimension = len(corpus.vectors), corpus.vectors.shape[1]
                config, trials, unsupported = _select_config(
                    candidate, corpus, arguments.profile
                )
                trial_count = len(trials) + len(unsupported)
                operational[f"{name}.tuning_trials"] = float(trial_count)
                operational[f"{name}.unsupported_configurations"] = float(len(unsupported))
                for trial_config, trial_recall, trial_p95 in trials:
                    trial_name = (
                        f"{name}.tuning_m{trial_config.m}_"
                        f"construction{trial_config.ef_construction}_"
                        f"search{trial_config.ef_search}"
                    )
                    operational[f"{trial_name}.ann_recall_at_10"] = trial_recall
                    operational[f"{trial_name}.p95_latency_seconds"] = trial_p95
                for unsupported_config in unsupported:
                    key = (
                        f"{name}.unsupported_m{unsupported_config.m}_"
                        f"construction{unsupported_config.ef_construction}_"
                        f"search{unsupported_config.ef_search}"
                    )
                    operational[key] = 1.0
                operational[f"{name}.selected_m"] = float(config.m)
                operational[f"{name}.selected_ef_construction"] = float(config.ef_construction)
                operational[f"{name}.selected_ef_search"] = float(config.ef_search)
                selected_configs[name] = {
                    "m": config.m,
                    "ef_construction": config.ef_construction,
                    "ef_search": config.ef_search,
                }
                if workload_number == 0:
                    adapter, conformance = check(candidate, config, COMPOSE)
                    operational["restart_readiness_seconds"] = float(
                        conformance.pop("restart_readiness_seconds")
                    )
                    operational["process_cold_query_seconds"] = float(
                        conformance.pop("process_cold_query_seconds")
                    )
                    correctness.update(conformance)
                else:
                    adapter = create(candidate, config)
                try:
                    adapter.reset()
                    build_started = time.perf_counter()
                    adapter.upsert(records(corpus))
                    _finish_index(adapter)
                    adapter.index_info()
                    build_seconds = time.perf_counter() - build_started
                    workload_samples, measured, success = _measure(
                        adapter, corpus, name, arguments.profile, plan.repetitions
                    )
                    samples.extend(workload_samples)
                    operational[f"{name}.build_seconds"] = build_seconds
                    operational[f"{name}.build_vectors_per_second"] = size / max(build_seconds, 1e-9)
                    operational.update({f"{name}.{key}": value for key, value in measured.items()})
                    correctness[f"{name}.target_concurrency_success"] = success
                    incremental = [
                        Record(
                            f"incremental-{index}",
                            corpus.vectors[index],
                            f"incremental {index}",
                            corpus.metadata[index],
                        )
                        for index in range(min(1_000, size))
                    ]
                    incremental_started = time.perf_counter()
                    adapter.upsert(incremental)
                    incremental_seconds = time.perf_counter() - incremental_started
                    operational[f"{name}.incremental_upsert_per_second"] = len(incremental) / max(
                        incremental_seconds, 1e-9
                    )
                    delete_started = time.perf_counter()
                    adapter.delete([row.identifier for row in incremental])
                    delete_seconds = time.perf_counter() - delete_started
                    operational[f"{name}.incremental_delete_per_second"] = len(incremental) / max(
                        delete_seconds, 1e-9
                    )
                finally:
                    adapter.close()
        operational.update(docker.metrics())
        correctness["target_concurrency_success"] = min(
            value for key, value in correctness.items() if key.endswith("target_concurrency_success")
        )
        return samples, operational, correctness, {
            "server": candidate,
            "selected_hnsw": json.dumps(selected_configs, sort_keys=True),
        }

    result = run_benchmark(
        plan,
        evaluate,
        dataset_checksum=stable_hash(
            [{"name": name, "fingerprint": corpus.fingerprint} for name, corpus in workloads]
        ),
        directions={
            "ann_recall_at_1": "max",
            "ann_recall_at_3": "max",
            "ann_recall_at_5": "max",
            "ann_recall_at_10": "max",
            "filtered_ann_recall_at_1": "max",
            "filtered_ann_recall_at_3": "max",
            "filtered_ann_recall_at_5": "max",
            "filtered_ann_recall_at_10": "max",
            "filter_correctness": "max",
            "empty_filter_correctness": "max",
            "health_correctness": "max",
            "cosine_correctness": "max",
            "compound_filter_correctness": "max",
            "dimension_rejection_correctness": "max",
            "duplicate_id_correctness": "max",
            "replacement_correctness": "max",
            "deletion_correctness": "max",
            "restart_persistence_correctness": "max",
            "ann_index_verified": "max",
            "ann_index_verified_after_restart": "max",
            "target_concurrency_success": "max",
            "operational.peak_server_memory_bytes": "min",
            "operational.persistent_storage_bytes": "min",
        },
        primary_metric="ann_recall_at_10",
        revisions=revisions,
        decision_files={
            name: path
            for name, path in {
                "shortlist": arguments.shortlist,
                "embedding": arguments.embedding_selection,
            }.items()
            if path is not None
        },
        no_mlflow=arguments.no_mlflow,
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if result.complete else 2


def _workloads(profile: str, embedding_selection: Path | None):
    if profile == "smoke":
        return (("smoke-1k-d384", clustered(1_000, 384, 50)),)
    if profile == "standard":
        return (
            ("standard-100k-d384", clustered(100_000, 384, 500)),
            ("standard-100k-d1024", clustered(100_000, 1_024, 500)),
        )
    if embedding_selection is None:
        raise ValueError("Full vector benchmark requires --embedding-selection DECISION_JSON")
    real = _real_corpus(embedding_selection)
    dimension = real.vectors.shape[1]
    return (
        ("full-real-selected", real),
        (f"full-1m-d{dimension}", clustered(1_000_000, dimension, 1_000)),
    )


def _real_corpus(selection_path: Path) -> Corpus:
    from experiments.benchmarks.common.datasets import load_manifest
    from experiments.benchmarks.preparation.models import load_selected_model_lock
    from experiments.benchmarks.rag.evaluation import build_index

    selected = load_engineer_decision(selection_path, exact=1).selected_candidates[0]
    chunker, embedding = selected.split("|", 1)
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/rag-selection-validation.json")
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json"
    )
    index = build_index(manifest, chunker, embedding, model_lock, with_bm25=False)
    questions = [row for row in manifest.samples if row.get("kind") == "question"][:1_000]
    query_vectors = np.asarray(
        [index.embedder.embed_query(str(row["question"])) for row in questions], dtype=np.float32
    )
    metadata = tuple(
        {
            "source_id": chunk.document_id,
            "scope_50": str(position % 2),
            "scope_10": str(position % 10),
            "scope_1": str(position % 100),
            "scope_01": str(position % 1000),
        }
        for position, chunk in enumerate(index.chunks)
    )
    query_indices = np.arange(len(query_vectors)) % len(index.chunks)
    return Corpus(
        index.vectors,
        query_vectors,
        query_indices,
        metadata,
        stable_hash(
            {
                "embedding_selection": sha256_file(selection_path),
                "manifest": manifest.fingerprint,
                "selected": selected,
                "vector_count": len(index.vectors),
                "query_count": len(query_vectors),
            }
        ),
    )


def _select_config(candidate, corpus, profile):
    configurations = [Config(corpus.vectors.shape[1])]
    if profile != "smoke":
        configurations = [
            Config(corpus.vectors.shape[1], m, construction, search)
            for m in (16, 32)
            for construction in (100, 200)
            for search in (64, 128)
        ]
    validation_size = min(10_000, len(corpus.vectors))
    validation = type(corpus)(
        corpus.vectors[:validation_size],
        corpus.queries[: min(50, len(corpus.queries))],
        corpus.query_indices[: min(50, len(corpus.query_indices))],
        corpus.metadata[:validation_size],
        stable_hash(
            {
                "parent": corpus.fingerprint,
                "validation_vectors": validation_size,
                "validation_queries": min(50, len(corpus.queries)),
            }
        ),
    )
    trials = []
    unsupported = []
    for config in configurations:
        adapter = None
        try:
            adapter = create(candidate, config)
            adapter.reset()
            adapter.upsert(records(validation))
            _finish_index(adapter)
            adapter.index_info()
            recalls, latencies = [], []
            for query in validation.queries:
                expected = exact_ids(validation, query, 10)
                started = time.perf_counter()
                hits = adapter.search(query, 10)
                latencies.append(time.perf_counter() - started)
                recalls.append(recall(hits, expected, 10))
            trials.append((config, statistics.fmean(recalls), float(np.quantile(latencies, 0.95))))
        except InvalidIndexState:
            raise
        except Exception:
            unsupported.append(config)
        finally:
            if adapter is not None:
                adapter.close()
    if not trials:
        raise RuntimeError(f"{candidate} could not execute any declared HNSW configuration")
    passing = [trial for trial in trials if trial[1] >= 0.99]
    if passing:
        selected_trial = min(
            passing, key=lambda trial: (trial[2], trial[0].m, trial[0].ef_search)
        )
    else:
        # A low recall result is evidence, not a failed experiment. Measure the
        # highest-recall supported profile so the engineer can review it.
        selected_trial = max(trials, key=lambda trial: (trial[1], -trial[2]))
    selected = selected_trial[0]
    return selected, trials, unsupported


def _measure(adapter, corpus, workload, profile, repetitions):
    samples = []
    unfiltered_latencies = []
    filter_names = ("scope_50", "scope_10", "scope_1") if profile != "full" else (
        "scope_50", "scope_10", "scope_1", "scope_01"
    )
    filtered_latencies = {name: [] for name in filter_names}
    filter_groups: dict[tuple[str, str], list[int]] = {}
    for index, metadata_row in enumerate(corpus.metadata):
        for filter_name in filter_names:
            filter_groups.setdefault(
                (filter_name, str(metadata_row[filter_name])), []
            ).append(index)
    filter_indices = {
        key: np.asarray(values, dtype=np.int64) for key, values in filter_groups.items()
    }
    del filter_groups
    for query_number, query in enumerate(corpus.queries):
        expected = exact_ids(corpus, query, 10)
        started = time.perf_counter()
        hits = adapter.search(query, 10)
        unfiltered_latencies.append(time.perf_counter() - started)
        metrics = {
            f"ann_recall_at_{cutoff}": recall(hits, expected, cutoff)
            for cutoff in (1, 3, 5, 10)
        }
        filtered_values = {cutoff: [] for cutoff in (1, 3, 5, 10)}
        correct = []
        metadata = corpus.metadata[int(corpus.query_indices[query_number])]
        for filter_name in filter_names:
            filters = {filter_name: metadata[filter_name]}
            filtered_started = time.perf_counter()
            filtered = adapter.search(query, 10, filters)
            filtered_latencies[filter_name].append(time.perf_counter() - filtered_started)
            expected_filtered = exact_ids(
                corpus,
                query,
                10,
                filters,
                candidate_indices=filter_indices[(filter_name, str(metadata[filter_name]))],
            )
            for cutoff in (1, 3, 5, 10):
                value = recall(filtered, expected_filtered, cutoff)
                metrics[f"filtered_ann_recall_{filter_name}_at_{cutoff}"] = value
                filtered_values[cutoff].append(value)
            correct.append(all(str(hit.metadata.get(filter_name)) == metadata[filter_name] for hit in filtered))
        for cutoff in (1, 3, 5, 10):
            metrics[f"filtered_ann_recall_at_{cutoff}"] = statistics.fmean(filtered_values[cutoff])
        metrics["filter_correctness"] = float(all(correct))
        metrics["empty_filter_correctness"] = float(not adapter.search(query, 10, {"scope_1": "missing"}))
        samples.append(SampleResult(f"{workload}:q{query_number}", metrics, unfiltered_latencies[-1]))

    operational = {
        "p50_latency_seconds": float(np.median(unfiltered_latencies)),
        "p95_latency_seconds": float(np.quantile(unfiltered_latencies, 0.95)),
        "p99_latency_seconds": float(np.quantile(unfiltered_latencies, 0.99)),
    }
    for filter_name, values in filtered_latencies.items():
        operational[f"p50_filtered_{filter_name}_seconds"] = float(np.median(values))
        operational[f"p95_filtered_{filter_name}_seconds"] = float(np.quantile(values, 0.95))
        operational[f"p99_filtered_{filter_name}_seconds"] = float(np.quantile(values, 0.99))
    concurrencies = (1,) if profile == "smoke" else (1, 8, 32) if profile == "standard" else (1, 8, 32, 64)
    target_success = 1.0
    for concurrency in concurrencies:
        for query in corpus.queries[:2]:
            adapter.search(query, 10)
        elapsed, errors, latencies = _concurrent(adapter, corpus.queries, concurrency, repetitions)
        operational[f"throughput_concurrency_{concurrency}_qps"] = len(corpus.queries) * repetitions / max(elapsed, 1e-9)
        operational[f"error_rate_concurrency_{concurrency}"] = errors / (len(corpus.queries) * repetitions)
        if latencies:
            operational[f"p50_concurrency_{concurrency}_seconds"] = float(np.median(latencies))
            operational[f"p95_concurrency_{concurrency}_seconds"] = float(np.quantile(latencies, 0.95))
            operational[f"p99_concurrency_{concurrency}_seconds"] = float(np.quantile(latencies, 0.99))
        if concurrency == concurrencies[-1] and errors:
            target_success = 0.0
    return samples, operational, target_success


def _concurrent(adapter, queries, concurrency, repetitions):
    def measured(query):
        started_query = time.perf_counter()
        adapter.search(query, 10)
        return time.perf_counter() - started_query

    started = time.perf_counter()
    errors = 0
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(measured, query) for _ in range(repetitions) for query in queries]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception:
                errors += 1
    return time.perf_counter() - started, errors, latencies


if __name__ == "__main__":
    raise SystemExit(main())
