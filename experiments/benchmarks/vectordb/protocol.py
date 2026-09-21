"""Strict protocol for vector-database server benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.vectordb.adapters import ADAPTERS
from experiments.benchmarks.common.protocol import (
    ProtocolMetadata,
    boolean,
    choice,
    execution_profiles,
    increasing_integers,
    integer,
    load_yaml,
    metadata,
    number,
    sequence,
    strict_object,
    string,
)


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")


@dataclass(frozen=True)
class SyntheticSettings:
    centroid_count: int
    records_per_document: int
    cluster_noise: float
    near_duplicate_fraction: float
    near_duplicate_noise: float
    query_noise: float
    metadata_divisors: Mapping[str, int]


@dataclass(frozen=True)
class VectorDatabaseProtocol:
    meta: ProtocolMetadata
    workloads: Mapping[str, object]
    synthetic: SyntheticSettings
    smoke_hnsw: Mapping[str, int]
    hnsw_grid: Mapping[str, tuple[int, ...]]
    validation_corpus_limit: int
    validation_query_limit: int
    target_recall_at_10: float
    tie_breaking: str
    cutoffs: tuple[int, ...]
    incremental_mutation_count: int
    development_filters: tuple[str, ...]
    validation_filters: tuple[str, ...]
    shortlist_limit: int
    conformance: Mapping[str, object]
    upsert_batch_sizes: Mapping[str, int]
    index_verification_limit: int
    qdrant_full_scan_threshold: int
    qdrant_indexing_threshold: int
    request_timeout_seconds: float
    connection_pool_limit: int
    index_readiness_timeout_seconds: float
    index_readiness_poll_seconds: float
    docker_monitor_interval_seconds: float
    docker_stats_timeout_seconds: float
    docker_storage_timeout_seconds: float
    docker_helper_timeout_seconds: float
    docker_restart_timeout_seconds: float
    monitor_storage: bool
    monitor_server_memory: bool
    confidence_level: float

    def profile(self, name: str):
        return self.meta.profile(name)

    def concurrency_levels(
        self, profile: str, workload_name: str | None = None
    ) -> tuple[int, ...]:
        if profile == "validation":
            return tuple(self.workloads["validation"]["concurrency"])
        rows = tuple(self.workloads[profile])
        if workload_name is not None:
            try:
                row = next(row for row in rows if row["name"] == workload_name)
            except StopIteration as exc:
                raise ValueError(
                    f"Unknown {profile} vector workload: {workload_name}"
                ) from exc
            return tuple(row["concurrency"])
        return tuple(
            sorted(
                {
                    int(level)
                    for row in rows
                    for level in row["concurrency"]
                }
            )
        )


def load_protocol(path: Path = DEFAULT_PROTOCOL_PATH) -> VectorDatabaseProtocol:
    root = strict_object(
        load_yaml(path, "vector-database"),
        "vector-database protocol root",
        {"schema_version", "protocol_version", "seed", "workloads", "synthetic", "hnsw", "evaluation", "conformance", "adapters", "measurement", "statistics", "profiles"},
    )
    workloads = strict_object(root["workloads"], "workloads", {"smoke", "development", "validation"})
    for profile in ("smoke", "development"):
        rows = sequence(workloads[profile], f"workloads.{profile}")
        if not rows:
            raise ValueError(f"workloads.{profile} cannot be empty")
        for index, row in enumerate(rows):
            _workload(row, f"workloads.{profile}[{index}]")
    validation = strict_object(workloads["validation"], "workloads.validation", {"synthetic_size", "synthetic_queries", "concurrency", "selected_real_query_limit"})
    for name in ("synthetic_size", "synthetic_queries", "selected_real_query_limit"):
        integer(validation[name], f"workloads.validation.{name}", minimum=1)
    increasing_integers(validation["concurrency"], "workloads.validation.concurrency")
    synthetic_raw = strict_object(root["synthetic"], "synthetic", {"centroid_count", "records_per_document", "cluster_noise", "near_duplicate_fraction", "near_duplicate_noise", "query_noise", "metadata_divisors"})
    divisors = strict_object(synthetic_raw["metadata_divisors"], "synthetic.metadata_divisors", {"scope_50", "scope_10", "scope_1", "scope_01"})
    synthetic = SyntheticSettings(
        integer(synthetic_raw["centroid_count"], "synthetic.centroid_count", minimum=1),
        integer(synthetic_raw["records_per_document"], "synthetic.records_per_document", minimum=1),
        number(synthetic_raw["cluster_noise"], "synthetic.cluster_noise", minimum=0),
        number(synthetic_raw["near_duplicate_fraction"], "synthetic.near_duplicate_fraction", minimum=0, maximum=1),
        number(synthetic_raw["near_duplicate_noise"], "synthetic.near_duplicate_noise", minimum=0),
        number(synthetic_raw["query_noise"], "synthetic.query_noise", minimum=0),
        {name: integer(value, f"synthetic.metadata_divisors.{name}", minimum=1) for name, value in divisors.items()},
    )
    hnsw = strict_object(root["hnsw"], "hnsw", {"smoke", "grid", "validation_corpus_limit", "validation_query_limit", "target_recall_at_10", "tie_breaking"})
    smoke_hnsw = _hnsw_row(hnsw["smoke"], "hnsw.smoke")
    grid_raw = strict_object(hnsw["grid"], "hnsw.grid", {"m", "ef_construction", "ef_search"})
    grid = {name: increasing_integers(grid_raw[name], f"hnsw.grid.{name}") for name in grid_raw}
    corpus_limit = integer(hnsw["validation_corpus_limit"], "hnsw.validation_corpus_limit", minimum=1)
    query_limit = integer(hnsw["validation_query_limit"], "hnsw.validation_query_limit", minimum=1)
    target = number(hnsw["target_recall_at_10"], "hnsw.target_recall_at_10", minimum=0, maximum=1)
    tie = choice(
        hnsw["tie_breaking"],
        "hnsw.tie_breaking",
        {"latency-then-m-then-ef-search-then-ef-construction"},
    )
    evaluation = strict_object(root["evaluation"], "evaluation", {"cutoffs", "incremental_mutation_count", "development_filters", "validation_filters", "shortlist_limit"})
    cutoffs = increasing_integers(evaluation["cutoffs"], "evaluation.cutoffs")
    if 10 not in cutoffs:
        raise ValueError("evaluation.cutoffs must include 10 for target_recall_at_10")
    mutations = integer(evaluation["incremental_mutation_count"], "evaluation.incremental_mutation_count", minimum=1)
    development_filters = _strings(evaluation["development_filters"], "evaluation.development_filters")
    validation_filters = _strings(evaluation["validation_filters"], "evaluation.validation_filters")
    shortlist = integer(evaluation["shortlist_limit"], "evaluation.shortlist_limit", minimum=1)
    conformance_raw = strict_object(
        root["conformance"],
        "conformance",
        {
            "size",
            "dimension_query_count",
            "seed",
            "search_limit",
            "compound_filters",
            "empty_filter",
        },
    )
    conformance = {
        name: integer(
            value,
            f"conformance.{name}",
            minimum=0 if name == "seed" else 1,
        )
        for name, value in conformance_raw.items()
        if name in {"size", "dimension_query_count", "seed", "search_limit"}
    }
    conformance["compound_filters"] = _strings(
        conformance_raw["compound_filters"], "conformance.compound_filters"
    )
    conformance["empty_filter"] = string(
        conformance_raw["empty_filter"], "conformance.empty_filter"
    )
    metadata_names = set(synthetic.metadata_divisors)
    if not set(conformance["compound_filters"]) <= metadata_names:
        raise ValueError("Conformance compound filters must be declared metadata fields")
    if conformance["empty_filter"] not in metadata_names:
        raise ValueError("Conformance empty filter must be a declared metadata field")
    if conformance["size"] < synthetic.records_per_document:
        raise ValueError("Conformance size must cover one complete document")
    adapters = strict_object(
        root["adapters"],
        "adapters",
        {
            "upsert_batch_sizes",
            "index_verification_limit",
            "qdrant_full_scan_threshold",
            "qdrant_indexing_threshold",
            "request_timeout_seconds",
            "connection_pool_limit",
            "index_readiness_timeout_seconds",
            "index_readiness_poll_seconds",
            "docker_monitor_interval_seconds",
            "docker_stats_timeout_seconds",
            "docker_storage_timeout_seconds",
            "docker_helper_timeout_seconds",
            "docker_restart_timeout_seconds",
        },
    )
    declared_adapters = set(ADAPTERS)
    batches = strict_object(
        adapters["upsert_batch_sizes"],
        "adapters.upsert_batch_sizes",
        declared_adapters,
    )
    batch_sizes = {name: integer(value, f"adapters.upsert_batch_sizes.{name}", minimum=1) for name, value in batches.items()}
    verification_limit = integer(
        adapters["index_verification_limit"],
        "adapters.index_verification_limit",
        minimum=1,
    )
    full_scan_threshold = integer(
        adapters["qdrant_full_scan_threshold"],
        "adapters.qdrant_full_scan_threshold",
        minimum=0,
    )
    indexing_threshold = integer(
        adapters["qdrant_indexing_threshold"],
        "adapters.qdrant_indexing_threshold",
        minimum=0,
    )
    timeouts = {name: number(adapters[name], f"adapters.{name}", minimum=0, minimum_exclusive=True) for name in adapters if name.endswith("_seconds")}
    pool_limit = integer(adapters["connection_pool_limit"], "adapters.connection_pool_limit", minimum=1)
    measurement = strict_object(root["measurement"], "measurement", {"monitor_storage", "monitor_server_memory"})
    monitor_storage = boolean(measurement["monitor_storage"], "measurement.monitor_storage")
    monitor_memory = boolean(measurement["monitor_server_memory"], "measurement.monitor_server_memory")
    statistics = strict_object(root["statistics"], "statistics", {"confidence_level"})
    confidence = number(
        statistics["confidence_level"],
        "statistics.confidence_level",
        minimum=0,
        maximum=1,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    profiles = execution_profiles(root["profiles"], names=("smoke", "development", "validation"))
    if {profile.batch_size for profile in profiles.values()} != {1}:
        raise ValueError("Vector-database execution profiles must use batch size one")
    return VectorDatabaseProtocol(
        metadata("vector_database", path, root, profiles=profiles), workloads,
        synthetic, smoke_hnsw, grid, corpus_limit, query_limit, target, tie,
        cutoffs, mutations, development_filters, validation_filters, shortlist,
        conformance,
        batch_sizes, verification_limit, full_scan_threshold, indexing_threshold,
        timeouts["request_timeout_seconds"], pool_limit,
        timeouts["index_readiness_timeout_seconds"], timeouts["index_readiness_poll_seconds"],
        timeouts["docker_monitor_interval_seconds"], timeouts["docker_stats_timeout_seconds"],
        timeouts["docker_storage_timeout_seconds"], timeouts["docker_helper_timeout_seconds"],
        timeouts["docker_restart_timeout_seconds"], monitor_storage, monitor_memory,
        confidence,
    )


def _workload(value: object, label: str) -> None:
    row = strict_object(value, label, {"name", "size", "dimension", "queries", "concurrency"})
    string(row["name"], f"{label}.name")
    for name in ("size", "dimension", "queries"):
        integer(row[name], f"{label}.{name}", minimum=1)
    increasing_integers(row["concurrency"], f"{label}.concurrency")


def _hnsw_row(value: object, label: str) -> dict[str, int]:
    row = strict_object(value, label, {"m", "ef_construction", "ef_search"})
    return {name: integer(item, f"{label}.{name}", minimum=1) for name, item in row.items()}


def _strings(value: object, label: str) -> tuple[str, ...]:
    values = tuple(string(item, f"{label}[{index}]") for index, item in enumerate(sequence(value, label)))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique strings")
    return values
