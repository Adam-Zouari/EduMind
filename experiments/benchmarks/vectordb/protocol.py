"""Strict frozen database/configuration protocol (no automatic tuning)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.benchmarks.common.protocol import (
    boolean,
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
from experiments.benchmarks.vectordb.adapters import ADAPTERS

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("protocol.yaml")
FILTER_FIELDS = {
    "source_id",
    "document_id",
    "scope_50",
    "scope_10",
    "scope_1",
    "scope_01",
}


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
    meta: object
    workloads: Mapping
    synthetic: SyntheticSettings
    smoke_hnsw: Mapping
    candidates: Mapping
    conformance: Mapping
    upsert_batch_sizes: Mapping
    adapter_settings: Mapping
    monitor_storage: bool
    monitor_server_memory: bool
    confidence_level: float
    shortlist_limit: int
    ci_support: Mapping

    def profile(self, name):
        return self.meta.profile(name)

    def __getattr__(self, name):
        settings = object.__getattribute__(self, "adapter_settings")
        if name in settings:
            return settings[name]
        raise AttributeError(name)


def load_protocol(path=DEFAULT_PROTOCOL_PATH):
    root = strict_object(
        load_yaml(path, "vector-database"),
        "vector protocol",
        {
            "schema_version",
            "protocol_version",
            "seed",
            "workloads",
            "synthetic",
            "candidates",
            "smoke_hnsw",
            "conformance",
            "adapters",
            "measurement",
            "statistics",
            "profiles",
            "shortlist_limit",
        },
    )
    profiles = execution_profiles(
        root["profiles"], names=("smoke", "development", "validation", "locked")
    )
    for name, profile in profiles.items():
        if profile.device != "cpu" or profile.batch_size != 1 or profile.warmups != 1:
            raise ValueError("Vector profiles require CPU, batch one, one warmup")
        if profile.repetitions != (1 if name == "smoke" else 3):
            raise ValueError(
                "Vector profiles require one smoke / three measured attempts"
            )
    workloads = strict_object(root["workloads"], "workloads", set(profiles))
    for profile, rows in workloads.items():
        names = []
        for row in sequence(rows, "workloads." + profile):
            row = strict_object(
                row, "workload", {"name", "size", "dimension", "queries", "concurrency"}
            )
            names.append(string(row["name"], "workload.name"))
            for field in ("size", "dimension", "queries"):
                integer(row[field], "workload." + field, minimum=1)
            increasing_integers(row["concurrency"], "workload.concurrency")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate workload condition identities")
    raw = strict_object(
        root["synthetic"],
        "synthetic",
        {
            "centroid_count",
            "records_per_document",
            "cluster_noise",
            "near_duplicate_fraction",
            "near_duplicate_noise",
            "query_noise",
            "metadata_divisors",
        },
    )
    synthetic = SyntheticSettings(
        integer(raw["centroid_count"], "centroid_count", minimum=1),
        integer(raw["records_per_document"], "records_per_document", minimum=1),
        *[
            number(
                raw[field],
                field,
                minimum=0,
                **({"maximum": 1} if field == "near_duplicate_fraction" else {}),
            )
            for field in (
                "cluster_noise",
                "near_duplicate_fraction",
                "near_duplicate_noise",
                "query_noise",
            )
        ],
        {
            key: integer(value, key, minimum=1)
            for key, value in strict_object(
                raw["metadata_divisors"],
                "metadata_divisors",
                FILTER_FIELDS - {"source_id", "document_id"},
            ).items()
        },
    )
    candidates = strict_object(
        root["candidates"], "candidates", {"smoke", "development"}
    )
    for profile, rows in candidates.items():
        ids = []
        for row in sequence(rows, "candidates." + profile):
            strict_object(row, "candidate", {"id", "database", "settings"})
            candidate_id = string(row["id"], "candidate.id")
            if row["database"] not in ADAPTERS or not candidate_id.startswith(
                row["database"] + "|"
            ):
                raise ValueError("Candidate ID must be database|alias")
            hnsw_settings(row["settings"])
            ids.append(candidate_id)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate identity")
    smoke = hnsw_settings(root["smoke_hnsw"])
    conformance = strict_object(
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
    for field in ("size", "dimension_query_count", "search_limit", "seed"):
        integer(conformance[field], field, minimum=0 if field == "seed" else 1)
    filters = sequence(conformance["compound_filters"], "compound_filters")
    if (
        not filters
        or len(filters) != len(set(filters))
        or not set(filters) <= FILTER_FIELDS
        or conformance["empty_filter"] not in FILTER_FIELDS
    ):
        raise ValueError("Unsupported trusted filter field")
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
    batches = {
        key: integer(value, key, minimum=1)
        for key, value in strict_object(
            adapters["upsert_batch_sizes"], "batches", set(ADAPTERS)
        ).items()
    }
    settings = {}
    for key, value in adapters.items():
        if key == "upsert_batch_sizes":
            continue
        settings[key] = (
            number(value, key, minimum=0, minimum_exclusive=True)
            if key.endswith("seconds")
            else integer(
                value,
                key,
                minimum=10
                if key == "qdrant_full_scan_threshold"
                else 0
                if key == "qdrant_indexing_threshold"
                else 1,
            )
        )
    measurement = strict_object(
        root["measurement"], "measurement", {"monitor_storage", "monitor_server_memory"}
    )
    stats = strict_object(
        root["statistics"], "statistics", {"confidence_level", "ci_support"}
    )
    confidence = number(
        stats["confidence_level"],
        "confidence_level",
        minimum=0,
        maximum=1,
        minimum_exclusive=True,
        maximum_exclusive=True,
    )
    if not isinstance(stats["ci_support"], dict):
        raise ValueError("ci_support must be a reviewed mapping; empty means unfrozen")
    for metric, policy in stats["ci_support"].items():
        strict_object(
            policy, metric, {"minimum_groups", "reviewed_by", "reviewed_date"}
        )
        integer(policy["minimum_groups"], metric, minimum=2)
        string(policy["reviewed_by"], metric)
        string(policy["reviewed_date"], metric)
    return VectorDatabaseProtocol(
        metadata("vector_database", path, root, profiles=profiles),
        workloads,
        synthetic,
        smoke,
        candidates,
        conformance,
        batches,
        settings,
        boolean(measurement["monitor_storage"], "monitor_storage"),
        boolean(measurement["monitor_server_memory"], "monitor_server_memory"),
        confidence,
        integer(root["shortlist_limit"], "shortlist_limit", minimum=1),
        stats["ci_support"],
    )


def hnsw_settings(value):
    return {
        key: integer(item, key, minimum=1)
        for key, item in strict_object(
            value, "HNSW settings", {"m", "ef_construction", "ef_search"}
        ).items()
    }
