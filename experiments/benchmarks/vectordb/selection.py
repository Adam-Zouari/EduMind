"""Explicit selected-server configuration shared with Final RAG."""

from __future__ import annotations

import json
from dataclasses import asdict

from edumind.common.artifacts import sha256_file, stable_hash
from experiments.benchmarks.common.decisions import load_engineer_decision

from .adapters import Config

SUITE = "vectordb-server-v5"
STAGE = "database-configuration"


def make_config(
    database,
    dimension,
    protocol,
    *,
    m,
    ef_construction,
    ef_search,
    collection="edumind_benchmark",
):
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("A positive actual embedding dimension is required")
    return Config(
        dimension=dimension,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        upsert_batch_size=protocol.upsert_batch_sizes[database],
        **{
            key: getattr(protocol, key)
            for key in (
                "request_timeout_seconds",
                "connection_pool_limit",
                "index_readiness_timeout_seconds",
                "index_readiness_poll_seconds",
                "index_verification_limit",
                "qdrant_full_scan_threshold",
                "qdrant_indexing_threshold",
            )
        },
        collection=collection,
    )


def candidate_config(protocol, candidate_id, dimension):
    rows = [
        row
        for profile in protocol.candidates.values()
        for row in profile
        if row["id"] == candidate_id
    ]
    if not rows or any(row != rows[0] for row in rows):
        raise ValueError(
            f"Unknown or conflicting explicit vector candidate: {candidate_id}"
        )
    row = rows[0]
    return row["database"], make_config(
        row["database"], dimension, protocol, **row["settings"]
    )


def selected_candidates(path, profile, protocol):
    decision = load_engineer_decision(
        path,
        exact=1 if profile == "locked" else None,
        maximum=protocol.shortlist_limit,
        expected_source=(
            SUITE,
            STAGE,
            "validation" if profile == "locked" else "development",
        ),
        expected_protocols={"vector_database": protocol.meta.checksum},
    )
    summary = json.loads(decision.source_summary.read_text(encoding="utf-8"))
    for candidate_id in decision.selected_candidates:
        row = next(
            row for row in summary["candidates"] if row["candidate"] == candidate_id
        )
        database = candidate_id.split("|", 1)[0]
        declared = next(
            (
                item
                for item in protocol.candidates["development"]
                if item["id"] == candidate_id
            ),
            None,
        )
        parameters = row.get("parameters", {})
        if (
            declared is None
            or parameters.get("configuration_settings") != declared["settings"]
            or parameters.get("server") != database
        ):
            raise ValueError(
                "Selected configuration settings are missing/stale; legacy inferred settings are not accepted"
            )
    return decision


def resolve_selected_config(decision_path, protocol, dimension):
    """Return (database, Config, provenance) for the one reviewed validation winner."""
    decision = selected_candidates(decision_path, "locked", protocol)
    candidate_id = decision.selected_candidates[0]
    database, config = candidate_config(protocol, candidate_id, dimension)
    provenance = {
        "candidate_id": candidate_id,
        "decision_checksum": sha256_file(decision_path),
        "source_run_id": decision.source_run_id,
        "source_summary": str(decision.source_summary),
        "configuration": asdict(config),
        "configuration_checksum": stable_hash(asdict(config)),
        "protocol_checksum": protocol.meta.checksum,
    }
    return database, config, provenance
