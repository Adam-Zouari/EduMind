"""Reviewed system composition and atomic held-out execution ownership."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from edumind.common.artifacts import sha256_file, stable_hash
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.protocol import (
    choice,
    integer,
    strict_object,
    string,
)
from experiments.benchmarks.rag.generation.evidence import write_new_json


@dataclass(frozen=True)
class Composition:
    path: Path
    system: dict
    upstream: dict
    protocols: dict
    checksum: str


def load_composition(path, protocols):
    path = Path(path).resolve()
    root = strict_object(
        json.loads(path.read_text(encoding="utf-8")),
        "Final composition",
        {
            "schema_version",
            "selected_by",
            "selected_date",
            "reason",
            "system",
            "upstream_decisions",
            "protocol_checksums",
        },
    )
    if integer(root["schema_version"], "schema_version", minimum=1) != 1:
        raise ValueError("Unsupported Final composition schema")
    for field in ("selected_by", "reason"):
        string(root[field], field)
    date.fromisoformat(string(root["selected_date"], "selected_date"))
    system = strict_object(
        root["system"],
        "system",
        {
            "chunking_embedding",
            "retrieval",
            "generator",
            "database",
            "top_k",
            "residency",
        },
    )
    for field in ("chunking_embedding", "retrieval", "generator", "database"):
        string(system[field], field)
    if integer(system["top_k"], "top_k", minimum=1) not in (3, 5):
        raise ValueError("Select exactly one Final top-K: 3 or 5")
    choice(system["residency"], "residency", ("resident", "sequential"))
    recorded = strict_object(
        root["protocol_checksums"], "protocol_checksums", set(protocols)
    )
    if recorded != {key: value.checksum for key, value in protocols.items()}:
        raise ValueError(
            "Final composition protocol identities changed; review a new composition"
        )
    fields = {"chunking_embedding", "retrieval", "generator", "database"}
    decisions = strict_object(root["upstream_decisions"], "upstream_decisions", fields)
    upstream = {}
    for key, source in decisions.items():
        source = Path(string(source, "upstream decision"))
        source = source if source.is_absolute() else path.parent / source
        identities = {
            "chunking_embedding": ("chunking_embedding",),
            "retrieval": ("chunking_embedding", "retrieval"),
            "generator": ("generation",),
            "database": ("vector_database",),
        }[key]
        decision = load_engineer_decision(
            source,
            exact=1,
            expected_protocols={name: protocols[name].checksum for name in identities},
        )
        if decision.selected_candidates != (system[key],):
            raise ValueError(
                f"Final {key} does not match its reviewed component winner"
            )
        summary = json.loads(decision.source_summary.read_text(encoding="utf-8"))
        if summary["plan"]["profile"] != "validation":
            raise ValueError(
                "Final composition must be selected before component locked reports"
            )
        expected_stage = {
            "chunking_embedding": "chunking-embedding",
            "retrieval": "retrieval-reranking",
            "generator": "generation",
            "database": "database-configuration",
        }[key]
        if summary["plan"]["stage"] != expected_stage:
            raise ValueError(f"Wrong upstream benchmark for {key}")
        if (
            key == "retrieval"
            and summary["plan"].get("settings", {}).get("chunker_embedding")
            != system["chunking_embedding"]
        ):
            raise ValueError(
                "Reviewed retrieval stack used a different chunking/embedding selection"
            )
        upstream[key] = {
            "path": str(source.resolve()),
            "sha256": sha256_file(source),
            "source_summary": str(decision.source_summary),
            "summary_sha256": sha256_file(decision.source_summary),
            "source_run_id": decision.source_run_id,
        }
    database_summary = json.loads(
        Path(upstream["database"]["source_summary"]).read_text(encoding="utf-8")
    )
    lineage = (
        database_summary["plan"].get("settings", {}).get("integration_provenance", {})
    )
    expected_lineage = {
        "chunking_embedding_candidate": system["chunking_embedding"],
        "retrieval_candidate": system["retrieval"],
        "chunking_protocol_checksum": recorded["chunking_embedding"],
        "retrieval_protocol_checksum": recorded["retrieval"],
        "embedding_source_run_id": upstream["chunking_embedding"]["source_run_id"],
        "retrieval_source_run_id": upstream["retrieval"]["source_run_id"],
        "embedding_decision_sha256": upstream["chunking_embedding"]["sha256"],
        "retrieval_decision_sha256": upstream["retrieval"]["sha256"],
    }
    if any(lineage.get(key) != value for key, value in expected_lineage.items()):
        raise ValueError(
            "Reviewed database integration used different component/protocol/decision identities"
        )
    identity = {
        "system": system,
        "upstream": upstream,
        "protocols": recorded,
        "decision_sha256": sha256_file(path),
    }
    return Composition(path, system, upstream, recorded, stable_hash(identity))


def reserve_locked(path, identity):
    """A failed held-out inference still consumes its one execution permission."""
    try:
        write_new_json(path, {"schema": "final-locked-consumption-v2", **identity})
    except FileExistsError as exc:
        raise ValueError(
            "Final locked data was already consumed; resume evaluation only"
        ) from exc
