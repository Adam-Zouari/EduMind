"""Final winners must retain the reviewed cross-benchmark integration lineage."""

import json
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import sha256_file
from experiments.benchmarks.rag.final.composition import load_composition
from experiments.benchmarks.rag.final.run import _verify_model_lineage


@pytest.fixture
def composition_inputs(tmp_path):
    protocols = {
        name: SimpleNamespace(checksum="test-only-" + name)
        for name in (
            "chunking_embedding",
            "retrieval",
            "generation",
            "vector_database",
            "final_rag",
        )
    }
    system = {
        "chunking_embedding": "test-chunker|test-embedding",
        "retrieval": "dense|none",
        "generator": "test-generator",
        "database": "chroma|test-config",
        "top_k": 3,
        "residency": "resident",
    }
    model_lock = tmp_path / "selected.json"
    model_lock.write_text('{"fixture_only": true}', encoding="utf-8")
    decisions, summaries = {}, {}
    for key, stage in (
        ("chunking_embedding", "chunking-embedding"),
        ("retrieval", "retrieval-reranking"),
        ("generator", "generation"),
        ("database", "database-configuration"),
    ):
        summary = tmp_path / (key + "-summary.json")
        payload = {
            "complete": True,
            "run_id": "test-only-" + key,
            "plan": {
                "suite": "test-only",
                "profile": "validation",
                "stage": stage,
                "settings": {"chunker_embedding": system["chunking_embedding"]},
            },
            "protocols": {
                name: {"checksum": protocol.checksum}
                for name, protocol in protocols.items()
            },
            "provenance": {
                "model_revisions": {
                    "test-embedding": "test-embedding-revision",
                    "test-generator-model": "test-generator-revision",
                }
            },
            "candidates": [{"candidate": system[key], "status": "success"}],
        }
        summary.write_text(json.dumps(payload), encoding="utf-8")
        decision = tmp_path / (key + "-decision.json")
        decision.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_summary": str(summary),
                    "source_run_id": payload["run_id"],
                    "selected_candidates": [system[key]],
                    "selected_by": "fixture-only",
                    "selected_date": "2026-11-13",
                    "reason": "Test fixture, not benchmark evidence",
                }
            ),
            encoding="utf-8",
        )
        decisions[key], summaries[key] = decision, summary
    lineage = {
        "chunking_embedding_candidate": system["chunking_embedding"],
        "retrieval_candidate": system["retrieval"],
        "model_lock_sha256": sha256_file(model_lock),
        "chunking_protocol_checksum": protocols["chunking_embedding"].checksum,
        "retrieval_protocol_checksum": protocols["retrieval"].checksum,
        "embedding_source_run_id": "test-only-chunking_embedding",
        "retrieval_source_run_id": "test-only-retrieval",
        "embedding_decision_sha256": sha256_file(decisions["chunking_embedding"]),
        "retrieval_decision_sha256": sha256_file(decisions["retrieval"]),
    }
    summary = json.loads(summaries["database"].read_text(encoding="utf-8"))
    summary["plan"]["settings"]["integration_provenance"] = lineage
    summaries["database"].write_text(json.dumps(summary), encoding="utf-8")
    path = tmp_path / "composition.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_by": "fixture-only",
                "selected_date": "2026-11-13",
                "reason": "Test fixture, not benchmark evidence",
                "system": system,
                "upstream_decisions": {
                    key: str(value) for key, value in decisions.items()
                },
                "protocol_checksums": {
                    key: value.checksum for key, value in protocols.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        path=path,
        protocols=protocols,
        summaries=summaries,
        lock_path=model_lock,
        lineage=lineage,
    )


@pytest.mark.parametrize(
    "field",
    [
        "chunking_embedding_candidate",
        "retrieval_candidate",
        "chunking_protocol_checksum",
        "retrieval_protocol_checksum",
        "embedding_source_run_id",
        "retrieval_source_run_id",
        "embedding_decision_sha256",
        "retrieval_decision_sha256",
    ],
)
def test_database_lineage_cannot_silently_select_another_stack(
    composition_inputs, field
):
    case = composition_inputs
    summary = json.loads(case.summaries["database"].read_text(encoding="utf-8"))
    summary["plan"]["settings"]["integration_provenance"][field] = "different"
    case.summaries["database"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="database integration"):
        load_composition(case.path, case.protocols)


def test_missing_database_lineage_is_not_legacy_compatible(composition_inputs):
    case = composition_inputs
    summary = json.loads(case.summaries["database"].read_text(encoding="utf-8"))
    summary["plan"]["settings"].pop("integration_provenance")
    case.summaries["database"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="database integration"):
        load_composition(case.path, case.protocols)


def test_database_model_lock_is_verified_not_only_candidate_aliases(composition_inputs):
    case = composition_inputs
    composition = load_composition(case.path, case.protocols)
    lock = {
        "test-embedding": {"revision": "test-embedding-revision"},
        "test-generator-model": {"revision": "test-generator-revision"},
    }
    generation = SimpleNamespace(model_id=lambda _: "test-generator-model")
    retrieval = SimpleNamespace(reranker_model=None)
    _verify_model_lineage(composition, lock, generation, retrieval, case.lock_path)
    case.lock_path.write_text('{"fixture_only": "changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="model lock differs"):
        _verify_model_lineage(composition, lock, generation, retrieval, case.lock_path)
