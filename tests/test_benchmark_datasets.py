from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.common.contracts import SampleResult
from benchmarks.common.data_validation import manifest_path
from benchmarks.common.datasets import (
    DatasetValidationError,
    assert_no_split_leakage,
    load_manifest,
    require_manifest_split,
    validate_evidence,
)
from benchmarks.preparation.fixtures import _write_minimal_docx
from benchmarks.rag.chunking_embedding.metrics import aggregate_quality, score_question
from benchmarks.rag.evaluation import Chunk
from edumind.common.paths import PROJECT_ROOT
from edumind.rag.tokenizers import TiktokenOffsetTokenizer


def test_smoke_dataset_has_thirty_documents_24_questions_and_exact_offsets() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    assert sum(item.get("kind") == "document" for item in manifest.samples) == 30
    assert sum(item.get("kind") == "question" for item in manifest.samples) == 24


def test_manifest_checksum_tampering_is_detected(tmp_path) -> None:
    source = PROJECT_ROOT / "data/benchmarks/rag/smoke.json"
    content = source.read_text(encoding="utf-8").replace(
        "chemical energy", "changed", 1
    )
    path = tmp_path / "tampered.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="checksum"):
        load_manifest(path)


def test_manifest_split_must_match_the_requested_profile() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    require_manifest_split(manifest, "smoke", "smoke")
    with pytest.raises(
        DatasetValidationError, match="Profile development requires split"
    ):
        require_manifest_split(manifest, "development", {"dev", "development"})


def test_document_level_leakage_is_rejected() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    with pytest.raises(DatasetValidationError, match="leakage"):
        assert_no_split_leakage([manifest, replace(manifest, split="other")])


def test_near_duplicate_document_leakage_is_rejected() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    document = next(item for item in manifest.samples if item.get("kind") == "document")
    altered = {
        **document,
        "id": "near-copy",
        "text": str(document["text"]) + " extra",
    }
    second = replace(manifest, name="near", split="other", samples=(altered,))
    with pytest.raises(DatasetValidationError, match="Near-duplicate"):
        assert_no_split_leakage([manifest, second])


def test_generated_docx_fixture_is_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first.docx"
    second = tmp_path / "second.docx"
    _write_minimal_docx(first, "Heading\nDeterministic content")
    _write_minimal_docx(second, "Heading\nDeterministic content")
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    "benchmark", ["chunking-embedding", "retrieval-reranking", "generation"]
)
def test_rag_validators_resolve_the_same_frozen_corpus_inventory(benchmark):
    fixture = "generation-smoke" if benchmark == "generation" else "smoke"
    assert (
        manifest_path(benchmark, "smoke")
        == PROJECT_ROOT / f"data/benchmarks/rag/{fixture}.json"
    )
    assert manifest_path(benchmark, "development").name == "rag-selection-dev.json"
    assert (
        manifest_path(benchmark, "validation").name == "rag-selection-validation.json"
    )
    assert manifest_path(benchmark, "locked").name == "rag-selection-locked-test.json"


def test_linked_rag_questions_keep_provenance_and_one_vote_per_draw():
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/generation-smoke.json")
    documents = [
        {
            "kind": "document",
            "id": name,
            "text": "alpha",
            "independent_source_id": family,
        }
        for name, family in (
            ("a", "A"),
            ("a-copy", "A"),
            ("b", "B"),
            ("c", "C"),
            ("d", "D"),
        )
    ]

    def question(name, *sources):
        return {
            "kind": "question",
            "id": name,
            "question": "What?",
            "answerable": True,
            "evidence_type": "text",
            "evidence": [
                {
                    "id": f"unit-{source}",
                    "evidence_type": "text",
                    "document_id": source,
                    "intervals": [{"start": 0, "end": 5}],
                }
                for source in sources
            ],
        }

    def unanswerable(name, *reviewed):
        return {
            "kind": "question",
            "id": name,
            "question": "What else?",
            "answerable": False,
            "evidence": [],
            "reviewed_document_ids": list(reviewed),
        }

    multiple = question("a-b", "a", "b")
    questions = [
        multiple,
        question("a-single", "a-copy"),
        question("c-single", "c"),
        question("d-single", "d"),
        unanswerable("b-c", "b", "c"),
        unanswerable("unknown"),
    ]
    manifest = replace(manifest, samples=(*documents, *questions))
    sources = validate_evidence(manifest)
    shared = sources["a-b"].independent_source_id
    assert shared is not None
    assert shared == sources["a-single"].independent_source_id
    assert (
        shared
        == sources["b-c"].independent_source_id
        == sources["c-single"].independent_source_id
    )
    assert shared != sources["d-single"].independent_source_id
    assert sources["unknown"].independent_source_id is None
    assert sources["a-b"].document_ids == ("a", "b")

    corpus = [Chunk(doc["id"], doc["id"], "alpha", 0, 5, 1) for doc in documents]
    tokenizer = TiktokenOffsetTokenizer("cl100k_base")
    score = score_question(
        multiple, [corpus[1], corpus[2]], corpus, tokenizer, cutoffs=(3,), alpha=0.5
    )
    assert score.metrics["evidence_unit_recall_at_3"] == 0.5
    assert score.metrics["evidence_token_precision_at_3"] == 0.5

    overall, single, multi = [
        f"quality.{scope}.evidence_unit_recall_at_3"
        for scope in ("overall", "single_source", "multi_source")
    ]
    rows = [
        SampleResult(
            row["id"],
            {overall: float(i != 3), (multi if i == 0 else single): float(i != 3)},
            None,
            {**sources[row["id"]].metadata(), "evidence_type": "text"},
        )
        for i, row in enumerate(questions[:4])
    ]
    metrics, bounds = aggregate_quality(
        rows, resamples=100, seed=42, confidence=0.95, minimum_sources=2
    )
    assert metrics[overall] == 0.75
    assert metrics[single] == pytest.approx(2 / 3)
    assert metrics[multi] == 1
    assert metrics[overall + ".sample_count"] == 4
    assert bounds[overall]["eligible_document_count"] == 5
    assert bounds[overall]["independent_source_count"] == 2
    assert bounds[overall]["lower"] == 0 and bounds[overall]["upper"] == 1
    rows[0] = replace(
        rows[0], metadata={**rows[0].metadata, "independent_source_id": None}
    )
    metrics, bounds = aggregate_quality(
        rows, resamples=100, seed=42, confidence=0.95, minimum_sources=2
    )
    assert metrics[overall] == 0.75
    assert bounds[overall]["reason"] == "unknown_independent_source_grouping"
    assert bounds[overall]["lower"] is None


@pytest.mark.parametrize("stage", ["chunking_embedding", "retrieval_reranking"])
def test_component_preflight_runs_new_schema_multi_source_stress(
    tmp_path, monkeypatch, stage
):
    import json
    from importlib import import_module
    from types import SimpleNamespace

    from benchmarks.common.datasets import manifest_content_checksum
    from benchmarks.common.preflight import rag_stress_manifest, stress_input_identity

    rows = [
        {"kind": "document", "id": key, "text": text, "independent_source_id": key}
        for key, text in (("a", "one " * 30), ("b", "two " * 30), ("other", "three"))
    ]
    for qid, refs in (("single", ("other",)), ("linked", ("a", "b"))):
        rows.append(
            {
                "kind": "question",
                "id": qid,
                "question": "What is stated?",
                "answerable": True,
                "evidence_type": "text",
                "evidence": [
                    {
                        "id": f"{qid}-{doc}",
                        "evidence_type": "text",
                        "document_id": doc,
                        "intervals": [{"start": 0, "end": 3}],
                    }
                    for doc in refs
                ],
            }
        )
    base = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    path = tmp_path / "development.json"
    path.write_text(
        json.dumps(
            {
                **vars(base),
                "split": "development",
                "samples": rows,
                "checksum": manifest_content_checksum(rows),
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(path)
    stress = rag_stress_manifest(manifest)
    assert [row["id"] for row in stress.samples] == ["a", "b", "linked"]
    assert stress.checksum == manifest_content_checksum(stress.samples)
    assert stress.checksum != manifest.checksum
    assert validate_evidence(stress)["linked"].document_ids == ("a", "b")

    runner = import_module(f"benchmarks.rag.{stage}.run")
    monkeypatch.setattr(runner, "verify_inputs", lambda *a, **kw: (path, {}))
    monkeypatch.setattr(runner, "load_selected_model_lock", lambda *a, **kw: {})
    if stage == "retrieval_reranking":
        monkeypatch.setattr(
            runner,
            "_selected_embedding",
            lambda p, source, proto: (proto.smoke_pair, None),
        )

    def qualification(**options):
        expected = stress_input_identity(manifest, stress.samples)
        assert all(
            options["input_envelope"][key] == value for key, value in expected.items()
        )
        return "frozen-stress", {}

    probed = []

    def worker(candidate, input_manifest, models, plan, **kwargs):
        probed.append(input_manifest)
        return {}

    def launch(**options):
        options["probe"](options["candidates"][0])
        assert options["context"]["stress_manifest_fingerprint"] == stress.fingerprint
        return SimpleNamespace(
            run_id="test",
            ready_for_development=True,
            qualified_candidates=[],
            excluded_candidates=[],
            blocked_candidates=[],
            artifact_directory=tmp_path,
        )

    monkeypatch.setattr(runner, "current_qualification_fingerprint", qualification)
    monkeypatch.setattr(runner, "preflight_in_fresh_process", worker)
    monkeypatch.setattr(runner, "run_preflight", launch)
    assert (
        runner.main(["--profile", "preflight", "--manifest", str(path), "--no-mlflow"])
        == 0
    )
    assert probed == [stress]
