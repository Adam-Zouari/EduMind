import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.benchmarks.common.arguments import load_candidates
from experiments.benchmarks.common.selection import included_candidates, selection_entries
from experiments.benchmarks.preparation.models import (
    DOCLING_BENCHMARK_COMPONENTS,
    EXTRACTION_COMPONENTS,
    MODEL_COMPONENTS,
    RAG_COMPONENTS,
    preparation_plan,
    load_selected_model_lock,
    selected_model_names,
    snapshot_specs,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import (
    EXPERIMENTAL_EMBEDDING_SPECS,
)
from experiments.benchmarks.rag.generation.models import GENERATOR_PROFILES
from edumind.rag.contracts import EMBEDDING_SPECS
from edumind.extraction.pipeline import LOCK_CANDIDATE_BY_ENGINE


ROOT = Path(__file__).resolve().parents[1]


def test_selection_history_counts_and_keys_are_preserved() -> None:
    with (ROOT / "experiments/benchmarks/selection_evidence.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 66
    assert len({(row["component"], row["candidate"]) for row in rows}) == 66
    assert Counter(row["decision"] for row in rows) == {
        "include": 29,
        "exclude": 37,
    }


def test_executable_model_registries_match_approved_selection() -> None:
    approved_embeddings = set(included_candidates("embedding"))
    assert set(EMBEDDING_SPECS) == {"sentence-transformers/all-MiniLM-L6-v2"}
    assert set(EMBEDDING_SPECS) | set(EXPERIMENTAL_EMBEDDING_SPECS) == approved_embeddings
    assert all(
        spec.revision == "from-lock"
        for spec in [*EMBEDDING_SPECS.values(), *EXPERIMENTAL_EMBEDDING_SPECS.values()]
    )

    chunk_pairs = load_candidates(
        ROOT / "experiments/benchmarks/rag/chunking_embedding/candidates.yaml",
        "standard",
    )
    assert len(chunk_pairs) == 64
    assert len({pair.split("|", 1)[0] for pair in chunk_pairs}) == 8
    assert {pair.split("|", 1)[1] for pair in chunk_pairs} == approved_embeddings

    assert {model for model, _ in GENERATOR_PROFILES.values()} == set(
        included_candidates("generator")
    )


def test_reranker_audio_and_document_registries_are_exact() -> None:
    retrieval = set(
        load_candidates(
            ROOT / "experiments/benchmarks/rag/retrieval/candidates.yaml", "standard"
        )
    )
    assert retrieval == {
        "dense",
        "bm25",
        "rrf",
        "rrf-minilm-reranker",
        "rrf-ettin-150m-reranker",
        "rrf-ettin-400m-reranker",
        "rrf-ettin-1b-reranker",
        "rrf-qwen3-4b-reranker",
    }
    approved_asr = {
        "whisper-small-en-control",
        "canary-180m",
        "parakeet-tdt-0.6b-v2",
        "moss-transcribe-diarize",
        "qwen3-asr-1.7b-aligned",
    }
    audio_registry = ROOT / "experiments/benchmarks/extraction/audio/candidates.yaml"
    assert set(load_candidates(audio_registry, "smoke")) == approved_asr
    assert set(load_candidates(audio_registry, "standard")) == approved_asr
    document = load_candidates(
        ROOT / "experiments/benchmarks/extraction/document/candidates.yaml", "standard"
    )
    assert len(document) == len(set(document)) == 24
    assert LOCK_CANDIDATE_BY_ENGINE["docling-vlm-granite-258m"] == (
        "ibm-granite/granite-docling-258M"
    )
    assert LOCK_CANDIDATE_BY_ENGINE["paddleocr-vl-1.6"] == (
        "PaddlePaddle/PaddleOCR-VL-1.6"
    )


def test_preparation_plan_contains_only_approved_models_and_docling() -> None:
    approved = {
        entry.candidate
        for entry in selection_entries()
        if entry.component in MODEL_COMPONENTS
    }
    selected = selected_model_names(MODEL_COMPONENTS)
    assert set(selected) == approved
    assert set(selected_model_names(RAG_COMPONENTS)) <= approved
    assert set(selected_model_names(EXTRACTION_COMPONENTS)) <= approved
    plan = preparation_plan(selected, DOCLING_BENCHMARK_COMPONENTS)
    assert {str(item["candidate"]) for item in plan} == approved | {"docling-standard"}


def test_stage_model_lock_ignores_unrequested_missing_models(tmp_path) -> None:
    entries = {entry.candidate: entry for entry in selection_entries()}
    requested = entries["openai/whisper-small.en"]
    unrelated = entries["sentence-transformers/all-MiniLM-L6-v2"]
    requested_directory = tmp_path / "whisper"
    requested_directory.mkdir()

    def lock_entry(entry, model_path):
        repository, revision, _ = snapshot_specs(entry)[0]
        return {
            "model": repository,
            "revision": revision,
            "selection_revision": entry.revision,
            "model_path": str(model_path),
        }

    path = tmp_path / "selected.json"
    path.write_text(
        json.dumps(
            {
                "models": {
                    requested.candidate: lock_entry(requested, requested_directory),
                    unrelated.candidate: lock_entry(unrelated, tmp_path / "missing"),
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_selected_model_lock(path, candidates=(requested.candidate,))
    assert set(loaded) == {requested.candidate}


def test_document_stage_requests_only_its_candidate_models(monkeypatch) -> None:
    from experiments.benchmarks.extraction.document import benchmark

    requested = None

    def capture(_path, *, candidates=None):
        nonlocal requested
        requested = candidates
        return {}

    monkeypatch.setattr(benchmark, "load_selected_model_lock", capture)
    assert benchmark._model_lock(
        (
            "docling-standard-native|ocr=rapidocr|mode=full_page|table=fast|formula=off",
            "docling-vlm-granite-258m",
            "docling-vlm-granite-258m",
        )
    ) == {}
    assert requested == (
        "docling-standard",
        "ibm-granite/granite-docling-258M",
    )


def test_schema_three_model_lock_verifies_the_complete_cache_manifest(tmp_path) -> None:
    from experiments.benchmarks.preparation.models import _directory_manifest_hash

    entry = next(
        item for item in selection_entries() if item.candidate == "openai/whisper-small.en"
    )
    repository, revision, _ = snapshot_specs(entry)[0]
    model_path = tmp_path / "model"
    model_path.mkdir()
    weights = model_path / "weights.bin"
    weights.write_bytes(b"pinned")
    path = tmp_path / "selected.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "models": {
                    entry.candidate: {
                        "model": repository,
                        "revision": revision,
                        "selection_revision": entry.revision,
                        "model_path": str(model_path),
                        "model_cache_manifest_sha256": _directory_manifest_hash(model_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_selected_model_lock(path, candidates=(entry.candidate,))
    weights.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="cache manifest checksum mismatch"):
        load_selected_model_lock(path, candidates=(entry.candidate,))


def test_torch_and_torchaudio_lock_versions_match() -> None:
    pins = {}
    for line in (ROOT / "requirements/app.lock").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, value = line.split("==", 1)
            pins[name.casefold()] = value
    assert pins["torch"] == pins["torchaudio"] == "2.11.0"
