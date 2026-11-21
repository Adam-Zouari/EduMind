import csv
import json
from pathlib import Path

import pytest

from benchmarks.common.selection import (
    included_candidates,
    selection_entries,
)
from benchmarks.extraction.audio.protocol import (
    load_protocol as load_audio_protocol,
)
from benchmarks.extraction.document.profiles import DOCUMENT_LOCK_CANDIDATES
from benchmarks.preparation.models import (
    DOCLING_BENCHMARK_COMPONENTS,
    EMBEDDING_COMPONENTS,
    EXTRACTION_COMPONENTS,
    MODEL_COMPONENTS,
    RAG_COMPONENTS,
    preparation_plan,
    selected_model_names,
)
from benchmarks.rag.chunking_embedding.profiles import (
    EXPERIMENTAL_EMBEDDING_SPECS,
)
from benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as load_chunking_protocol,
)
from benchmarks.rag.generation.protocol import (
    load_protocol as load_generation_protocol,
)
from benchmarks.rag.retrieval_reranking.profiles import (
    development_candidates,
    required_reranker_models,
)
from edumind.common.config import ConfigurationError, load_settings
from edumind.common.model_assets import (
    INFERENCE_SNAPSHOT_IGNORE_PATTERNS,
    snapshot_ignore_patterns,
)
from edumind.extraction.pipeline import LOCK_CANDIDATE_BY_ENGINE
from edumind.prepare_models import APP_MODELS
from edumind.rag.contracts import EMBEDDING_SPECS, PRODUCTION_EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[1]


def test_gte_control_has_reviewed_selection_evidence() -> None:
    with (ROOT / "benchmarks/selection_evidence.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = csv.DictReader(handle)
        gte = next(
            row
            for row in rows
            if (row["component"], row["candidate"])
            == ("embedding", PRODUCTION_EMBEDDING_MODEL)
        )
    assert gte["purpose"] == "control"
    assert gte["decision"] == "include"
    assert gte["approx_params_b"] == "0.149"
    assert gte["public_benchmark"] == "BEIR (15)"
    assert gte["public_metric"] == "Avg nDCG@10"
    assert gte["public_score"] == "55.33"
    assert gte["candidate_revision"] == ("e7f32e3c00f91d699e8c43b53106206bcc72bb22")
    assert gte["license"] == "Apache-2.0"
    assert gte["reviewed_date"] == "2026-09-12"


def test_executable_model_registries_match_approved_selection() -> None:
    approved_embeddings = set(included_candidates("embedding"))
    assert set(EMBEDDING_SPECS) == {PRODUCTION_EMBEDDING_MODEL}
    assert (
        set(EMBEDDING_SPECS) | set(EXPERIMENTAL_EMBEDDING_SPECS) == approved_embeddings
    )
    assert all(
        spec.revision == "from-lock"
        for spec in [*EMBEDDING_SPECS.values(), *EXPERIMENTAL_EMBEDDING_SPECS.values()]
    )

    chunk_pairs = load_chunking_protocol().development_candidates
    assert len(chunk_pairs) == 48
    assert len({pair.split("|", 1)[0] for pair in chunk_pairs}) == 8
    assert {pair.split("|", 1)[1] for pair in chunk_pairs} == approved_embeddings

    assert set(load_generation_protocol().models.values()) == set(
        included_candidates("generator")
    )


def test_gte_production_contract_and_fixed_chunking_tokenizer() -> None:
    spec = EMBEDDING_SPECS[PRODUCTION_EMBEDDING_MODEL]
    assert spec.revision == "from-lock"
    assert spec.tokenizer == PRODUCTION_EMBEDDING_MODEL
    assert spec.dimension == 768
    assert spec.maximum_length == 8192
    assert spec.pooling == "cls"
    assert spec.similarity == "cosine"
    assert spec.normalize is True
    assert spec.query_prefix == spec.document_prefix == ""
    assert spec.trust_remote_code is False

    settings = load_settings(ROOT / "config/base.yaml")
    assert settings.embedding.model_name == PRODUCTION_EMBEDDING_MODEL
    assert settings.chunking.tokenizer == "cl100k_base"
    assert PRODUCTION_EMBEDDING_MODEL in APP_MODELS

    with pytest.raises(ConfigurationError, match="frozen 'cl100k_base'"):
        load_settings(
            ROOT / "config/base.yaml",
            overrides={"chunking": {"tokenizer": "embedding"}},
        )


def test_embedding_docs_reference_approved_models_and_tokenizer() -> None:
    docs = ROOT / "docs/benchmarks"
    selection = (docs / "model-selection.md").read_text(encoding="utf-8")
    for candidate in included_candidates("embedding"):
        assert candidate in selection, candidate
    for guide in ("methodology.md", "metrics.md"):
        path = docs / "rag/chunking_embedding" / guide
        assert "tiktoken:cl100k_base" in path.read_text(encoding="utf-8"), path


def test_retrieval_reranker_models_match_approved_selection() -> None:
    assert set(required_reranker_models(development_candidates())) == set(
        included_candidates("reranker")
    )


def test_audio_model_ids_match_approved_selection() -> None:
    assert {
        candidate.model_id for candidate in load_audio_protocol().candidates.values()
    } == set(included_candidates("asr"))


def test_document_lock_registries_match_approved_selection() -> None:
    assert set(DOCUMENT_LOCK_CANDIDATES.values()) == {
        "docling-standard",
        *included_candidates("document_extraction"),
    }
    assert set(DOCUMENT_LOCK_CANDIDATES) == {
        "docling-standard",
        "docling-vlm-granite-258m",
        "paddleocr-vl-1.6",
    }
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
    assert set(selected_model_names(EMBEDDING_COMPONENTS)) == set(
        included_candidates("embedding")
    )
    assert set(selected_model_names(RAG_COMPONENTS)) <= approved
    assert set(selected_model_names(EXTRACTION_COMPONENTS)) <= approved
    plan = preparation_plan(selected, DOCLING_BENCHMARK_COMPONENTS)
    assert {str(item["candidate"]) for item in plan} == approved | {"docling-standard"}


def test_embedding_and_reranker_preparation_excludes_unused_exports() -> None:
    assert "onnx/**" in INFERENCE_SNAPSHOT_IGNORE_PATTERNS
    assert "openvino/**" in INFERENCE_SNAPSHOT_IGNORE_PATTERNS
    assert "*.gguf" in INFERENCE_SNAPSHOT_IGNORE_PATTERNS
    assert snapshot_ignore_patterns("embedding") == INFERENCE_SNAPSHOT_IGNORE_PATTERNS
    assert snapshot_ignore_patterns("reranker") == INFERENCE_SNAPSHOT_IGNORE_PATTERNS
    assert snapshot_ignore_patterns("generator") == ()


def test_snapshot_size_counts_nested_files_without_temporary_caches(tmp_path) -> None:
    from edumind.common.model_assets import _snapshot_size_bytes

    (tmp_path / "weights.bin").write_bytes(b"weights")
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer/vocab.json").write_bytes(b"vocab")
    for cache in (tmp_path / ".cache/huggingface", tmp_path / "__pycache__"):
        cache.mkdir(parents=True)
        (cache / "temporary").write_bytes(b"excluded")
    assert _snapshot_size_bytes(tmp_path) == 12


@pytest.mark.parametrize("directory_state", ["missing", "empty", "cache_only"])
def test_snapshot_size_rejects_missing_or_empty_snapshots(
    tmp_path, directory_state
) -> None:
    from edumind.common.model_assets import _snapshot_size_bytes

    root = tmp_path / "model"
    if directory_state != "missing":
        root.mkdir()
    if directory_state == "cache_only":
        (root / ".cache").mkdir()
        (root / ".cache/download.lock").write_bytes(b"not a snapshot")
    with pytest.raises(RuntimeError, match="missing|empty"):
        _snapshot_size_bytes(root)


@pytest.mark.parametrize("candidate", [PRODUCTION_EMBEDDING_MODEL, "docling-standard"])
def test_model_lock_writes_measured_size_and_preserves_existing_entries(
    tmp_path, candidate
) -> None:
    from benchmarks.preparation.models import (
        _directory_manifest_hash,
        _merge_model_lock,
    )

    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"pinned")
    path = tmp_path / "selected.json"
    previous = {
        "revision": "previous",
        "model_path": "unavailable-but-unrequested",
        # Structurally valid; an unrequested directory is deliberately not read.
        "model_cache_manifest_sha256": "0" * 64,
    }
    path.write_text(
        json.dumps({"models": {"openai/whisper-small.en": previous}}),
        encoding="utf-8",
    )
    entry = {
        "revision": "unchanged",
        "model_path": str(root),
        "model_cache_manifest_sha256": _directory_manifest_hash(root),
        "snapshot_size_bytes": 999,
    }
    _merge_model_lock(path, {candidate: entry})
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    assert models[candidate] == {**entry, "snapshot_size_bytes": 6}
    assert models["openai/whisper-small.en"] == previous
    assert entry["snapshot_size_bytes"] == 999  # Do not mutate the caller's input.

    (root / "weights.bin").write_bytes(b"updated")
    entry["model_cache_manifest_sha256"] = _directory_manifest_hash(root)
    _merge_model_lock(path, {candidate: entry})
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    assert models[candidate]["snapshot_size_bytes"] == 7


def test_document_stage_requests_only_its_candidate_models(monkeypatch) -> None:
    from benchmarks.extraction.document import benchmark

    requested = None

    def capture(_path, *, candidates=None):
        nonlocal requested
        requested = candidates
        return {}

    monkeypatch.setattr(benchmark, "load_selected_model_lock", capture)
    assert (
        benchmark._model_lock(
            (
                "docling-standard-native",
                "docling-vlm-granite-258m",
                "docling-vlm-granite-258m",
            )
        )
        == {}
    )
    assert requested == (
        "docling-standard",
        "ibm-granite/granite-docling-258M",
    )


def test_torch_and_torchaudio_lock_versions_match() -> None:
    pins = {}
    for line in (
        (ROOT / "requirements/app.lock").read_text(encoding="utf-8").splitlines()
    ):
        if "==" in line and not line.startswith("#"):
            name, value = line.split("==", 1)
            pins[name.casefold()] = value
    assert pins["torch"] == pins["torchaudio"] == "2.11.0"
