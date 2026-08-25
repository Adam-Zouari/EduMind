import csv
from collections import Counter
from pathlib import Path

from experiments.benchmarks.common.arguments import load_candidates
from experiments.benchmarks.common.selection import included_candidates, selection_entries
from experiments.benchmarks.preparation.models import (
    EXTRACTION_COMPONENTS,
    MODEL_COMPONENTS,
    RAG_COMPONENTS,
    preparation_plan,
    selected_model_names,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import (
    EXPERIMENTAL_EMBEDDING_SPECS,
)
from experiments.benchmarks.rag.generation.models import GENERATOR_PROFILES
from edumind.rag.contracts import EMBEDDING_SPECS


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
    assert set(
        load_candidates(
            ROOT / "experiments/benchmarks/extraction/audio/candidates.yaml", "standard"
        )
    ) == {
        "whisper-small-en-control",
        "canary-180m",
        "parakeet-tdt-0.6b-v2",
        "moss-transcribe-diarize",
        "qwen3-asr-1.7b-aligned",
    }
    document = load_candidates(
        ROOT / "experiments/benchmarks/extraction/document/candidates.yaml", "standard"
    )
    assert len(document) == len(set(document)) == 24


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
    plan = preparation_plan(selected, include_docling=True)
    assert {str(item["candidate"]) for item in plan} == approved | {"docling-standard"}
