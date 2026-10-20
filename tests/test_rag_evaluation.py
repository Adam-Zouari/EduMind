"""Measurement boundaries in the index/search engine shared by RAG suites."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.benchmarks.rag import evaluation
from experiments.benchmarks.rag.chunking_embedding.profiles import embedding_spec
from experiments.benchmarks.rag.chunking_embedding.protocol import load_protocol


class Embedder:
    def __init__(self, spec, **_):
        self.spec = spec
        self.batch_size = 1

    def prepare(self):
        pass

    def input_token_counts(self, texts, *, role, **_):
        return [len(text.split()) + 2 for text in texts]

    def input_configuration(self, role):
        return {"role": role}

    def embed_texts(self, texts):
        return np.tile([1.0, 0.0], (len(texts), 1))

    def embed_query(self, _):
        return np.array([1.0, 0.0])


@pytest.fixture
def index(monkeypatch):
    model = "Alibaba-NLP/gte-modernbert-base"
    spec = replace(embedding_spec(model, revision="r", local_path="p"), dimension=2)
    monkeypatch.setattr(evaluation, "embedding_spec", lambda *_, **__: spec)
    monkeypatch.setattr(evaluation, "Embedder", Embedder)
    manifest = SimpleNamespace(
        samples=(
            {"kind": "document", "id": "doc", "text": "alpha beta gamma"},
            {"kind": "question", "id": "q", "answerable": True, "question": "alpha"},
        )
    )
    return evaluation.build_index(
        manifest,
        "token-256-32",
        model,
        {model: {"revision": "r", "model_path": "p"}},
        with_bm25=False,
        device="cpu",
        dtype="float32",
        chunking_protocol=load_protocol(),
    )


def test_index_phase_times_are_completed_nonoverlapping_observations(index):
    phases = index.phase_seconds
    for name in (
        "cold_model_load_seconds",
        "chunking_seconds",
        "chunk_embedding_seconds",
    ):
        assert phases[name] >= 0
    assert phases["dense_index_build_seconds"] == phases["chunk_embedding_seconds"]
    assert "bm25_index_build_seconds" not in phases
    assert index.corpus_build_seconds >= (
        phases["chunking_seconds"] + phases["chunk_embedding_seconds"]
    )


def test_query_components_finish_inside_enclosing_request(index):
    timings = {}
    started = evaluation.time.perf_counter()
    ranking = evaluation.dense_rank_with_scores(index, "alpha", 20, timings=timings)
    elapsed_ms = (evaluation.time.perf_counter() - started) * 1000
    assert ranking == [(0, 1.0)]
    assert set(timings) == {"query_embedding_latency_ms", "vector_search_latency_ms"}
    assert 0 <= sum(timings.values()) <= elapsed_ms


def test_completed_embedding_time_survives_later_search_failure(index):
    index.vectors[:] = np.nan
    timings = {}
    with pytest.raises(
        evaluation.OutputIntegrityError, match="scores contain non-finite"
    ):
        evaluation.dense_rank_with_scores(index, "alpha", 20, timings=timings)
    assert timings["query_embedding_latency_ms"] >= 0
    assert "vector_search_latency_ms" not in timings
    assert timings["vector_search_elapsed_to_failure_ms"] >= 0


def test_embedding_failure_does_not_invent_completed_component_timings(
    index, monkeypatch
):
    def fail(_):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(index.embedder, "embed_query", fail)
    timings = {}
    with pytest.raises(RuntimeError, match="inference failed"):
        evaluation.dense_rank_with_scores(index, "alpha", 20, timings=timings)
    assert set(timings) == {"query_embedding_elapsed_to_failure_ms"}
    assert timings["query_embedding_elapsed_to_failure_ms"] >= 0


@pytest.mark.parametrize("phase", ["loading", "chunking", "embedding", "assembly"])
def test_build_failure_keeps_exact_entered_phase_and_completed_predecessors(
    index, monkeypatch, phase
):
    def fail(*_, **__):
        raise RuntimeError("injected phase failure")

    if phase == "loading":
        monkeypatch.setattr(Embedder, "prepare", fail)
    elif phase == "chunking":
        monkeypatch.setattr(evaluation, "build_chunking_strategy", fail)
    elif phase == "embedding":
        monkeypatch.setattr(Embedder, "embed_texts", fail)
    else:
        monkeypatch.setattr(Embedder, "placement_report", fail, raising=False)
    progress = {}
    model = "Alibaba-NLP/gte-modernbert-base"
    with pytest.raises(RuntimeError, match="injected phase failure"):
        evaluation.build_index(
            SimpleNamespace(
                samples=({"kind": "document", "id": "doc", "text": "alpha beta"},)
            ),
            "token-256-32",
            model,
            {model: {"revision": "r", "model_path": "p"}},
            with_bm25=False,
            device="cpu",
            dtype="float32",
            chunking_protocol=load_protocol(),
            progress=progress,
        )
    interrupted = progress.get("phase_elapsed_to_failure_seconds", {})
    key = {
        "loading": "cold_model_load_seconds",
        "chunking": "chunking_seconds",
        "embedding": "chunk_embedding_seconds",
    }.get(phase)
    assert set(interrupted) == ({key} if key else set())
    assert all(value >= 0 for value in interrupted.values())
    assert (
        "build_elapsed_to_failure" not in progress
        if phase == "loading"
        else progress["build_elapsed_to_failure"] >= 0
    )
    completed = progress["phase_seconds"]
    if phase != "loading":
        assert completed["cold_model_load_seconds"] >= 0
    if phase in {"embedding", "assembly"}:
        assert completed["chunking_seconds"] >= 0
    if phase == "assembly":
        assert completed["chunk_embedding_seconds"] >= 0


def test_malformed_native_counts_are_input_contract_errors_not_truncation(
    index, monkeypatch
):
    monkeypatch.setattr(index.embedder, "input_token_counts", lambda *_, **__: [True])
    with pytest.raises(evaluation.InputCompatibilityError) as captured:
        evaluation.dense_rank_with_scores(index, "alpha", 20)
    assert captured.value.report["reason_code"] == "invalid_native_token_counts"
    assert captured.value.report["truncated_inputs"] == 0
