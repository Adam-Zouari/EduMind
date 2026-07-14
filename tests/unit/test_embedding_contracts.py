from __future__ import annotations

import numpy as np
import pytest

from edumind.rag.contracts import EmbeddingSpec, embedding_spec
from edumind.rag.embedder import Embedder
from edumind.rag.errors import RAGConfigurationError


class FakeModel:
    def __init__(self, dimension=2):
        self.dimension = dimension
        self.inputs = []

    def encode(self, texts, **kwargs):
        self.inputs.append((texts, kwargs))
        return np.ones((len(texts), self.dimension), dtype=np.float32)


def test_known_embedding_contracts_have_model_specific_prefixes() -> None:
    assert embedding_spec("BAAI/bge-base-en-v1.5").query_prefix.startswith("Represent")
    assert embedding_spec("nomic-ai/nomic-embed-text-v1.5").document_prefix == "search_document: "
    with pytest.raises(ValueError, match="No audited embedding contract"):
        embedding_spec("unknown/model")


def test_embedder_applies_query_and_document_contract(monkeypatch) -> None:
    spec = EmbeddingSpec("fake", "1", "fake", "Q: ", "D: ", True, 2, "cosine", 10)
    embedder = Embedder(spec=spec)
    fake = FakeModel()
    monkeypatch.setattr(embedder, "_model", lambda device: fake)
    embedder.embed_query("question")
    embedder.embed_texts(["document"])
    assert fake.inputs[0][0] == ["Q: question"]
    assert fake.inputs[1][0] == ["D: document"]
    assert fake.inputs[0][1]["normalize_embeddings"] is True


def test_dimension_mismatch_is_fatal(monkeypatch) -> None:
    spec = EmbeddingSpec("fake", "1", "fake", "", "", True, 3, "cosine", 10)
    embedder = Embedder(spec=spec)
    monkeypatch.setattr(embedder, "_model", lambda device: FakeModel(2))
    with pytest.raises(RAGConfigurationError, match="dimension mismatch"):
        embedder.embed_query("x")
