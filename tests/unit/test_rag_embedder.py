from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from edumind.rag.embedder import Embedder
from edumind.rag.errors import RAGConfigurationError
from edumind.rag.types import EmbeddingSettings


def _install_fake_sentence_transformers(monkeypatch, output_dim: int) -> None:
    class FakeModel:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

        def encode(
            self,
            texts: list[str],
            *,
            convert_to_numpy: bool,
            show_progress_bar: bool,
            batch_size: int,
        ) -> np.ndarray:
            del convert_to_numpy, show_progress_bar, batch_size
            return np.array(
                [[float(index + 1)] * output_dim for index, _ in enumerate(texts)],
                dtype=float,
            )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeModel),
    )


def test_embed_text_blank_input_skips_model_loading() -> None:
    embedder = Embedder(settings=EmbeddingSettings("fake-model", embedding_dim=3))

    embedding = embedder.embed_text("")

    assert np.array_equal(embedding, np.zeros(3, dtype=float))
    assert embedder.model_loaded is False


def test_embed_texts_loads_model_and_returns_embeddings(monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, output_dim=3)
    embedder = Embedder(settings=EmbeddingSettings("fake-model", embedding_dim=3))

    embeddings = embedder.embed_texts(["alpha", "beta"])

    assert embeddings.shape == (2, 3)
    assert embedder.model_loaded is True


def test_embed_texts_validates_embedding_dimension(monkeypatch) -> None:
    _install_fake_sentence_transformers(monkeypatch, output_dim=4)
    embedder = Embedder(settings=EmbeddingSettings("fake-model", embedding_dim=3))

    with pytest.raises(RAGConfigurationError, match="Configured embedding_dim"):
        embedder.embed_texts(["alpha"])
