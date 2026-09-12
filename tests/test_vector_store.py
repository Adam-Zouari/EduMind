from types import SimpleNamespace

import pytest

from edumind.rag.errors import RAGConfigurationError
from edumind.rag.vector_store import VectorStore
import edumind.rag.vector_store as vector_store_module


class NotFoundError(Exception):
    pass


def _store(client) -> VectorStore:
    store = object.__new__(VectorStore)
    store.client = client
    store.collection_name = "documents"
    store.distance_metric = "cosine"
    store.collection = "old"
    store._manifest = "old-manifest"
    return store


def _fake_chromadb():
    return SimpleNamespace(errors=SimpleNamespace(NotFoundError=NotFoundError))


def test_reset_collection_ignores_only_a_missing_collection(monkeypatch) -> None:
    fresh = object()

    class Client:
        def delete_collection(self, *, name):
            assert name == "documents"
            raise NotFoundError

        def get_or_create_collection(self, *, name, metadata):
            assert name == "documents"
            assert metadata == {"hnsw:space": "cosine"}
            return fresh

    monkeypatch.setattr(vector_store_module, "_load_chromadb", _fake_chromadb)
    store = _store(Client())

    store.reset_collection()

    assert store.collection is fresh
    assert store._manifest is None


def test_reset_collection_does_not_reuse_data_after_delete_failure(monkeypatch) -> None:
    class Client:
        recreated = False

        def delete_collection(self, *, name):
            assert name == "documents"
            raise RuntimeError("server unavailable")

        def get_or_create_collection(self, **_kwargs):
            self.recreated = True

    monkeypatch.setattr(vector_store_module, "_load_chromadb", _fake_chromadb)
    client = Client()
    store = _store(client)

    with pytest.raises(RAGConfigurationError, match="Could not reset"):
        store.reset_collection()

    assert client.recreated is False
    assert store.collection == "old"
    assert store._manifest == "old-manifest"
