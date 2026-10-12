from types import SimpleNamespace

import numpy as np
import pytest

import edumind.rag.vector_store as vector_store_module
from edumind.rag.contracts import IndexManifest
from edumind.rag.errors import IndexCompatibilityError, RAGConfigurationError
from edumind.rag.vector_store import COMPATIBILITY, VectorStore


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

        def get_or_create_collection(self, *, name, configuration, embedding_function):
            assert name == "documents"
            assert configuration == {"hnsw": {"space": "cosine"}}
            assert embedding_function is None
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


@pytest.mark.parametrize("legacy_metadata", [{}, {"hnsw:space": "cosine"}])
def test_manifest_initialization_preserves_index_configuration(legacy_metadata) -> None:
    from chromadb.api.models.CollectionCommon import CollectionCommon

    class Collection:
        def __init__(self):
            self.metadata = legacy_metadata
            self.configuration = {"hnsw": {"space": "cosine"}}

        def count(self):
            return 0

        def modify(self, *, metadata):
            CollectionCommon._validate_modify_request(self, metadata)
            self.metadata = metadata

    store = _store(None)
    store.collection = Collection()
    manifest = IndexManifest(1, "embedding", "chunking", "chroma", "documents")

    assert store.ensure_manifest(manifest) == manifest
    assert store.collection.metadata == {COMPATIBILITY: manifest.compatibility_key}
    assert store.collection.configuration == {"hnsw": {"space": "cosine"}}
    assert store.ensure_manifest(manifest) == manifest


@pytest.mark.parametrize(
    ("metadata", "count", "space"),
    [({}, 1, "cosine"), ({COMPATIBILITY: "old"}, 0, "cosine"), ({}, 0, "l2")],
)
def test_manifest_rejects_unverified_or_incompatible_indexes(metadata, count, space):
    store = _store(None)
    store.collection = SimpleNamespace(
        metadata=metadata,
        count=lambda: count,
        configuration={"hnsw": {"space": space}},
    )
    manifest = IndexManifest(1, "embedding", "chunking", "chroma", "documents")

    with pytest.raises(IndexCompatibilityError):
        store.ensure_manifest(manifest)


def test_failed_document_replacement_restores_numpy_embeddings(monkeypatch) -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    previous = {
        "ids": ["one", "two"],
        "documents": ["old one", "old two"],
        "metadatas": [{"source": "old"}, {"source": "old"}],
        "embeddings": vectors,
    }
    restored = []
    store = _store(None)
    store.collection = SimpleNamespace(
        get=lambda **kwargs: previous,
        delete=lambda **kwargs: None,
        upsert=lambda **kwargs: restored.append(kwargs),
    )

    def fail(_chunks):
        raise RuntimeError("replacement failed")

    monkeypatch.setattr(store, "_upsert", fail)

    with pytest.raises(RuntimeError, match="replacement failed"):
        store.replace_document("old", [])

    assert len(restored) == 1
    assert restored[0]["embeddings"] is vectors
    assert restored[0]["ids"] == previous["ids"]


def test_dot_product_collection_uses_chromas_inner_product_space() -> None:
    calls = []
    store = _store(
        SimpleNamespace(get_or_create_collection=lambda **kwargs: calls.append(kwargs))
    )
    store.distance_metric = "dot"

    store._get_or_create_collection()

    assert calls == [
        {
            "name": "documents",
            "configuration": {"hnsw": {"space": "ip"}},
            "embedding_function": None,
        }
    ]


@pytest.mark.parametrize("metric", ["cosine", "dot"])
def test_dense_hit_score_converts_chroma_distance(metric):
    store = _store(None)
    store.distance_metric = metric
    store.collection = SimpleNamespace(
        query=lambda **kwargs: {
            "ids": [["chunk"]],
            "documents": [["text"]],
            "metadatas": [[{}]],
            "distances": [[0.25]],
        }
    )
    assert store.query_dense([1.0, 0.0])[0].score == 0.75
