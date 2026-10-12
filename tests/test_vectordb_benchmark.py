"""Vector-server contracts and the exact retrieval oracle."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from experiments.benchmarks.vectordb import conformance
from experiments.benchmarks.vectordb.adapters import InvalidIndexState, Record
from experiments.benchmarks.vectordb.adapters.chroma import Chroma
from experiments.benchmarks.vectordb.adapters.qdrant import Qdrant
from experiments.benchmarks.vectordb.adapters.weaviate import Weaviate
from experiments.benchmarks.vectordb.docker_metrics import DockerMonitor
from experiments.benchmarks.vectordb.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)
from experiments.benchmarks.vectordb.run import COMPOSE, _config
from experiments.benchmarks.vectordb.workload import Corpus, exact_ids


@pytest.mark.parametrize("indices", [None, np.array([6, 3, 1, 5, 2, 0, 4])])
def test_exact_oracle_uses_ids_to_break_ties_at_cutoff(indices):
    vectors = np.ones((7, 2), dtype=np.float32)
    corpus = Corpus(vectors, vectors[:1], np.array([0]), ({},) * 7, "fixture")
    assert exact_ids(corpus, vectors[0], 3, candidate_indices=indices) == [
        "0",
        "1",
        "2",
    ]


def test_exact_oracle_ranks_similarity_before_id_and_handles_short_filters():
    vectors = np.array([[0, 1], [1, 0], [1, 0]], dtype=np.float32)
    corpus = Corpus(
        vectors,
        vectors[:1],
        np.array([0]),
        ({"scope": "other"}, {"scope": "yes"}, {"scope": "yes"}),
        "fixture",
    )
    assert exact_ids(corpus, vectors[1], 2) == ["1", "2"]
    assert exact_ids(corpus, vectors[1], 5, {"scope": "yes"}) == ["1", "2"]
    assert exact_ids(corpus, vectors[1], 5, {"scope": "missing"}) == []
    assert exact_ids(corpus, vectors[1], 0) == []


def test_chroma_accepts_numpy_vectors_without_numpy_scalar_lists():
    from chromadb.api.types import normalize_embeddings

    protocol = load_protocol()
    adapter = object.__new__(Chroma)
    adapter.config = _config("chroma", 2, protocol, **protocol.smoke_hnsw)
    calls = []

    def upsert(**kwargs):
        normalize_embeddings(kwargs["embeddings"])
        calls.append(kwargs)

    def query(**kwargs):
        normalize_embeddings(kwargs["query_embeddings"])
        calls.append(kwargs)
        return {"ids": [[]], "metadatas": [[]], "distances": [[]]}

    adapter.collection = SimpleNamespace(upsert=upsert, query=query)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    adapter.upsert([Record("id", vector, "text", {"source_id": "source"})])
    assert adapter.search(vector, 1) == []
    assert calls[0]["embeddings"] == [[1.0, 0.0]]
    assert all(type(value) is float for value in calls[0]["embeddings"][0])


def test_chroma_applies_http_limits_and_closes_transport(monkeypatch):
    import chromadb
    import httpx

    protocol = load_protocol()
    config = _config("chroma", 2, protocol, **protocol.smoke_hnsw)
    transport = httpx.Client(transport=httpx.MockTransport(lambda request: None))
    admin_transport = httpx.Client(transport=httpx.MockTransport(lambda request: None))
    calls = []
    client = SimpleNamespace(
        _server=SimpleNamespace(_session=transport),
        _admin_client=SimpleNamespace(
            _server=SimpleNamespace(_session=admin_transport)
        ),
        close=lambda: calls.append("closed"),
    )

    def http_client(**kwargs):
        settings = kwargs["settings"]
        assert settings.chroma_http_max_connections == config.connection_pool_limit
        assert (
            settings.chroma_http_max_keepalive_connections
            == config.connection_pool_limit
        )
        return client

    monkeypatch.setattr(chromadb, "HttpClient", http_client)
    adapter = Chroma(config)
    assert transport.timeout == httpx.Timeout(config.request_timeout_seconds)
    assert admin_transport.timeout == httpx.Timeout(config.request_timeout_seconds)
    adapter.close()
    assert calls == ["closed"]
    assert transport.is_closed
    assert admin_transport.is_closed


def test_weaviate_reset_uses_current_client_hnsw_api():
    protocol = load_protocol()
    adapter = object.__new__(Weaviate)
    adapter.config = _config("weaviate", 2, protocol, **protocol.smoke_hnsw)
    calls = []
    adapter.client = SimpleNamespace(
        collections=SimpleNamespace(
            exists=lambda name: False,
            create=lambda *args, **kwargs: calls.append(kwargs),
            get=lambda name: object(),
        )
    )
    adapter.reset()
    options = calls[0]["vector_config"].vectorIndexConfig
    assert options.distance.value == "cosine"
    assert options.ef == adapter.config.ef_search
    assert options.maxConnections == adapter.config.m


@pytest.mark.parametrize("mismatch", [False, True])
def test_weaviate_verifies_named_vector_hnsw_settings(mismatch):
    protocol = load_protocol()
    adapter = object.__new__(Weaviate)
    adapter.config = _config("weaviate", 2, protocol, **protocol.smoke_hnsw)
    index = SimpleNamespace(
        vector_index_type=lambda: "hnsw",
        skip=False,
        distance_metric=SimpleNamespace(value="cosine"),
        max_connections=adapter.config.m,
        ef_construction=adapter.config.ef_construction,
        ef=1 if mismatch else adapter.config.ef_search,
    )
    configuration = SimpleNamespace(
        vector_config={"default": SimpleNamespace(vector_index_config=index)}
    )
    adapter.collection = SimpleNamespace(
        config=SimpleNamespace(get=lambda: configuration)
    )
    if mismatch:
        with pytest.raises(InvalidIndexState, match="configuration mismatch"):
            adapter.index_info()
    else:
        assert (
            adapter.index_info()["configuration"]["ef_search"]
            == adapter.config.ef_search
        )


def test_qdrant_protocol_rejects_unsupported_zero_threshold(tmp_path):
    payload = yaml.safe_load(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert load_protocol().qdrant_full_scan_threshold == 10
    payload["adapters"]["qdrant_full_scan_threshold"] = 0
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="qdrant_full_scan_threshold"):
        load_protocol(path)


def test_qdrant_resumes_paused_optimizers_without_bypassing_readiness(monkeypatch):
    from qdrant_client import models

    protocol = load_protocol()
    adapter = object.__new__(Qdrant)
    adapter.config = _config("qdrant", 2, protocol, **protocol.smoke_hnsw)
    adapter.models = models
    states = iter(
        SimpleNamespace(status=status, points_count=35, indexed_vectors_count=count)
        for status, count in [("grey", 40), ("grey", 40), ("yellow", 30), ("green", 35)]
    )
    updates = []
    adapter.client = SimpleNamespace(
        get_collection=lambda name: next(states),
        update_collection=lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        "experiments.benchmarks.vectordb.adapters.qdrant.time.sleep", lambda _: None
    )
    assert adapter.index_info()["status"] == "green"
    assert len(updates) == 1
    assert updates[0][0] == (adapter.config.collection,)
    assert updates[0][1]["optimizers_config"].model_dump(exclude_none=True) == {}


def test_conformance_closes_client_on_initial_failure(monkeypatch):
    events = []

    def fail():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(
        conformance,
        "create",
        lambda *args: SimpleNamespace(
            reset=fail, close=lambda: events.append("closed")
        ),
    )
    protocol = load_protocol()
    config = _config("chroma", 2, protocol, **protocol.smoke_hnsw)
    with pytest.raises(RuntimeError, match="reset failed"):
        conformance.check("chroma", config, COMPOSE, protocol)
    assert events == ["closed"]


def test_docker_monitor_finishes_sampler_before_final_snapshot(monkeypatch):
    events = []
    monitor = DockerMonitor(
        "chroma",
        interval_seconds=0.25,
        stats_timeout_seconds=10,
        storage_timeout_seconds=30,
        helper_timeout_seconds=60,
        monitor_memory=True,
        monitor_storage=False,
    )
    monitor.thread = SimpleNamespace(join=lambda: events.append("joined"))
    monkeypatch.setattr(monitor, "_sample", lambda: events.append("final sample"))
    monitor.__exit__()
    assert monitor.stop_event.is_set()
    assert events == ["joined", "final sample"]
