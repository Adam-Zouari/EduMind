"""Pinned vector client transport and adapter contracts without live servers."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from queue import Queue
from threading import BoundedSemaphore, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.vectordb.adapters import Record
from benchmarks.vectordb.adapters import weaviate as weaviate_adapter
from benchmarks.vectordb.adapters.pgvector import PgVector
from benchmarks.vectordb.adapters.qdrant import Qdrant
from benchmarks.vectordb.adapters.weaviate import Weaviate
from benchmarks.vectordb.integration import verify_inventory
from benchmarks.vectordb.protocol import load_protocol
from benchmarks.vectordb.selection import candidate_config


def test_qdrant_real_rest_transport_accepts_frozen_connection_limits(monkeypatch):
    import qdrant_client

    # Construct the installed client's actual transport without server I/O.
    # Its constructor rejects simultaneous pool_size/limits even without I/O.
    monkeypatch.setattr(
        qdrant_client,
        "QdrantClient",
        partial(qdrant_client.QdrantClient, check_compatibility=False),
    )
    _, config = candidate_config(load_protocol(), "qdrant|smoke-hnsw", 2)
    adapter = Qdrant(config)
    try:
        limits = adapter.client._client._rest_args["limits"]
        assert limits.max_connections == config.connection_pool_limit
        assert limits.max_keepalive_connections == config.connection_pool_limit
    finally:
        adapter.close()


def test_pgvector_rejects_unhonored_collection_override_before_connect():
    _, config = candidate_config(load_protocol(), "pgvector|smoke-hnsw", 2)
    with pytest.raises(ValueError, match="collection"):
        PgVector(replace(config, collection="not_the_table"))


def test_weaviate_missing_distance_never_becomes_perfect_similarity():
    adapter = object.__new__(Weaviate)
    adapter.collection = SimpleNamespace(
        query=SimpleNamespace(
            near_vector=lambda **kwargs: SimpleNamespace(
                objects=[
                    SimpleNamespace(
                        properties={"edumind_id": "a"},
                        uuid="a",
                        metadata=SimpleNamespace(distance=None),
                    )
                ]
            )
        )
    )
    with pytest.raises(ValueError, match="distance"):
        adapter.search([1.0, 0.0], 20)


def test_weaviate_configures_http_connection_pool(monkeypatch):
    import weaviate

    connect = Mock()
    monkeypatch.setattr(weaviate, "connect_to_custom", connect)
    _, config = candidate_config(load_protocol(), "weaviate|smoke-hnsw", 2)
    Weaviate(config)

    connect.assert_called_once()
    connection = connect.call_args.kwargs["additional_config"].connection
    assert connection.session_pool_connections == config.connection_pool_limit
    assert connection.session_pool_maxsize == config.connection_pool_limit


@pytest.mark.parametrize("first_rpc_fails", [False, True], ids=["success", "failure"])
def test_weaviate_search_caps_concurrent_rpcs_and_releases_a_slot(
    monkeypatch, first_rpc_fails
):
    import weaviate

    _, config = candidate_config(load_protocol(), "weaviate|smoke-hnsw", 2)
    config = replace(config, connection_pool_limit=2)
    events = Queue()
    release = [Event(), Event()]
    watchdog = 5

    class ObservedSemaphore(BoundedSemaphore):
        def __enter__(self):
            if not self.acquire(blocking=False):
                # Report real contention before blocking, not merely thread start.
                events.put(("waiting", None))
                assert self.acquire(timeout=watchdog), "search slot was not released"

    def near_vector(*, near_vector, **_):
        index = int(near_vector[0])
        events.put(("dispatched", index))
        if index < len(release):
            assert release[index].wait(watchdog), "test did not release the held RPC"
        if index == 0 and first_rpc_fails:
            raise ConnectionError("fixture RPC failure")
        return SimpleNamespace(objects=[])

    # Patch only this adapter's factory; the real semaphore still owns the
    # permits and context-manager release, including the exception path.
    monkeypatch.setattr(
        weaviate_adapter,
        "threading",
        SimpleNamespace(BoundedSemaphore=ObservedSemaphore),
    )
    monkeypatch.setattr(weaviate, "connect_to_custom", Mock())
    adapter = Weaviate(config)
    adapter.collection = SimpleNamespace(query=SimpleNamespace(near_vector=near_vector))
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(adapter.search, [0.0, 0.0], 1)
            second = pool.submit(adapter.search, [1.0, 0.0], 1)
            assert {events.get(timeout=watchdog) for _ in release} == {
                ("dispatched", 0),
                ("dispatched", 1),
            }
            overflow = pool.submit(adapter.search, [2.0, 0.0], 1)
            assert events.get(timeout=watchdog) == ("waiting", None)

            release[0].set()
            if first_rpc_fails:
                with pytest.raises(ConnectionError, match="fixture RPC failure"):
                    first.result(timeout=watchdog)
            else:
                assert first.result(timeout=watchdog) == []
            assert events.get(timeout=watchdog) == ("dispatched", 2)
            assert overflow.result(timeout=watchdog) == []
            assert not second.done()
        finally:
            for event in release:
                event.set()
        assert second.result(timeout=watchdog) == []


@pytest.mark.parametrize("unexpected", [False, True])
def test_weaviate_null_schema_fields_are_absent_not_corrupt_metadata(unexpected):
    _, config = candidate_config(load_protocol(), "weaviate|smoke-hnsw", 2)
    expected = Record(
        "test-id",
        [1.0, 0.0],
        "canonical text",
        {
            "source_id": "course 1",
            "token_count": 0,
        },
    )
    properties = {
        "edumind_id": expected.identifier,
        "text": expected.text,
        **expected.metadata,
        "document_id": "unrecognized extra fact" if unexpected else None,
        "start": None,
        "end": None,
    }
    row = SimpleNamespace(
        uuid="test-server-uuid",
        properties=properties,
        vector={"default": expected.vector},
        metadata=SimpleNamespace(distance=0.0),
    )
    adapter = Weaviate.__new__(Weaviate)
    adapter.config = config
    adapter.collection = SimpleNamespace(
        query=SimpleNamespace(
            fetch_object_by_id=lambda *a, **k: row,
            near_vector=lambda **k: SimpleNamespace(objects=[row]),
        ),
        aggregate=SimpleNamespace(over_all=lambda **k: SimpleNamespace(total_count=1)),
    )
    assert adapter.fetch([expected.identifier])[0].metadata["token_count"] == 0
    assert adapter.search(expected.vector, 1)[0].metadata["token_count"] == 0
    if unexpected:
        with pytest.raises(ValueError, match="content mismatch"):
            verify_inventory(adapter, [expected])
    else:
        assert verify_inventory(adapter, [expected])["status"] == "passed"
        assert adapter.search(expected.vector, 1)[0].metadata == expected.metadata


def _deletion_adapter(count, outcomes):
    adapter = Weaviate.__new__(Weaviate)
    adapter.collection = SimpleNamespace(
        aggregate=SimpleNamespace(
            over_all=Mock(return_value=SimpleNamespace(total_count=count))
        ),
        data=SimpleNamespace(delete_many=Mock(side_effect=outcomes)),
    )
    return adapter


def test_weaviate_document_deletion_drains_all_server_batches():
    adapter = _deletion_adapter(
        10005,
        [
            SimpleNamespace(successful=10000, failed=0),
            SimpleNamespace(successful=5, failed=0),
        ],
    )
    assert adapter.delete_document("Course 1: Biology") == 10005
    assert adapter.collection.data.delete_many.call_count == 2
    calls = adapter.collection.data.delete_many.call_args_list
    assert calls[0].kwargs["where"] == calls[1].kwargs["where"]
    assert (
        adapter.collection.aggregate.over_all.call_args.kwargs["filters"]
        == calls[0].kwargs["where"]
    )


def test_weaviate_absent_document_needs_no_delete_call():
    adapter = _deletion_adapter(0, [])
    assert adapter.delete_document("absent") == 0
    adapter.collection.data.delete_many.assert_not_called()


@pytest.mark.parametrize(
    "outcome",
    [
        SimpleNamespace(successful=0, failed=0),
        SimpleNamespace(successful=1, failed=1),
        SimpleNamespace(successful=3, failed=0),
    ],
)
def test_weaviate_incomplete_deletion_fails_without_unbounded_retry(outcome):
    adapter = _deletion_adapter(2, [outcome])
    with pytest.raises(RuntimeError, match="no valid progress"):
        adapter.delete_document("source")
    adapter.collection.data.delete_many.assert_called_once()
