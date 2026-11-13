"""Regressions for adapter incompatibilities exposed by live pinned servers."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.benchmarks.vectordb.adapters import Record
from experiments.benchmarks.vectordb.adapters.qdrant import Qdrant
from experiments.benchmarks.vectordb.adapters.weaviate import Weaviate
from experiments.benchmarks.vectordb.integration import verify_inventory
from experiments.benchmarks.vectordb.protocol import load_protocol
from experiments.benchmarks.vectordb.selection import candidate_config


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
