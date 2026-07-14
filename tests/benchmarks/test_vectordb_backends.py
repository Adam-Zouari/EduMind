from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from edumind.benchmarks.contracts import CandidateResult
from edumind.benchmarks.vectordb import (
    ChromaBackend,
    LanceBackend,
    _backend,
    _exact,
    _recall,
    recommend_vector_backend,
)


def test_real_chroma_backend_conforms_for_multi_field_filters(tmp_path) -> None:
    pytest.importorskip("chromadb")
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    metadata = [
        {"course": "ml", "page": 1},
        {"course": "math", "page": 2},
        {"course": "ml", "page": 2},
    ]
    backend = ChromaBackend(tmp_path / "chroma")
    backend.build(vectors, metadata)
    assert backend.query(vectors[0], 2)[0] == 0
    assert backend.query(vectors[0], 2, {"course": "ml", "page": 2}) == [2]
    backend.close()


class FakeSearch:
    def __init__(self, rows):
        self.rows = rows
        self.filter = None

    def where(self, expression, prefilter):
        assert prefilter is True
        self.filter = expression
        return self

    def limit(self, count):
        self.count = count
        return self

    def to_list(self):
        rows = self.rows
        if self.filter:
            rows = [row for row in rows if row["course"] == "ml" and row["page"] == 1]
        return rows[: self.count]


class FakeTable:
    def __init__(self, rows):
        self.rows = rows

    def search(self, vector):
        assert vector
        return FakeSearch(self.rows)


class FakeDatabase:
    def create_table(self, name, data, mode):
        assert name == "benchmark" and mode == "overwrite"
        return FakeTable(data)


def test_lancedb_adapter_uses_prefilter_and_persistent_rows(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(
        sys.modules, "lancedb", SimpleNamespace(connect=lambda path: FakeDatabase())
    )
    backend = LanceBackend(tmp_path / "lance")
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    backend.build(vectors, [{"course": "ml", "page": 1}, {"course": "math", "page": 2}])
    assert backend.query(vectors[0], 2, {"course": "ml", "page": 1}) == [0]
    backend.close()
    with pytest.raises(RuntimeError, match="not built"):
        backend.query(vectors[0], 1)


def test_vectordb_helper_validation(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown vector backend"):
        _backend("missing", tmp_path, 2)
    with pytest.raises(ValueError, match="requires metadata"):
        _exact(np.eye(2), np.asarray([1.0, 0.0]), 1, filters={"x": 1})
    assert _recall([], []) == 1.0


def test_vector_migration_requires_correctness_and_twenty_percent_gain() -> None:
    def result(name, recall, latency, disk):
        return CandidateResult(
            name,
            "success",
            name,
            {
                "ann_recall_at_10": recall,
                "filtered_ann_recall_at_10": recall,
                "filter_correctness": 1.0,
            },
            {},
            (),
            {"p95_latency_seconds": latency, "disk_bytes": disk},
        )

    chroma = result("chroma", 1.0, 10.0, 100.0)
    fast_but_wrong = result("qdrant-local", 0.98, 5.0, 50.0)
    assert recommend_vector_backend([chroma, fast_but_wrong]) == "chroma"
    qualified = result("lancedb-local", 0.995, 8.0, 95.0)
    assert recommend_vector_backend([chroma, qualified]) == "lancedb-local"
