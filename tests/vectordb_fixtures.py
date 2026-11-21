"""In-memory adapter and selected-server fixtures for vector benchmark tests."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from benchmarks.vectordb.adapters import Hit
from benchmarks.vectordb.integration import SelectedServer
from benchmarks.vectordb.protocol import load_protocol
from benchmarks.vectordb.selection import candidate_config


class MemoryAdapter:
    def __init__(self, config):
        self.config = replace(config, index_readiness_timeout_seconds=0)
        self.rows = {}
        self.calls = []
        self.failure = None

    def reset(self):
        self.calls.append("reset")
        self.rows.clear()

    def upsert(self, rows):
        self.calls.append("upsert")
        self.rows.update((row.identifier, row) for row in rows)

    def count(self):
        return len(self.rows)

    def fetch(self, ids):
        return [self.rows[value] for value in ids if value in self.rows]

    def delete(self, ids):
        for value in ids:
            self.rows.pop(value, None)

    def delete_document(self, source_id):
        ids = [
            key
            for key, row in self.rows.items()
            if row.metadata["source_id"] == source_id
        ]
        self.delete(ids)
        return len(ids)

    def index_info(self):
        return {"status": "passed"}

    def health(self):
        return True

    def close(self):
        self.calls.append("close")

    def search(self, vector, limit, filters=None):
        self.calls.append("search")
        if self.failure:
            raise self.failure
        rows = [
            row
            for row in self.rows.values()
            if all(
                row.metadata.get(key) == value for key, value in (filters or {}).items()
            )
        ]
        rows.sort(
            key=lambda row: (
                -float(np.dot(row.vector, vector)) / np.linalg.norm(row.vector)
            )
        )
        return [
            Hit(row.identifier, float(np.dot(row.vector, vector)), row.metadata)
            for row in rows[:limit]
        ]


def fixture_protocol():
    return replace(load_protocol(), monitor_storage=False, monitor_server_memory=False)


def make_server(strategy="dense", reranker=None):
    chunks = [
        SimpleNamespace(
            identifier=value,
            document_id="doc",
            text=value,
            start=i,
            end=i + 1,
            tokens=1,
        )
        for i, value in enumerate("abc")
    ]
    embed_calls, lexical_calls = [], []
    index = SimpleNamespace(
        chunks=chunks,
        vectors=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
        embedder=SimpleNamespace(
            device="cpu",
            embed_query=lambda query: embed_calls.append(query) or [1, 0],
            spec=SimpleNamespace(maximum_length=8192, query_device="cpu"),
            input_token_counts=lambda texts, role: [len(text) for text in texts],
        ),
        bm25=SimpleNamespace(
            rank=lambda query, limit: (
                lexical_calls.append(query) or [(i, 1.0) for i in range(3)]
            )
        ),
    )
    protocol = fixture_protocol()
    _, config = candidate_config(protocol, "chroma|smoke-hnsw", 2)
    adapter = MemoryAdapter(config)
    server = SelectedServer(
        "chroma",
        config,
        index,
        strategy,
        SimpleNamespace(rrf_constant=60, rrf_weights=(1, 1)),
        adapter=adapter,
        reranker=reranker,
    )
    server.prepare()
    return server, embed_calls, lexical_calls
