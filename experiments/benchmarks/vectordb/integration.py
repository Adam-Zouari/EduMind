"""Selected live-server path; reusable by the vector comparison and Final RAG."""

from __future__ import annotations

import math
import time
from dataclasses import asdict

import numpy as np

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.rag.methods import reciprocal_rank_fusion

from .adapters import Record, create
from .conformance import _finish_index
from .workload import validate_hits


def verify_inventory(adapter, expected):
    """Verify stored contents, not an ANN approximation or count alone."""
    if adapter.count() != len(expected):
        raise ValueError("Stored corpus count differs from expected inventory")
    for start in range(0, len(expected), adapter.config.upsert_batch_size):
        rows = expected[start : start + adapter.config.upsert_batch_size]
        observed = adapter.fetch([row.identifier for row in rows])
        by_id = {row.identifier: row for row in observed}
        if len(by_id) != len(observed) or set(by_id) != {
            row.identifier for row in rows
        }:
            raise ValueError("Stored corpus IDs differ from expected inventory")
        for row in rows:
            actual = by_id[row.identifier]
            left, right = np.asarray(actual.vector), np.asarray(row.vector)
            # Cosine servers may normalize stored vectors. Compare directions in
            # float32 serialization precision, not backend search scores/tie policy.
            if (
                actual.text != row.text
                or dict(actual.metadata) != dict(row.metadata)
                or left.shape != right.shape
                or not np.isfinite(left).all()
                or np.linalg.norm(left) <= 0
                or np.linalg.norm(right) <= 0
                or not np.allclose(
                    left / np.linalg.norm(left),
                    right / np.linalg.norm(right),
                    rtol=1e-6,
                    atol=1e-7,
                )
            ):
                raise ValueError(f"Stored record content mismatch: {row.identifier}")
    return {
        "status": "passed",
        "expected_count": len(expected),
        "verified_count": len(expected),
    }


def wait_inventory(adapter, expected):
    deadline = time.monotonic() + adapter.config.index_readiness_timeout_seconds
    while True:
        try:
            return verify_inventory(adapter, expected)
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(adapter.config.index_readiness_poll_seconds)


def validate_permutation(local, count):
    if (
        not isinstance(local, (list, tuple, np.ndarray))
        or len(local) != count
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in local
        )
        or sorted(int(value) for value in local) != list(range(count))
    ):
        raise ValueError("Reranker must return the complete actual pool permutation")


class SelectedServer:
    """Own one adapter; corpus preparation is outside online request timing.

    `index` is the prepared chunk/embed/BM25 bundle, never its ExactIndex search.
    `retrieve` returns actual pool/ranking positions and durable pool identity.
    BM25 needs neither an embedder call nor a database request.
    """

    def __init__(
        self,
        database,
        config,
        index,
        strategy,
        retrieval_protocol,
        *,
        reranker=None,
        adapter=None,
    ):
        self.database, self.config, self.index = database, config, index
        self.strategy = strategy.split("|", 1)[0]
        if self.strategy not in {"dense", "bm25", "rrf"}:
            raise ValueError(f"Unsupported selected retrieval strategy: {strategy}")
        self.protocol, self.reranker = retrieval_protocol, reranker
        self.limit = getattr(retrieval_protocol, "pool_size", 20)
        if self.limit != 20:
            raise ValueError(
                "Selected vector integration requires protocol pool_size 20"
            )
        self.readiness = None
        self.adapter = adapter if adapter is not None else create(database, config)
        self.records = [
            Record(
                chunk.identifier,
                index.vectors[i],
                chunk.text,
                {
                    "source_id": chunk.document_id,
                    "document_id": chunk.document_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "token_count": chunk.tokens,
                },
            )
            for i, chunk in enumerate(index.chunks)
        ]
        self.trusted = {row.identifier: row.metadata for row in self.records}
        self.by_id = {row.identifier: i for i, row in enumerate(self.records)}
        self.identity = stable_hash(
            {
                "configuration": asdict(config),
                "records": [
                    {
                        "id": row.identifier,
                        "text": row.text,
                        "metadata": dict(row.metadata),
                        "vector": list(map(float, row.vector)),
                    }
                    for row in self.records
                ],
            }
        )

    def prepare(self):
        self.adapter.reset()
        self.adapter.upsert(self.records)
        _finish_index(self.adapter)
        index = self.adapter.index_info()
        self.readiness = {
            "index": index,
            "inventory": wait_inventory(self.adapter, self.records),
            "identity": self.identity,
        }
        return self.readiness

    def retrieve(self, question, *, filters=None, evidence=None, input_envelope=None):
        started = time.perf_counter()
        query = str(question["question"] if isinstance(question, dict) else question)
        eligible = [
            i
            for i, row in enumerate(self.records)
            if all(
                row.metadata.get(key) == value for key, value in (filters or {}).items()
            )
        ]
        expected = min(self.limit, len(eligible))
        database_hits, dense, lexical = [], [], []
        details = {
            "identity": self.identity,
            "configuration": asdict(self.config),
            "expected_count": expected,
            "database_executed": False,
            "pool": None,
            "ranking": None,
            "readiness": self.readiness,
        }
        if evidence is not None:
            evidence.update(details)
            details = evidence
        if self.strategy in {"dense", "rrf"}:
            from experiments.benchmarks.rag.evaluation import (
                InputCompatibilityError,
                OutputIntegrityError,
                _model_input_counts,
                _validate_vectors,
            )

            spec = getattr(self.index.embedder, "spec", None) or getattr(
                self.index, "embedding_spec", None
            )
            if spec is None:
                raise InputCompatibilityError(
                    "Selected query embedding contract is unavailable",
                    {"reason_code": "missing_query_embedding_contract"},
                )
            counts = _model_input_counts(
                self.index.embedder, [query], role="query", batch_size=1
            )
            details["query_input"] = {
                "native_token_count": counts[0],
                "maximum_length": spec.maximum_length,
                "truncated_inputs": 0,
            }
            if counts[0] > spec.maximum_length:
                raise InputCompatibilityError(
                    "Selected query exceeds native embedding capacity; truncation is forbidden",
                    {"reason_code": "query_input_overflow", **details["query_input"]},
                )
            _check_envelope(input_envelope, "maximum_query_tokens", counts, details)
            vector = self.index.embedder.embed_query(query)
            vector = np.asarray(vector)
            if (
                vector.shape != (self.config.dimension,)
                or vector.dtype.kind not in "fiu"
            ):
                raise OutputIntegrityError(
                    "Selected query embedding has invalid shape/dtype"
                )
            _validate_vectors(vector[None, :], 1, self.config.dimension, "query")
            db_started = time.perf_counter()
            details["database_executed"] = True
            database_hits = self.adapter.search(vector, self.limit, filters)
            details.update(
                database_ids=[
                    getattr(hit, "identifier", None) for hit in database_hits
                ],
                database_scores=[getattr(hit, "score", None) for hit in database_hits],
            )
            violation = validate_hits(database_hits, self.trusted, self.limit, filters)
            details.update(
                database_executed=True,
                database_latency_seconds=time.perf_counter() - db_started,
                database_ids=[hit.identifier for hit in database_hits],
                database_scores=[hit.score for hit in database_hits],
                filter_violation=violation,
            )
            if violation:
                raise ValueError("Known forbidden ID returned by selected server")
            if len(database_hits) != expected:
                raise ValueError("Unexpectedly short integration database pool")
            dense = [self.by_id[hit.identifier] for hit in database_hits]
        if self.strategy in {"bm25", "rrf"}:
            if self.index.bm25 is None:
                raise ValueError("Selected lexical source is unavailable")
            # Filter the full lexical ranking before taking its required prefix.
            rows = self.index.bm25.rank(query, len(self.records))
            positions = [row[0] for row in rows]
            if (
                len(set(positions)) != len(positions)
                or any(
                    isinstance(position, (bool, np.bool_))
                    or not isinstance(position, (int, np.integer))
                    or not 0 <= position < len(self.records)
                    for position in positions
                )
                or any(
                    isinstance(score, (bool, np.bool_))
                    or not isinstance(score, (float, int, np.number))
                    or not math.isfinite(score)
                    for _, score in rows
                )
            ):
                raise ValueError(
                    "Invalid lexical IDs, duplicate positions or nonfinite scores"
                )
            allowed = set(eligible)
            lexical = [position for position, _ in rows if position in allowed][
                : self.limit
            ]
            if len(lexical) != expected or len(set(lexical)) != len(lexical):
                raise ValueError("Unexpectedly short/invalid lexical source pool")
        if self.strategy == "dense":
            pool = dense
        elif self.strategy == "bm25":
            pool = lexical
        else:
            pool = reciprocal_rank_fusion(
                [dense, lexical],
                self.limit,
                self.protocol.rrf_constant,
                weights=self.protocol.rrf_weights,
                tie_keys={
                    i: chunk.identifier for i, chunk in enumerate(self.index.chunks)
                },
            )
        details.update(
            pool=pool, pool_ids=[self.index.chunks[i].identifier for i in pool]
        )
        details["pool_checksum"] = stable_hash(
            {
                "identity": self.identity,
                "query": query,
                "filters": filters,
                "database_ids": details.get("database_ids"),
                "database_scores": details.get("database_scores"),
                "pool_ids": details["pool_ids"],
            }
        )
        if not pool or any(not self.index.chunks[i].text.strip() for i in pool):
            raise ValueError("Empty/token-free integration pool")
        order = pool
        if self.reranker is not None:
            documents = [self.index.chunks[i].text for i in pool]
            count_inputs = getattr(self.reranker, "input_token_counts", None)
            counts = count_inputs(query, documents) if callable(count_inputs) else None
            details["reranker_input_token_counts"] = counts
            if counts is not None and (
                len(counts) != len(documents)
                or any(type(count) is not int or count <= 0 for count in counts)
            ):
                raise ValueError(
                    "Reranker native token counts must cover the actual pool"
                )
            _check_envelope(input_envelope, "maximum_reranker_tokens", counts, details)
            try:
                local = self.reranker.rank(query, documents)
            except Exception as exc:
                details["reranker_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "diagnostic": getattr(exc, "diagnostic", None),
                }
                raise
            details["reranker_positions"] = (
                local.tolist()
                if isinstance(local, np.ndarray)
                else local
                if isinstance(local, (list, tuple))
                else repr(local)
            )
            validate_permutation(local, len(pool))
            order = [pool[int(i)] for i in local]
        from experiments.benchmarks.rag.evaluation import synchronize

        devices = {getattr(self.reranker, "device", "cpu")}
        if self.strategy in {"dense", "rrf"}:
            devices.add(spec.query_device)
        for device in devices:
            synchronize(device)
        details.update(
            ranking=order,
            ranking_ids=[self.index.chunks[i].identifier for i in order],
            latency_seconds=time.perf_counter() - started,
        )
        return details

    def close(self):
        self.adapter.close()


def _check_envelope(envelope, key, counts, evidence):
    if envelope is None:
        return
    maximum = envelope.get(key)
    if type(maximum) is not int or maximum <= 0 or counts is None:
        raise ValueError(f"Missing/unverifiable tested input envelope: {key}")
    if any(count > maximum for count in counts):
        from experiments.benchmarks.rag.evaluation import InputCompatibilityError

        raise InputCompatibilityError(
            f"Input exceeds tested qualification envelope: {key}",
            {
                "reason_code": "qualification_input_envelope_exceeded",
                "bound": key,
                "maximum": maximum,
                "actual_counts": counts,
                "truncated_inputs": 0,
            },
        )
