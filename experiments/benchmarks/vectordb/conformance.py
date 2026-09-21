"""Correctness checks run against real servers before timing them."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .adapters import Adapter, Config, Record, create
from .workload import clustered, records
from .protocol import VectorDatabaseProtocol


def check(
    name: str,
    config: Config,
    compose: Path,
    protocol: VectorDatabaseProtocol,
) -> tuple[Adapter, Mapping[str, float]]:
    corpus = clustered(
        protocol.conformance["size"],
        config.dimension,
        protocol.conformance["dimension_query_count"],
        seed=protocol.conformance["seed"],
        settings=protocol.synthetic,
    )
    adapter = create(name, config)
    adapter.reset()
    adapter.upsert(records(corpus))
    _finish_index(adapter)
    flags = {
        "health_correctness": float(adapter.health()),
        "cosine_correctness": float(
            adapter.search(corpus.vectors[0], 1)[0].identifier == "0"
        ),
        "compound_filter_correctness": _compound(
            adapter,
            corpus.vectors[0],
            limit=protocol.conformance["search_limit"],
            filter_names=protocol.conformance["compound_filters"],
        ),
        "empty_filter_correctness": float(
            not adapter.search(
                corpus.vectors[0],
                protocol.conformance["search_limit"],
                {str(protocol.conformance["empty_filter"]): "missing"},
            )
        ),
        "ann_index_verified": float(bool(adapter.index_info())),
        "dimension_rejection_correctness": _dimension_rejection(adapter, config),
    }
    original_count = adapter.count()
    adapter.upsert([Record("0", corpus.vectors[1], "replacement", corpus.metadata[0])])
    flags["duplicate_id_correctness"] = float(adapter.count() == original_count)
    deleted = adapter.delete_document("doc-0")
    adapter.upsert(
        [Record("replacement-0", corpus.vectors[0], "replacement", {**corpus.metadata[0], "source_id": "doc-0"})]
    )
    replacement_hits = adapter.search(
        corpus.vectors[0],
        protocol.conformance["search_limit"],
        {"source_id": "doc-0"},
    )
    flags["replacement_correctness"] = float(
        deleted == protocol.synthetic.records_per_document
        and [hit.identifier for hit in replacement_hits] == ["replacement-0"]
    )
    adapter.delete(["replacement-0"])
    flags["deletion_correctness"] = float(
        not adapter.search(
            corpus.vectors[0],
            protocol.conformance["search_limit"],
            {"source_id": "doc-0"},
        )
    )
    persisted_count = adapter.count()
    adapter.close()
    restarted_at = time.perf_counter()
    _restart(name, compose, timeout_seconds=protocol.docker_restart_timeout_seconds)
    adapter = _wait_for_adapter(
        name,
        config,
        timeout_seconds=protocol.index_readiness_timeout_seconds,
        poll_seconds=protocol.index_readiness_poll_seconds,
    )
    flags["restart_readiness_seconds"] = time.perf_counter() - restarted_at
    flags["restart_persistence_correctness"] = float(adapter.count() == persisted_count)
    flags["ann_index_verified_after_restart"] = float(bool(adapter.index_info()))
    cold_query_started = time.perf_counter()
    adapter.search(corpus.vectors[0], protocol.conformance["search_limit"])
    flags["process_cold_query_seconds"] = time.perf_counter() - cold_query_started
    return adapter, flags


def _compound(
    adapter: Adapter,
    query: np.ndarray,
    *,
    limit: int,
    filter_names,
) -> float:
    filters = {str(name): "0" for name in filter_names}
    hits = adapter.search(query, limit, filters)
    return float(
        bool(hits)
        and all(
            all(str(hit.metadata.get(name)) == "0" for name in filters)
            for hit in hits
        )
    )


def _dimension_rejection(adapter: Adapter, config: Config) -> float:
    try:
        adapter.upsert([Record("bad-dimension", [0.0] * (config.dimension + 1), "bad", {})])
    except Exception:
        return 1.0
    return 0.0


def _finish_index(adapter: Adapter) -> None:
    finish = getattr(adapter, "finish_index", None)
    if callable(finish):
        finish()


def _restart(name: str, compose: Path, *, timeout_seconds: float) -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(compose), "restart", name],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _wait_for_adapter(
    name: str,
    config: Config,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> Adapter:
    deadline = time.monotonic() + timeout_seconds
    error: Exception | None = None
    while time.monotonic() < deadline:
        adapter = None
        try:
            adapter = create(name, config)
            if adapter.health():
                return adapter
        except Exception as exc:
            error = exc
        if adapter is not None:
            adapter.close()
        time.sleep(poll_seconds)
    raise RuntimeError(f"{name} was not ready after restart: {error}")


__all__ = ["check", "_finish_index"]
