"""Isolated correctness probes; none of these timings are workload startup."""

from __future__ import annotations

import subprocess
import time

from .adapters import Record, create
from .adapters.base import DimensionMismatch
from .workload import clustered, records, validate_hits


def check(name, config, compose, protocol):
    from .integration import verify_inventory

    corpus = clustered(
        protocol.conformance["size"],
        config.dimension,
        protocol.conformance["dimension_query_count"],
        protocol.conformance["seed"],
        settings=protocol.synthetic,
    )
    adapter = create(name, config)
    report = {}
    expected = records(corpus)
    try:
        adapter.reset()
        adapter.upsert(expected)
        _finish_index(adapter)
        report["inventory"] = verify_inventory(adapter, expected)
        report["index"] = {"status": "passed", "details": adapter.index_info()}
        if not adapter.health():
            raise ValueError("Server health failed")
        trusted = {row.identifier: row.metadata for row in expected}
        hits = adapter.search(corpus.vectors[0], 1)
        validate_hits(hits, trusted, 1)
        report["cosine"] = {
            "status": "passed" if hits and hits[0].identifier == "0" else "failed"
        }
        filters = {str(key): "0" for key in protocol.conformance["compound_filters"]}
        hits = adapter.search(
            corpus.vectors[0], protocol.conformance["search_limit"], filters
        )
        violation = validate_hits(
            hits, trusted, protocol.conformance["search_limit"], filters
        )
        report["compound_filter"] = {
            "status": "passed" if hits and not violation else "failed"
        }
        filters = {protocol.conformance["empty_filter"]: "missing"}
        hits = adapter.search(
            corpus.vectors[0], protocol.conformance["search_limit"], filters
        )
        violation = validate_hits(
            hits, trusted, protocol.conformance["search_limit"], filters
        )
        report["empty_filter"] = {"status": "failed" if violation else "passed"}
        report["dimension_rejection"] = _dimension_rejection(adapter, config)
        replacement = Record("0", corpus.vectors[1], "replacement", corpus.metadata[0])
        adapter.upsert([replacement])
        expected[0] = replacement
        report["unique_id_replacement"] = verify_inventory(adapter, expected)
        source_id = corpus.metadata[0]["source_id"]
        adapter.delete_document(source_id)
        expected = [row for row in expected if row.metadata["source_id"] != source_id]
        replacement = Record(
            "replacement-0",
            corpus.vectors[0],
            "replacement",
            {**corpus.metadata[0], "source_id": source_id},
        )
        adapter.upsert([replacement])
        expected.append(replacement)
        report["document_replacement"] = verify_inventory(adapter, expected)
        adapter.delete(["replacement-0"])
        expected.pop()
        report["deletion"] = verify_inventory(adapter, expected)
        adapter.close()
        adapter = None
        _restart(name, compose, timeout_seconds=protocol.docker_restart_timeout_seconds)
        adapter = _wait_for_adapter(
            name,
            config,
            timeout_seconds=protocol.index_readiness_timeout_seconds,
            poll_seconds=protocol.index_readiness_poll_seconds,
        )
        report["persistence"] = verify_inventory(adapter, expected)
        report["index_after_restart"] = {
            "status": "passed",
            "details": adapter.index_info(),
        }
        if any(row.get("status") != "passed" for row in report.values()):
            raise ValueError(f"Conformance failed/unverified: {report}")
        return adapter, report
    except BaseException:
        if adapter is not None:
            adapter.close()
        raise


def _dimension_rejection(adapter, config):
    try:
        adapter.upsert(
            [Record("bad-dimension", [0.0] * (config.dimension + 1), "bad", {})]
        )
    except DimensionMismatch as exc:
        return {
            "status": "passed",
            "scope": "adapter",
            "server_status": "unverified",
            "reason": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - unknown rejection must remain unverified
        return {
            "status": "unverified",
            "scope": "unknown",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "failed", "scope": "adapter"}


def _finish_index(adapter):
    finish = getattr(adapter, "finish_index", None)
    if callable(finish):
        finish()


def _restart(name, compose, *, timeout_seconds):
    subprocess.run(
        ["docker", "compose", "-f", str(compose), "restart", name],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _wait_for_adapter(name, config, *, timeout_seconds, poll_seconds):
    deadline = time.monotonic() + timeout_seconds
    error = None
    while time.monotonic() < deadline:
        adapter = None
        try:
            adapter = create(name, config)
            if adapter.health():
                return adapter
        except Exception as exc:  # noqa: BLE001 - retain bounded startup transport errors
            error = exc
        if adapter is not None:
            adapter.close()
        time.sleep(poll_seconds)
    raise RuntimeError(f"{name} was not ready after start: {error}")
