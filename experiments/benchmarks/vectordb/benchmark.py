"""Server execution ledger and offline condition-local evaluation."""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)

from .adapters import Record, create
from .adapters.base import UnsupportedConfiguration
from .comparisons import operation_condition
from .conformance import _finish_index, _wait_for_adapter, check
from .docker_metrics import DockerMonitor
from .integration import wait_inventory
from .metrics import (
    aggregate_condition,
    aggregate_operations,
    percentile,
    repeatability,
)
from .selection import candidate_config
from .workload import records, tie_recall, tie_reference, validate_hits


def search_window(adapter, corpus, condition, schedule, window):
    """Dispatch through drain; validation is inside each common request boundary."""
    trusted = dict(zip(corpus.ids, corpus.metadata, strict=True))
    query_map = dict(zip(corpus.query_ids, corpus.queries, strict=True))
    groups = dict(zip(corpus.query_ids, corpus.source_groups, strict=True))
    configuration = (
        stable_hash(asdict(adapter.config)) if hasattr(adapter, "config") else None
    )

    def request(item):
        query_id, attempt = item
        row = {
            "condition": condition["id"],
            "corpus": condition["corpus"],
            "query_id": query_id,
            "attempt": attempt,
            "window": window,
            "source_group": groups[query_id],
            "operation": "search",
            "status": "failed",
            "filter_violation": None,
            "latency_seconds": None,
            "elapsed_to_failure_seconds": None,
            "ids": [],
            "scores": [],
            "error": None,
            "quality_designated": window == 0 and attempt == 1,
            "corpus_checksum": corpus.fingerprint,
            "configuration_checksum": configuration,
        }
        started = time.perf_counter()
        try:
            hits = adapter.search(query_map[query_id], 20, condition["filters"] or None)
            # Preserve even invalid raw outputs, rather than losing them on validation.
            row["ids"] = (
                [getattr(hit, "identifier", None) for hit in hits]
                if isinstance(hits, (list, tuple))
                else []
            )
            row["scores"] = (
                [getattr(hit, "score", None) for hit in hits]
                if isinstance(hits, (list, tuple))
                else []
            )
            violation = validate_hits(hits, trusted, 20, condition["filters"])
            row.update(
                status="success",
                latency_seconds=time.perf_counter() - started,
                filter_violation=bool(violation) if condition["filters"] else None,
            )
        except Exception as exc:  # noqa: BLE001 - retain backend request failures
            row.update(
                error=f"{type(exc).__name__}: {exc}",
                elapsed_to_failure_seconds=time.perf_counter() - started,
            )
        return row

    started = time.perf_counter()
    if condition["concurrency"] == 1:
        rows = [request(item) for item in schedule]
    else:
        with ThreadPoolExecutor(max_workers=condition["concurrency"]) as pool:
            rows = list(pool.map(request, schedule))
    elapsed = time.perf_counter() - started
    return rows, {
        "condition": condition["id"],
        "window": window,
        "wall_seconds": elapsed,
        "submitted": len(rows),
        "successful": sum(row["status"] == "success" for row in rows),
        "complete": True,
    }


def _prepared(rows):
    return [
        Record(row["id"], row["vector"], row["text"], row["metadata"]) for row in rows
    ]


def document_operation(adapter, baseline, spec, kind, attempt):
    source = spec["source_id"]
    kept = [row for row in baseline if row.metadata["source_id"] != source]
    incoming = (
        _prepared(spec["insert"] if kind == "insert" else spec["replacement"])
        if kind != "delete"
        else []
    )
    expected = baseline + incoming if kind == "insert" else kept + incoming
    row = {
        "operation": kind,
        "condition": operation_condition(spec),
        "operation_id": spec["id"],
        "source_group": spec.get("source_group"),
        "corpus": spec["corpus"],
        "source_id": source,
        "attempt": attempt,
        "status": "failed",
        "latency_seconds": None,
        "elapsed_to_failure_seconds": None,
        "chunk_count": len(incoming) if kind != "delete" else len(baseline) - len(kept),
        "error": None,
        "restoration_status": "not_started",
    }
    started = time.perf_counter()
    try:
        if kind in {"update", "delete"}:
            adapter.delete_document(source)
        if incoming:
            adapter.upsert(incoming)
        wait_inventory(adapter, expected)
        row.update(status="success", latency_seconds=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - record the logical operation failure
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_to_failure_seconds=time.perf_counter() - started,
        )
    finally:
        # Restoration is not part of operation completion latency. Never continue
        # with a different corpus after failed cleanup.
        try:
            if kind == "insert":
                adapter.delete([item.identifier for item in incoming])
            else:
                adapter.delete_document(source)
                adapter.upsert(
                    [item for item in baseline if item.metadata["source_id"] == source]
                )
            wait_inventory(adapter, baseline)
            row["restoration_status"] = "passed"
        except Exception as exc:  # noqa: BLE001 - preserve restoration failure separately
            row.update(
                restoration_status="failed",
                restoration_error=f"{type(exc).__name__}: {exc}",
            )
    return row


def startup(database, config, baseline, corpus, compose, protocol):
    """Stop outside the clock, start with preserved benchmark data, then query."""
    subprocess.run(
        ["docker", "compose", "-f", str(compose), "stop", database],
        check=True,
        capture_output=True,
        text=True,
        timeout=protocol.docker_restart_timeout_seconds,
    )
    started = time.perf_counter()
    adapter = None
    row = {"operation": "startup", "status": "failed", "seconds": None}
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose), "start", database],
            check=True,
            capture_output=True,
            text=True,
            timeout=protocol.docker_restart_timeout_seconds,
        )
        adapter = _wait_for_adapter(
            database,
            config,
            timeout_seconds=config.index_readiness_timeout_seconds,
            poll_seconds=config.index_readiness_poll_seconds,
        )
        adapter.index_info()
        wait_inventory(adapter, baseline)
        hits = adapter.search(corpus.queries[0], 20)
        validate_hits(hits, {row.identifier: row.metadata for row in baseline}, 20)
        if not hits:
            raise ValueError("Required startup query returned no preserved records")
        row.update(status="success", seconds=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - retain startup evidence
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_to_failure_seconds=time.perf_counter() - started,
        )
    finally:
        if adapter is not None:
            adapter.close()
    return row


def evaluate_candidate(
    candidate,
    payload,
    corpora,
    protocol,
    profile,
    compose,
    *,
    conformance=True,
    restart=True,
    evidence_sink=None,
):
    execution = protocol.profile(profile)
    requests, timings, windows, probes, resource_rows = [], [], [], {}, []
    problems, parameters = [], {}
    metrics = {}
    lifecycle = []
    for corpus_id, corpus in corpora.items():
        database, config = candidate_config(
            protocol, candidate, int(corpus.vectors.shape[1])
        )
        parameters = {
            "server": database,
            "configuration_settings": {
                key: getattr(config, key)
                for key in ("m", "ef_construction", "ef_search")
            },
            "configuration": asdict(config),
            "corpus_checksums": {
                key: value.content_checksum for key, value in corpora.items()
            },
            "transport": {
                "search": "grpc-single-channel-bounded-rpcs"
                if database == "weaviate"
                else "bounded-client-pool",
                "limit": config.connection_pool_limit,
                "request_timeout_seconds": config.request_timeout_seconds,
                "adapter_explicit_retries": 0,
            },
        }
        try:
            if conformance:
                checked, report = check(database, config, compose, protocol)
                checked.close()
                probes[corpus_id] = report
        except Exception as exc:  # noqa: BLE001 - retain conformance/infrastructure failures
            probes[corpus_id] = {
                "status": "unsupported"
                if isinstance(exc, UnsupportedConfiguration)
                else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            problems.append(f"conformance {corpus_id}: {type(exc).__name__}: {exc}")
        monitor = DockerMonitor(
            database,
            interval_seconds=protocol.docker_monitor_interval_seconds,
            stats_timeout_seconds=protocol.docker_stats_timeout_seconds,
            storage_timeout_seconds=protocol.docker_storage_timeout_seconds,
            helper_timeout_seconds=protocol.docker_helper_timeout_seconds,
            monitor_memory=protocol.monitor_server_memory,
            monitor_storage=protocol.monitor_storage,
        )
        adapter = None
        baseline = records(corpus)
        build = {
            "operation": "build",
            "corpus": corpus_id,
            "status": "not_executed",
            "seconds": None,
        }
        healthy = not problems
        with monitor:
            started = time.perf_counter()
            try:
                if healthy:
                    adapter = create(database, config)
                    started = time.perf_counter()
                    build["status"] = "failed"
                    adapter.reset()
                    adapter.upsert(baseline)
                    _finish_index(adapter)
                    build["index"] = adapter.index_info()
                    wait_inventory(adapter, baseline)
                    build.update(
                        status="success", seconds=time.perf_counter() - started
                    )
                    checkpoint = monitor.checkpoint(corpus_id)
                    if protocol.monitor_storage and not checkpoint["complete"]:
                        problems.append(f"Missing completed-build storage: {corpus_id}")
                    metrics[
                        f"operational.initial_build_time_seconds.corpus.{corpus_id}"
                    ] = build["seconds"]
                    metrics[
                        f"descriptor.persistent_storage_size_bytes.corpus.{corpus_id}"
                    ] = checkpoint["apparent_bytes"]
            except Exception as exc:  # noqa: BLE001 - preserve interrupted build evidence
                healthy = False
                build.update(
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_to_failure_seconds=time.perf_counter() - started,
                )
                problems.append(build["error"])
                monitor.checkpoint(corpus_id, complete=False)
            timings.append(build)
            for condition in [
                row for row in payload["conditions"] if row["corpus"] == corpus_id
            ]:
                schedule = [
                    (query_id, attempt)
                    for attempt in range(1, execution.repetitions + 1)
                    for query_id in condition["query_ids"]
                ]
                warm = healthy
                if warm:
                    warm_schedule = [
                        (condition["query_ids"][i % len(condition["query_ids"])], 0)
                        for i in range(condition["warmup_requests"])
                    ]
                    warm_rows, _ = search_window(
                        adapter, corpus, condition, warm_schedule, -1
                    )
                    warm = all(
                        row["status"] == "success" and not row["filter_violation"]
                        for row in warm_rows
                    )
                    if not warm:
                        problems.append(f"Warmup failed: {condition['id']}")
                        probes["warmup-" + condition["id"]] = warm_rows
                for window in range(condition["windows"]):
                    if warm:
                        rows, timing = search_window(
                            adapter, corpus, condition, schedule, window
                        )
                        requests.extend(rows)
                        if condition["mode"] == "load":
                            windows.append(timing)
                        timings.append({"operation": "search_window", **timing})
                    else:
                        groups = dict(
                            zip(corpus.query_ids, corpus.source_groups, strict=True)
                        )
                        requests.extend(
                            {
                                "condition": condition["id"],
                                "corpus": corpus_id,
                                "query_id": query_id,
                                "attempt": attempt,
                                "window": window,
                                "source_group": groups[query_id],
                                "operation": "search",
                                "status": "not_executed",
                                "filter_violation": None,
                                "ids": [],
                                "scores": [],
                                "latency_seconds": None,
                                "quality_designated": window == 0 and attempt == 1,
                                "error": "build_or_warmup_failed",
                            }
                            for query_id, attempt in schedule
                        )
            for spec in [
                row for row in payload["operations"] if row["corpus"] == corpus_id
            ]:
                spec = {
                    **spec,
                    "source_group": spec.get("source_group")
                    or next(
                        (
                            row.get("independent_source_id")
                            or row.get("source_group_id")
                            for row in corpus.metadata
                            if row["source_id"] == spec["source_id"]
                        ),
                        None,
                    ),
                }
                for kind in ("insert", "update", "delete"):
                    if healthy:
                        warm_row = document_operation(adapter, baseline, spec, kind, 0)
                        healthy = (
                            warm_row["status"] == "success"
                            and warm_row["restoration_status"] == "passed"
                        )
                        if not healthy:
                            problems.append(
                                f"Mutation warmup failed: {spec['id']}/{kind}"
                            )
                            probes[f"warmup-{spec['id']}-{kind}"] = warm_row
                    for attempt in range(1, execution.repetitions + 1):
                        if healthy:
                            row = document_operation(
                                adapter, baseline, spec, kind, attempt
                            )
                            if row["restoration_status"] != "passed":
                                healthy = False
                                problems.append(
                                    f"Restoration failed: {spec['id']}/{kind}"
                                )
                        else:
                            row = {
                                "condition": operation_condition(spec),
                                "operation_id": spec["id"],
                                "source_group": spec["source_group"],
                                "corpus": corpus_id,
                                "operation": kind,
                                "attempt": attempt,
                                "status": "not_executed",
                                "latency_seconds": None,
                                "error": "baseline_unavailable",
                            }
                        requests.append(row)
                        timings.append(row)
            if adapter is not None:
                adapter.close()
                adapter = None
            if healthy and restart:
                row = startup(database, config, baseline, corpus, compose, protocol)
                timings.append({"corpus": corpus_id, **row})
                metrics[f"operational.startup_time_seconds.corpus.{corpus_id}"] = row[
                    "seconds"
                ]
                if row["status"] != "success":
                    problems.append(f"Startup failed: {corpus_id}")
        evidence = monitor.evidence()
        lifecycle.append(evidence)
        resource_rows.extend(
            {"corpus": corpus_id, **row} for row in evidence["samples"]
        )
        resource_rows.extend(
            {"corpus": corpus_id, **row} for row in evidence["checkpoints"]
        )
        if protocol.monitor_server_memory and evidence["coverage"] != "complete":
            problems.append(f"Server telemetry {evidence['coverage']}: {corpus_id}")
    evidence = {
        "candidate": candidate,
        "requests": requests,
        "timings": timings,
        "windows": windows,
        "probes": probes,
        "resource_rows": resource_rows,
        "lifecycle": lifecycle,
        "problems": problems,
        "parameters": parameters,
        "metrics": metrics,
    }
    if evidence_sink is not None:
        evidence_sink(evidence)
    return score_server_evidence(payload, corpora, protocol, profile, evidence)


def score_server_evidence(payload, corpora, protocol, profile, evidence):
    """Pure offline scoring: no adapter, startup, model, or measurement calls."""
    execution = protocol.profile(profile)
    requests, timings, windows = (
        evidence[name] for name in ("requests", "timings", "windows")
    )
    probes, resource_rows, lifecycle = (
        evidence[name] for name in ("probes", "resource_rows", "lifecycle")
    )
    problems, parameters, metrics = (
        list(evidence["problems"]),
        dict(evidence["parameters"]),
        dict(evidence["metrics"]),
    )
    intervals, statuses = {}, {}
    references = {}
    for condition in payload["conditions"]:
        corpus = corpora[condition["corpus"]]
        query_map = dict(zip(corpus.query_ids, corpus.queries, strict=True))
        for query_id in condition["query_ids"]:
            references[condition["id"], query_id] = tie_reference(
                corpus, query_map[query_id], 20, condition["filters"]
            )
    # No offline oracle, scoring, or bootstrap runs inside server monitoring.
    query_rows, pair_rows, samples = [], [], []
    from .adapters import Hit

    for condition in payload["conditions"]:
        condition_requests = [
            row
            for row in requests
            if row["operation"] == "search" and row["condition"] == condition["id"]
        ]
        corpus = corpora[condition["corpus"]]
        groups = dict(zip(corpus.query_ids, corpus.source_groups, strict=True))
        rows = []
        for query_id in condition["query_ids"]:
            attempts = sorted(
                [
                    row
                    for row in condition_requests
                    if row["query_id"] == query_id and row["window"] == 0
                ],
                key=lambda row: row["attempt"],
            )
            reference = references[condition["id"], query_id]
            first = next((row for row in attempts if row["quality_designated"]), None)
            score = {
                "value": None,
                "status": "inapplicable"
                if not reference["effective_count"]
                else "unavailable",
                **reference,
            }
            if first and first["status"] == "success" and not first["filter_violation"]:
                score = tie_recall(
                    [Hit(identifier, 0, {}) for identifier in first["ids"]], reference
                )
            name = (
                "quality.filtered_ann_recall_at_20"
                if condition["filters"]
                else "quality.ann_recall_at_20"
            )
            repeated = repeatability(attempts, execution.repetitions)
            row = {
                "condition": condition["id"],
                "query_id": query_id,
                "source_group": groups[query_id],
                "metrics": {
                    name: score["value"],
                    "diagnostic.ranking_repeatability": repeated["value"],
                },
                "metric_statuses": {
                    name: score["status"],
                    "diagnostic.ranking_repeatability": repeated["status"],
                },
                "tie": score,
            }
            rows.append(row)
            pair_rows.extend(
                {"condition": condition["id"], "query_id": query_id, **pair}
                for pair in repeated["pairs"]
            )
            samples.append(
                SampleResult(
                    condition["id"] + ":" + query_id,
                    {
                        key + ".condition." + condition["id"]: value
                        for key, value in row["metrics"].items()
                    },
                    percentile(
                        [
                            item["latency_seconds"]
                            for item in attempts
                            if item["status"] == "success"
                        ],
                        0.5,
                    ),
                    {"source_group": groups[query_id], "condition": condition["id"]},
                )
            )
            if score["status"] == "unavailable" or repeated["status"] == "incomplete":
                problems.append(
                    f"Required query evidence missing: {condition['id']}/{query_id}"
                )
        query_rows.extend(rows)
        point, bounds, availability = aggregate_condition(
            rows,
            condition_requests,
            [window for window in windows if window["condition"] == condition["id"]],
            condition,
            protocol,
            profile,
        )
        metrics.update(point)
        intervals.update(bounds)
        statuses.update(availability)
    points, bounds, availability, operation_problems = aggregate_operations(
        requests, payload["operations"], protocol, profile
    )
    metrics.update(points)
    intervals.update(bounds)
    statuses.update(availability)
    problems.extend(operation_problems)
    usable = [row["bytes"] for row in resource_rows if row.get("bytes") is not None]
    metrics["operational.peak_server_ram_bytes.scope.server-lifecycle"] = (
        max(usable) if usable else None
    )
    rankings = [
        {
            key: row.get(key)
            for key in (
                "condition",
                "corpus",
                "query_id",
                "attempt",
                "window",
                "status",
                "ids",
                "scores",
                "filter_violation",
            )
        }
        for row in requests
        if row["operation"] == "search"
    ]
    artifacts = {
        "requests": requests,
        "rankings": rankings,
        "query_metrics": query_rows,
        "repeatability": pair_rows,
        "timings": timings,
        "resources": resource_rows,
        "metric_statuses": statuses,
        "validation_report": {
            "status": "failed" if problems else "passed",
            "problems": problems,
            "conformance": probes,
            "monitoring": lifecycle,
        },
        "workloads": payload,
    }
    if problems:
        raise CandidateExecutionError(
            "; ".join(problems),
            samples=tuple(samples),
            metrics=metrics,
            intervals=intervals,
            parameters=parameters,
            artifacts=artifacts,
        )
    return samples, {}, metrics, parameters, intervals, artifacts
