"""Server execution ledger and offline condition-local evaluation."""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict

from benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)
from edumind.common.artifacts import stable_hash

from .adapters import Record, create
from .adapters.base import UnsupportedConfiguration
from .comparisons import operation_condition
from .conformance import _finish_index, _wait_for_adapter, check
from .docker_metrics import DockerMonitor
from .integration import wait_inventory
from .metrics import (
    ANN_KEYS,
    aggregate_operations,
    aggregate_session_scope,
    paired_search_summary,
    repeatability,
)
from .selection import candidate_config
from .workload import (
    SCOPES,
    SEARCH_LIMIT,
    SELECTED_FILES,
    classify_hits,
    global_inventory,
    paired_search_schedule,
    records,
    search_rows,
    tie_recall,
    validate_hits,
)


def search_request(
    adapter, vector, identity, filters, allowed, session_ids, global_ids
):
    """Time one serial logical adapter operation, then validate its response."""
    row = {
        **identity,
        "status": "failed" if adapter is not None else "not_executed",
        "filter_violation": None,
        "session_isolation_violation": None,
        "outside_session_ids": [],
        "outside_selection_ids": [],
        "latency_seconds": None,
        "elapsed_to_failure_seconds": None,
        "ids": None,
        "scores": None,
        "error": "build_or_warmup_failed" if adapter is None else None,
    }
    if adapter is None:
        return row
    filtered = identity["condition"] == SELECTED_FILES
    started = time.perf_counter()
    completed = None
    try:
        hits = adapter.search(vector, SEARCH_LIMIT, filters)
        completed = time.perf_counter()
        if isinstance(hits, (list, tuple)):
            row["ids"] = [getattr(hit, "identifier", None) for hit in hits]
            row["scores"] = [getattr(hit, "score", None) for hit in hits]
        checks = classify_hits(
            hits,
            global_ids,
            session_ids,
            allowed,
            filtered=filtered,
        )
    except Exception as exc:  # noqa: BLE001 - capture any client/response failure in the request ledger
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            elapsed_to_failure_seconds=(completed or time.perf_counter()) - started,
        )
    else:
        row.update(status="success", latency_seconds=completed - started, **checks)
    return row


def search_session(adapter, session, corpus, global_ids, configuration, *, warmups):
    """Warm each active scope outside the ledger, then execute the frozen order."""
    query_map = dict(zip(corpus.query_ids, corpus.queries, strict=True))
    session_ids = set(corpus.ids)
    schedule = list(search_rows(session, corpus))
    warm = {}
    if adapter is not None:
        for scope in SCOPES:
            representative = next(
                (item for item in schedule if item[0]["condition"] == scope), None
            )
            if representative is None:
                continue
            identity, filters, allowed = representative
            rows = [
                search_request(
                    adapter,
                    query_map[identity["query_id"]],
                    {**identity, "attempt": 0, "quality_designated": False},
                    filters,
                    allowed,
                    session_ids,
                    global_ids,
                )
                for _ in range(warmups)
            ]
            warm[scope] = rows
    blocked = {
        scope
        for scope, rows in warm.items()
        if any(
            row["status"] != "success"
            or row["filter_violation"]
            or row["session_isolation_violation"]
            for row in rows
        )
    }
    return [
        search_request(
            adapter if identity["condition"] not in blocked else None,
            query_map[identity["query_id"]],
            {**identity, "configuration_checksum": configuration},
            filters,
            allowed,
            session_ids,
            global_ids,
        )
        for identity, filters, allowed in schedule
    ], {scope: warm[scope] for scope in sorted(blocked)}


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
    requests, timings, probes, resource_rows = [], [], {}, []
    problems, parameters = [], {}
    metrics = {}
    lifecycle = []
    globally_known = global_inventory(corpora)
    for session in payload["sessions"]:
        corpus_id = session["id"]
        corpus = corpora[corpus_id]
        database, config = candidate_config(
            protocol, candidate, int(corpus.vectors.shape[1])
        )
        configuration = asdict(config)
        if not parameters:
            parameters = {
                "server": database,
                "configuration_settings": {
                    key: getattr(config, key)
                    for key in ("m", "ef_construction", "ef_search")
                },
                "configuration": {
                    key: value
                    for key, value in configuration.items()
                    if key != "dimension"
                },
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
        healthy = probes.get(corpus_id, {}).get("status") not in {
            "failed",
            "unsupported",
        }
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
            search, warm_failures = search_session(
                adapter if healthy else None,
                session,
                corpus,
                globally_known,
                stable_hash(configuration),
                warmups=execution.warmups,
            )
            requests.extend(search)
            for scope, failure in warm_failures.items():
                problems.append(f"Warmup failed: {corpus_id}/{scope}")
                probes[f"warmup-{corpus_id}-{scope}"] = failure
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
        lifecycle.append(
            {**evidence, "corpus": corpus_id, "configuration": configuration}
        )
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


def _score_first_response(first, reference):
    from .adapters import Hit

    if (
        first
        and first["status"] == "success"
        and not first["filter_violation"]
        and not first["session_isolation_violation"]
    ):
        return tie_recall(
            [Hit(identifier, 0, {}) for identifier in first["ids"]], reference
        )
    return {
        "value": None,
        "status": "inapplicable" if not reference["effective_count"] else "unavailable",
        **reference,
    }


def paired_search_from_ledger(sessions, requests):
    """Score the paired subset only after reconciling both sides with the schedule."""
    expected = paired_search_schedule(sessions)
    questions = {(sid, qid) for sid, qid, _ in expected}
    sides = {scope: {} for scope in SCOPES}
    for row in requests:
        if row["operation"] != "search":
            continue
        scope = row["condition"]
        question = (row["session_id"], row["query_id"])
        if scope != SELECTED_FILES and not row["paired"] and question not in questions:
            continue
        identity = (*question, row["attempt"])
        if (
            not row["paired"]
            or scope not in sides
            or type(row["attempt"]) is not int
            or identity not in expected
            or row["quality_designated"] != (row["attempt"] == 1)
        ):
            raise ValueError(
                f"Unexpected paired search ledger identity: {scope}/{identity}"
            )
        if identity in sides[scope]:
            raise ValueError(
                f"Duplicate paired search ledger identity: {scope}/{identity}"
            )
        if row["status"] not in {"success", "failed", "not_executed"}:
            raise ValueError("Unknown paired search ledger status")
        sides[scope][identity] = row
    for scope, side in sides.items():
        if side.keys() != expected:
            raise ValueError(f"Missing scheduled paired search ledger rows: {scope}")

    rows = []
    for session in sessions:
        sid = session["id"]
        for qid in session["selected_sources"]:
            for scope in SCOPES:
                score = _score_first_response(
                    sides[scope][sid, qid, 1],
                    session["exact_references"][qid][scope],
                )
                name = ANN_KEYS[scope]
                rows.append(
                    {
                        "session_id": sid,
                        "query_id": qid,
                        "condition": scope,
                        "session_group_id": session["session_group_id"],
                        "paired": True,
                        "metrics": {name: score["value"]},
                        "metric_statuses": {name: score["status"]},
                    }
                )
    return paired_search_summary(
        rows, [row for side in sides.values() for row in side.values()]
    )


def score_server_evidence(payload, corpora, protocol, profile, evidence):
    """Pure offline scoring: no adapter, startup, model, or measurement calls."""
    execution = protocol.profile(profile)
    requests, timings = evidence["requests"], evidence["timings"]
    probes, resource_rows, lifecycle = (
        evidence[name] for name in ("probes", "resource_rows", "lifecycle")
    )
    problems, parameters, metrics = (
        list(evidence["problems"]),
        dict(evidence["parameters"]),
        dict(evidence["metrics"]),
    )
    intervals, statuses = {}, {}
    query_rows, pair_rows, samples = [], [], []

    search_requests = [row for row in requests if row["operation"] == "search"]
    attempts_by_query = {}
    for row in search_requests:
        attempts_by_query.setdefault(
            (row["session_id"], row["query_id"], row["condition"]), []
        ).append(row)
    for session in payload["sessions"]:
        session_id = session["id"]
        for query_id, references in session["exact_references"].items():
            for scope, reference in references.items():
                attempts = sorted(
                    attempts_by_query.get((session_id, query_id, scope), ()),
                    key=lambda row: row["attempt"],
                )
                first = next(
                    (row for row in attempts if row["quality_designated"]), None
                )
                score = _score_first_response(first, reference)
                name = ANN_KEYS[scope]
                repeated = repeatability(attempts, execution.repetitions)
                identity = {
                    "condition": scope,
                    "session_id": session_id,
                    "query_id": query_id,
                    "session_group_id": session["session_group_id"],
                    "selected_source_ids": session["selected_sources"].get(query_id)
                    if scope == SELECTED_FILES
                    else None,
                    "paired": query_id in session["selected_sources"],
                }
                row = {
                    **identity,
                    "first_attempt_status": first["status"] if first else "missing",
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
                query_rows.append(row)
                pair_rows.extend({**identity, **pair} for pair in repeated["pairs"])
                samples.append(
                    SampleResult(
                        f"{scope}:{session_id}:{query_id}",
                        {
                            key + ".condition." + scope: value
                            for key, value in row["metrics"].items()
                        },
                        None,
                        identity,
                    )
                )
                if (
                    score["status"] == "unavailable"
                    or repeated["status"] == "incomplete"
                ):
                    problems.append(
                        f"Required query evidence missing: {session_id}/{scope}/{query_id}"
                    )
    session_summary = []
    for scope in SCOPES:
        rows = [row for row in query_rows if row["condition"] == scope]
        scoped = [row for row in search_requests if row["condition"] == scope]
        point, bounds, availability, per_session = aggregate_session_scope(
            rows,
            scoped,
            scope,
            protocol,
            profile,
        )
        metrics.update(point)
        intervals.update(bounds)
        statuses.update(availability)
        session_summary.extend(per_session)
    for row in search_requests:
        if row["session_isolation_violation"] or row["filter_violation"]:
            problems.append(
                f"Forbidden result: {row['session_id']}/{row['query_id']}/{row['condition']}/{row['attempt']}"
            )
        if row["status"] == "not_executed":
            problems.append(
                f"Search not executed: {row['session_id']}/{row['query_id']}/{row['condition']}/{row['attempt']}"
            )
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
                "session_id",
                "session_group_id",
                "query_id",
                "attempt",
                "schedule_position",
                "selected_source_ids",
                "allowed_set_checksum",
                "eligible_chunk_count",
                "exact_reference_checksum",
                "corpus_checksum",
                "configuration_checksum",
                "status",
                "ids",
                "scores",
                "filter_violation",
                "outside_session_ids",
                "outside_selection_ids",
            )
        }
        for row in requests
        if row["operation"] == "search"
    ]
    artifacts = {
        "requests": requests,
        "rankings": rankings,
        "query_metrics": query_rows,
        "session_metrics": [
            {
                **row,
                "chunk_count": len(corpora[row["session_id"]].ids),
                "file_count": len(
                    {item["source_id"] for item in corpora[row["session_id"]].metadata}
                ),
            }
            for row in session_summary
        ],
        "paired_search": paired_search_summary(query_rows, search_requests),
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
