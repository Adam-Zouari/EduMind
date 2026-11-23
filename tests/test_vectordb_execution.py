"""Vector benchmark workload, conformance, and selected-server integration."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from vectordb_fixtures import (
    MemoryAdapter,
    fixture_protocol,
    freeze_test_sessions,
    make_server,
)

from benchmarks.common.contracts import CandidateExecutionError
from benchmarks.vectordb import benchmark, conformance, run, validation
from benchmarks.vectordb.adapters import Hit, Record
from benchmarks.vectordb.adapters.base import DimensionMismatch
from benchmarks.vectordb.integration import verify_inventory
from benchmarks.vectordb.metrics import (
    aggregate_session_scope,
    interval,
    paired_search_summary,
    repeatability,
)
from benchmarks.vectordb.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)
from benchmarks.vectordb.selection import (
    candidate_config,
    resolve_selected_config,
)
from benchmarks.vectordb.workload import (
    ALL_SESSION,
    SELECTED_FILES,
    Corpus,
    global_inventory,
    search_rows,
    tie_recall,
    tie_reference,
    validate_hits,
)


def test_tie_boundary_alternatives_cannot_replace_strict_neighbor():
    vectors = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [-1, 0]], dtype=float)
    corpus = Corpus(
        vectors,
        vectors[:1],
        np.array([0]),
        ({}, {"source_id": "same"}, {"source_id": "same"}, {}, {}),
        "fixture",
    )
    reference = tie_reference(corpus, vectors[0], 3)
    assert reference["boundary_slots"] == 2
    for ids, expected in [
        (["0", "2", "3"], 1),
        (["1", "2", "3"], 2 / 3),
        (["0", "1", "4"], 2 / 3),
    ]:
        assert (
            tie_recall([Hit(value, 1, {}) for value in ids], reference, 3)["value"]
            == expected
        )
    empty = tie_reference(corpus, vectors[0], filters={"source_id": "absent"})
    assert tie_recall([], empty)["status"] == "inapplicable"
    short = tie_reference(corpus, vectors[0], 20, filters={"source_id": "same"})
    assert short["effective_count"] == 2
    assert tie_recall([Hit("2", 0, {})], short)["value"] == 0.5


@pytest.mark.parametrize(
    "hits",
    [
        [Hit("unknown", 1, {})],
        [Hit("a", 1, {}), Hit("a", 1, {})],
        [Hit("a", float("nan"), {})],
        [Hit("a", None, {})],
        [Hit("a", True, {})],
        None,
    ],
)
def test_invalid_search_envelopes_are_execution_errors(hits):
    with pytest.raises(ValueError):
        validate_hits(hits, {"a": {"source_id": "actual"}}, 20)


def test_forged_echoed_metadata_does_not_hide_filter_violation():
    assert validate_hits(
        [Hit("a", 1, {"source_id": "forged"})],
        {"a": {"source_id": "actual"}},
        20,
        {"source_id": "forged"},
    )
    assert not validate_hits([], {"a": {}}, 20, {"source_id": "missing"})


def test_frozen_serial_schedule_warms_each_scope_and_matches_sealed_order():
    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    payload = freeze_test_sessions(payload, corpora, 3)
    known = global_inventory(corpora)
    for session in payload["sessions"]:
        corpus = corpora[session["id"]]
        _, config = candidate_config(fixture_protocol(), "chroma|smoke-hnsw", 2)
        adapter = MemoryAdapter(config)
        adapter.upsert(benchmark.records(corpus))
        actual = []
        search = adapter.search

        def record_order(vector, limit, filters=None, *, actual=actual, search=search):
            actual.append(filters)
            return search(vector, limit, filters)

        adapter.search = record_order
        rows, failures = benchmark.search_session(
            adapter, session, corpus, known, "config", warmups=1
        )
        assert not failures
        planned = list(search_rows(session, corpus))
        assert actual[2:] == [filters for _, filters, _ in planned]
        assert [row["condition"] for row in rows] == [
            row["condition"] for row, _, _ in planned
        ]
        paired = [row for row in rows if row["paired"]]
        assert paired[0]["condition"] == paired[3]["condition"]
        assert paired[0]["condition"] != paired[2]["condition"]
        assert all(row["status"] == "success" for row in rows)


def test_schedule_reuses_allowed_sets_but_not_query_reference_identities():
    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    session, corpus = payload["sessions"][0], corpora["fixture"]
    session["selected_sources"]["q2"] = session["selected_sources"]["q1"]
    freeze_test_sessions(payload, corpora, 3)
    for scope in (ALL_SESSION, SELECTED_FILES):
        rows = [
            (identity, allowed)
            for identity, _, allowed in search_rows(session, corpus)
            if identity["condition"] == scope
        ]
        assert len({id(allowed) for _, allowed in rows}) == 1
        assert isinstance(rows[0][1], frozenset)
        assert len({row["allowed_set_checksum"] for row, _ in rows}) == 1
        assert len({row["exact_reference_checksum"] for row, _ in rows}) == 2


@pytest.mark.parametrize("scope", [ALL_SESSION, SELECTED_FILES])
def test_global_known_but_outside_session_keeps_completed_timing_and_blocks_quality(
    scope,
):
    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    corpus = corpora["fixture"]
    session = payload["sessions"][0]
    identity, filters, allowed = next(
        item for item in search_rows(session, corpus) if item[0]["condition"] == scope
    )
    adapter = SimpleNamespace(
        search=lambda *args: [Hit("d", 1, {"source_id": "doc-a"})]
    )
    row = benchmark.search_request(
        adapter,
        corpus.queries[0],
        identity,
        filters,
        allowed,
        set(corpus.ids),
        global_inventory(corpora),
    )
    assert row["status"] == "success" and row["latency_seconds"] is not None
    assert row["outside_session_ids"] == ["d"] and row["session_isolation_violation"]
    assert row["filter_violation"] is (True if scope == SELECTED_FILES else None)
    adapter.search = lambda *args: [Hit("unknown", 1, {})]
    row = benchmark.search_request(
        adapter,
        corpus.queries[0],
        identity,
        filters,
        allowed,
        set(corpus.ids),
        global_inventory(corpora),
    )
    assert row["status"] == "failed" and row["latency_seconds"] is None


def test_repeatability_aligns_attempts_and_rejects_identical_outside_session_rankings():
    first = {"attempt": 1, "status": "success", "ids": ["a"]}
    second = {**first, "attempt": 2}
    third = {"attempt": 3, "status": "failed", "ids": None}
    reference = repeatability([first, second, third], 3)
    assert reference["value"] == 1 / 3
    assert repeatability([second, third, first], 3) == reference
    violated = [
        {**row, "session_isolation_violation": True, "filter_violation": None}
        for row in (first, second)
    ]
    assert repeatability([*violated, third], 3)["value"] == 0
    assert repeatability([first, second], 3)["status"] == "incomplete"
    assert repeatability([first], 1)["status"] == "inapplicable"


def test_session_macro_and_paired_timing_use_distinct_denominators():
    key = "quality.ann_recall_at_20"
    rows = [
        {
            "session_id": session,
            "session_group_id": session,
            "query_id": qid,
            "condition": ALL_SESSION,
            "paired": paired,
            "metrics": {key: score, "diagnostic.ranking_repeatability": None},
            "metric_statuses": {
                key: "scored",
                "diagnostic.ranking_repeatability": "inapplicable",
            },
        }
        for session, qid, score, paired in (
            ("one", "q1", 1, True),
            ("two", "q2", 0, False),
            ("two", "q3", 0, False),
        )
    ]
    filtered_key = "quality.filtered_ann_recall_at_20"
    rows.append(
        {
            **rows[0],
            "condition": SELECTED_FILES,
            "metrics": {filtered_key: 1},
            "metric_statuses": {filtered_key: "scored"},
        }
    )
    requests = [
        {
            **row,
            "status": "success",
            "filter_violation": None,
            "latency_seconds": latency,
        }
        for row, latency in (
            (rows[0], 0.01),
            (rows[1], 100),
            (rows[2], 200),
            (rows[3], None),
        )
    ]
    metrics, _, statuses, sessions = aggregate_session_scope(
        rows[:3], requests[:3], ALL_SESSION, fixture_protocol(), "smoke"
    )
    assert metrics[key + ".condition.all-session"] == 0.5
    assert metrics[key + ".condition.all-session.sample_count"] == 2
    assert [row["eligible_questions"] for row in sessions] == [1, 2]
    paired = paired_search_summary(rows, requests)
    assert paired["paired_sessions"] == paired["paired_questions"] == 1
    assert paired["unfiltered_search_latency_ms_p95"] == 10
    assert paired["filtered_completed"] == paired["filtered_missing_timing"] == 1
    assert paired["filtered_timed"] == paired["filtered_failed"] == 0
    assert paired["filtered_search_latency_ms_p95"] is None
    assert statuses[key + ".condition.all-session"]["contributing_questions"] == 3


def test_vector_input_seal_detects_actual_vector_and_metadata_changes(tmp_path):
    payload = json.loads(validation.manifest_path("smoke").read_text())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    protocol = fixture_protocol()
    report_path, report = validation.prepare_report(
        path, "smoke", protocol, report_root=tmp_path
    )
    assert report["status"] == "passed", report
    assert (
        validation.verify_inputs(path, "smoke", protocol, report_root=tmp_path)[0]
        == report_path
    )
    payload["sessions"][0]["vectors"][0][0] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        validation.verify_inputs(path, "smoke", protocol, report_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    ["duplicate", "filter", "all-files", "schedule", "reference", "legacy", "empty"],
)
def test_session_validator_rejects_invalid_membership_schedule_and_reference(
    tmp_path, change
):
    payload = json.loads(validation.manifest_path("smoke").read_text())
    session = payload["sessions"][0]
    if change == "duplicate":
        payload["sessions"].append(session)
    elif change == "filter":
        session["selected_sources"]["q1"] = ["unknown"]
    elif change == "all-files":
        session["selected_sources"]["q1"] = ["doc-a", "doc-b", "doc-c"]
    elif change == "schedule":
        session["schedule"][:2] = reversed(session["schedule"][:2])
    elif change == "reference":
        session["exact_references"]["q1"]["selected-files"]["boundary_slots"] = 0
    elif change == "legacy":
        payload["schema_version"] = 1
    else:
        session["queries"], session["query_ids"], session["query_sources"] = [], [], []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, report = validation.prepare_report(
        path, "smoke", fixture_protocol(), report_root=tmp_path
    )
    assert report["status"] == "failed"


def test_no_guessed_ci_minimum_even_many_questions_from_one_source():
    result = interval(
        [[1] * 50, [None]],
        lambda groups: next(
            (value for group in groups for value in group if value is not None), None
        ),
        protocol=fixture_protocol(),
        profile="development",
        metric="quality",
        unit="source",
    )
    assert result["estimate"] == 1 and result["lower"] is None
    assert (
        result["independent_source_count"] == 2
        and result["contributing_source_count"] == 1
    )
    assert result["reason"] == "ci_support_not_frozen"


def test_inventory_detects_noop_replacement_and_same_count_corruption():
    protocol = fixture_protocol()
    _, config = candidate_config(protocol, "chroma|smoke-hnsw", 2)
    adapter = MemoryAdapter(config)
    expected = [Record("a", [1, 0], "old", {"source_id": "doc"})]
    adapter.upsert(expected)
    changed = [Record("a", [0, 1], "new", {"source_id": "doc"})]
    with pytest.raises(ValueError, match="content mismatch"):
        verify_inventory(adapter, changed)
    adapter.rows = {"other": expected[0]}
    with pytest.raises(ValueError, match="IDs"):
        verify_inventory(adapter, expected)


def test_dimension_outage_not_passing_rejection():
    config = SimpleNamespace(dimension=2)

    def outage(rows):
        raise ConnectionError("down")

    assert (
        conformance._dimension_rejection(SimpleNamespace(upsert=outage), config)[
            "status"
        ]
        == "unverified"
    )

    def reject(rows):
        raise DimensionMismatch("wrong dimension")

    result = conformance._dimension_rejection(SimpleNamespace(upsert=reject), config)
    assert (
        result["status"] == "passed"
        and result["scope"] == "adapter"
        and result["server_status"] == "unverified"
    )


@pytest.mark.parametrize(
    "strategy,counts", [("dense", (1, 0)), ("bm25", (0, 1)), ("rrf", (1, 1))]
)
def test_live_strategy_executes_only_its_required_sources(strategy, counts):
    server, dense, lexical = make_server(strategy)
    if strategy == "bm25":
        server.adapter.failure = ConnectionError("dense server is down")
    result = server.retrieve("question")
    assert len(result["ranking"]) == 3
    assert (len(dense), len(lexical)) == counts


@pytest.mark.parametrize("permutation", [[0, 0, 1], [0, 1], [0, 1, 3], [False, 1, 2]])
def test_reranker_requires_complete_actual_pool(permutation):
    server, _, _ = make_server(reranker=SimpleNamespace(rank=lambda *args: permutation))
    evidence = {}
    with pytest.raises(ValueError, match="permutation"):
        server.retrieve("question", evidence=evidence)
    assert evidence["pool"] == [0, 1, 2] and evidence["ranking"] is None


def test_short_ann_response_valid_but_short_integration_unavailable():
    server, _, _ = make_server()
    server.adapter.search = lambda *args: [Hit("a", 1.0, {})]
    assert not validate_hits(server.adapter.search(), server.trusted, 20)
    evidence = {}
    with pytest.raises(ValueError, match="short"):
        server.retrieve("question", evidence=evidence)
    assert evidence["database_ids"] == ["a"] and evidence["pool"] is None


def test_document_operations_verify_visibility_and_restore_baseline():
    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    _, config = candidate_config(fixture_protocol(), "chroma|smoke-hnsw", 2)
    adapter = MemoryAdapter(config)
    from benchmarks.vectordb.workload import records

    baseline = records(corpora["fixture"])
    adapter.upsert(baseline)
    for kind in ("insert", "update", "delete"):
        row = benchmark.document_operation(
            adapter, baseline, payload["operations"][0], kind, 1
        )
        assert row["status"] == "success" and row["restoration_status"] == "passed"
        verify_inventory(adapter, baseline)


def test_evaluator_retains_failed_first_and_later_success(monkeypatch):
    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    payload["operations"] = []
    payload = freeze_test_sessions(payload, corpora, 3)
    protocol = fixture_protocol()

    def factory(database, config):
        adapter = MemoryAdapter(config)
        search = adapter.search
        calls = 0

        def fail_first(vector, limit, filters=None):
            nonlocal calls
            calls += 1
            if calls == 3:  # two scope warmups, then designated first q1
                raise TimeoutError("first failed")
            return search(vector, limit, filters)

        adapter.search = fail_first
        return adapter

    monkeypatch.setattr(benchmark, "create", factory)
    with pytest.raises(CandidateExecutionError) as caught:
        benchmark.evaluate_candidate(
            "chroma|smoke-hnsw",
            payload,
            corpora,
            protocol,
            "development",
            run.COMPOSE,
            conformance=False,
            restart=False,
        )
    error = caught.value
    requests = error.artifacts["requests"]
    q1 = [
        row
        for row in requests
        if row["query_id"] == "q1" and row["condition"] == ALL_SESSION
    ]
    assert [row["status"] for row in q1] == ["failed", "success", "success"]
    row = error.artifacts["query_metrics"][0]
    assert row["metrics"]["quality.ann_recall_at_20"] is None
    assert row["metrics"]["diagnostic.ranking_repeatability"] == 1 / 3


def test_mock_complete_cli_persists_equivalent_tables(tmp_path, monkeypatch):
    raw = yaml.safe_load(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["measurement"] = {"monitor_storage": False, "monitor_server_memory": False}
    raw["candidates"]["smoke"] = raw["candidates"]["smoke"][:1]
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    protocol = load_protocol(protocol_path)
    root = tmp_path / "reports"
    _, report = validation.prepare_report(
        validation.manifest_path("smoke"), "smoke", protocol, report_root=root
    )
    assert report["status"] == "passed"
    monkeypatch.setattr(run, "image_lock", lambda: {"image:chroma": "fixture"})
    monkeypatch.setattr(run, "verify_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, "create", lambda name, config: MemoryAdapter(config))
    monkeypatch.setattr(
        benchmark,
        "check",
        lambda name, config, *args: (MemoryAdapter(config), {"status": "passed"}),
    )
    monkeypatch.setattr(
        benchmark, "startup", lambda *args: {"status": "success", "seconds": 1.0}
    )
    assert (
        run.main(
            [
                "--profile",
                "smoke",
                "--protocol",
                str(protocol_path),
                "--report-root",
                str(root),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--no-mlflow",
            ]
        )
        == 0
    )
    summaries = list((tmp_path / "artifacts").rglob("summary.json"))
    assert len(summaries) == 1 and json.loads(summaries[0].read_text())["complete"]
    for path in summaries[0].parent.glob("candidates/*/*.parquet"):
        frame = pd.read_parquet(path)
        csv = (
            pd.read_csv(path.with_suffix(".csv"), float_precision="round_trip")
            if len(frame.columns)
            else pd.DataFrame()
        )
        assert list(frame.columns) == list(csv.columns)
        for column in frame.columns:
            assert (
                frame[column].fillna("").astype(str).tolist()
                == csv[column].fillna("").astype(str).tolist()
            )


def test_temporary_mlflow_parent_child_keep_statuses_and_csv_evidence(
    tmp_path, monkeypatch
):
    import mlflow
    from mlflow import MlflowClient

    from benchmarks.common.tracking import benchmark_experiment
    from benchmarks.vectordb.selection import STAGE, SUITE

    previous_uri = mlflow.get_tracking_uri()
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment(
        benchmark_experiment(SUITE, STAGE),
        artifact_location=(tmp_path / "mlflow-artifacts").as_uri(),
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    original = run.main
    monkeypatch.setattr(
        run,
        "main",
        lambda args: original([arg for arg in args if arg != "--no-mlflow"]),
    )
    try:
        test_mock_complete_cli_persists_equivalent_tables(tmp_path, monkeypatch)
        client = MlflowClient(tracking_uri=uri)
        runs = client.search_runs([experiment_id])
        assert len(runs) == 2
        child = next(item for item in runs if item.data.tags.get("mlflow.parentRunId"))
        parent = next(item for item in runs if item.info.run_id != child.info.run_id)
        assert parent.info.run_name == "smoke"
        names = {item.path for item in client.list_artifacts(child.info.run_id)}
        assert {"requests.csv", "requests.parquet", "metric_statuses.json"} <= names
        assert not any(
            key.endswith((".ci_lower", ".ci_upper")) for key in child.data.metrics
        )
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("dimension", [384, 768, 1024])
def test_selected_config_is_explicit_not_inferred_from_workload_name(
    tmp_path, dimension
):
    from benchmarks.vectordb.selection import STAGE, SUITE

    protocol = load_protocol()
    declaration = {
        "id": "chroma|reviewed",
        "database": "chroma",
        "settings": {"m": 16, "ef_construction": 100, "ef_search": 64},
    }
    protocol = replace(protocol, candidates={"smoke": [], "development": [declaration]})
    summary = {
        "complete": True,
        "run_id": "source",
        "plan": {"suite": SUITE, "stage": STAGE, "profile": "validation"},
        "protocols": {"vector_database": {"checksum": protocol.meta.checksum}},
        "candidates": [
            {
                "candidate": declaration["id"],
                "status": "success",
                "parameters": {
                    "server": "chroma",
                    "configuration_settings": dict(declaration["settings"]),
                },
            }
        ],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_summary": "summary.json",
                "source_run_id": "source",
                "selected_candidates": [declaration["id"]],
                "selected_by": "fixture-reviewer",
                "selected_date": "2026-11-13",
                "reason": "test-only control",
            }
        ),
        encoding="utf-8",
    )
    database, config, provenance = resolve_selected_config(
        decision, protocol, dimension
    )
    assert (
        database == "chroma"
        and config.dimension == dimension
        and config.ef_search == 64
    )
    assert provenance["candidate_id"] == declaration["id"]
    summary["candidates"][0]["parameters"]["configuration_settings"]["ef_search"] = 8
    # Mutating the independent serialized source cannot mutate protocol settings.
    declaration["settings"]["ef_search"] = 64
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        resolve_selected_config(decision, protocol, dimension)


def test_storage_fallback_preserves_apparent_bytes_and_no_block_conversion(monkeypatch):
    from benchmarks.vectordb import docker_metrics

    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            returncode=1 if "exec" in args else 0, stdout="1048579 /volume\n", stderr=""
        )

    monkeypatch.setattr(docker_metrics.subprocess, "run", command)
    assert (
        docker_metrics._storage_bytes(
            "chroma", timeout_seconds=1, helper_timeout_seconds=1
        )
        == 1048579
    )
    assert all("-sb" in call for call in calls)
    assert all("-sk" not in call for call in calls)


@pytest.mark.parametrize("reverse", [False, True])
def test_postbuild_checkpoint_and_configuration_are_corpus_scoped(monkeypatch, reverse):
    from benchmarks.vectordb.docker_metrics import DockerMonitor

    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    fixture = corpora["fixture-extra"]
    corpora["fixture-extra"] = replace(
        fixture,
        vectors=np.pad(fixture.vectors, ((0, 0), (0, 1))),
        queries=np.pad(fixture.queries, ((0, 0), (0, 1))),
    )
    payload = freeze_test_sessions(payload, corpora, 1)
    if reverse:
        corpora = dict(reversed(corpora.items()))
    events = []

    def factory(name, config):
        adapter = MemoryAdapter(config)
        search = adapter.search

        def observed(*args, **kwargs):
            events.append("query")
            return search(*args, **kwargs)

        adapter.search = observed
        return adapter

    monkeypatch.setattr(benchmark, "create", factory)
    monkeypatch.setattr(
        DockerMonitor,
        "checkpoint",
        lambda *args, **kwargs: (
            events.append("checkpoint") or {"complete": True, "apparent_bytes": 5}
        ),
    )
    result = benchmark.evaluate_candidate(
        "chroma|smoke-hnsw",
        payload,
        corpora,
        fixture_protocol(),
        "smoke",
        run.COMPOSE,
        conformance=False,
        restart=False,
    )
    assert len(result) == 6 and events[0] == "checkpoint" and "query" in events
    assert result[2]["descriptor.persistent_storage_size_bytes.corpus.fixture"] == 5
    assert "dimension" not in result[3]["configuration"]
    assert {
        row["corpus"]: row["configuration"]["dimension"]
        for row in result[5]["validation_report"]["monitoring"]
    } == {"fixture": 2, "fixture-extra": 3}


@pytest.mark.parametrize("mismatch", ["digest", "client", "missing_client"])
def test_deployment_lock_rejects_malformed_or_changed_identities(
    tmp_path, monkeypatch, mismatch
):
    from benchmarks.vectordb import docker_metrics

    payload = json.loads(
        (
            docker_metrics.PROJECT_ROOT / "data/benchmarks/models/vectordb.json"
        ).read_text()
    )
    versions = dict(payload["clients"])
    if mismatch == "digest":
        payload["images"]["chroma"] = "chroma:latest"
    elif mismatch == "client":
        versions["qdrant-client"] = "different"
    else:
        payload["clients"].pop("psycopg-pool")
    path = tmp_path / "data/benchmarks/models/vectordb.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(docker_metrics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(docker_metrics, "version", lambda name: versions[name])
    with pytest.raises(RuntimeError):
        docker_metrics.image_lock()


def test_startup_excludes_stop_and_checks_preserved_contents(monkeypatch):
    _, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    from benchmarks.vectordb.workload import records

    protocol = fixture_protocol()
    _, config = candidate_config(protocol, "chroma|smoke-hnsw", 2)
    adapter = MemoryAdapter(config)
    adapter.upsert(records(corpora["fixture"]))
    events = []
    monkeypatch.setattr(
        benchmark.subprocess, "run", lambda args, **kwargs: events.append(args[-2])
    )
    monkeypatch.setattr(benchmark, "_wait_for_adapter", lambda *args, **kwargs: adapter)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: events.append("clock") or next(clock)
    )
    result = benchmark.startup(
        "chroma",
        config,
        records(corpora["fixture"]),
        corpora["fixture"],
        run.COMPOSE,
        protocol,
    )
    assert result["status"] == "success" and result["seconds"] == 2
    assert events[:3] == ["stop", "clock", "start"]


@pytest.mark.parametrize(
    "positions",
    [
        [(0, float("nan")), (1, 1), (2, 1)],
        [(0, 1), (0, 1), (1, 1)],
        [(0, 1), (1, 1), (8, 1)],
    ],
)
def test_bm25_malformed_outputs_rejected_before_fusion(positions):
    server, _, _ = make_server("bm25")
    server.index.bm25.rank = lambda *args: positions
    with pytest.raises(ValueError, match="lexical"):
        server.retrieve("question")


def test_integration_explicit_six_tuple_uses_question_medians(monkeypatch):
    from benchmarks.vectordb import retrieval_run

    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    session = payload["sessions"][0]
    session["selected_sources"] = {}
    freeze_test_sessions(payload, corpora, 3)
    server, _, _ = make_server()
    index = server.index
    index.tokenizer = SimpleNamespace(spans=lambda text: [(0, len(text))])
    questions = [
        {
            "id": value,
            "question": value,
            "answerable": True,
            "independent_source_id": "same",
            "source_document_ids": ["doc"],
            "evidence": [
                {
                    "id": "E1",
                    "evidence_type": "text",
                    "document_id": "doc",
                    "intervals": [{"start": 0, "end": 1}],
                }
            ],
        }
        for value in ("q1", "q2")
    ]
    durations = iter([1.0, 2.0, 100.0, 2.0, 100.0, 2.0])

    class FakeServer:
        records = server.records

        def __init__(self, database, config, *args, **kwargs):
            self.config = config

        def prepare(self):
            pass

        def retrieve(self, question, filters=None, evidence=None):
            result = {
                "pool": [0, 1, 2],
                "ranking": [0, 1, 2],
                "database_ids": ["a", "b", "c"],
                "database_executed": True,
                "database_latency_seconds": 0.1,
                "filter_violation": False,
                "latency_seconds": next(durations) if evidence is not None else 0.01,
            }
            if evidence is not None:
                evidence.update(result)
            return result

        def close(self):
            pass

    monkeypatch.setattr(retrieval_run, "SelectedServer", FakeServer)
    prepared = {
        "sessions": [
            {
                "index": index,
                "questions": questions,
                "session": session,
                "corpus": corpora["fixture"],
            }
        ],
        "global_ids": set(corpora["fixture"].ids),
        "spec": {"session_ids": ["fixture"]},
        "retrieval": "dense|none",
        "retrieval_protocol": SimpleNamespace(),
        "reranker_factory": lambda: None,
    }
    result = retrieval_run.evaluate_integration(
        "chroma|smoke-hnsw", prepared, fixture_protocol(), "development"
    )
    assert len(result) == 6
    key = "operational.full_stack_retrieval_latency_ms_p50.condition.integration-all-session"
    assert result[2][key] == 51000
    assert (
        result[4][key]["lower"] is None
        and result[4][key]["independent_source_count"] == 1
    )
    assert len(result[5]["integration_requests"]) == 6


def test_integration_projects_one_embedding_runtime_and_builds_session_local_bm25(
    monkeypatch,
):
    from benchmarks.common.contracts import DatasetManifest
    from benchmarks.preparation import models
    from benchmarks.rag import evaluation
    from benchmarks.rag.retrieval_reranking import validation as rag_validation
    from benchmarks.vectordb import retrieval_run

    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    documents, questions, chunks, vectors = [], [], [], []
    for corpus in corpora.values():
        for identifier, text, metadata, vector in zip(
            corpus.ids, corpus.texts, corpus.metadata, corpus.vectors, strict=True
        ):
            doc = metadata["source_id"]
            documents.append(
                {
                    "kind": "document",
                    "id": doc,
                    "text": text,
                    "source_group_id": doc,
                }
            )
            chunks.append(evaluation.Chunk(identifier, doc, text, 0, len(text), 1))
            vectors.append(vector)
        for qid in corpus.query_ids:
            doc = corpus.metadata[0]["source_id"]
            questions.append(
                {
                    "kind": "question",
                    "id": qid,
                    "question": "What?",
                    "answerable": True,
                    "evidence_type": "text",
                    "evidence": [
                        {
                            "id": "e",
                            "evidence_type": "text",
                            "document_id": doc,
                            "intervals": [{"start": 0, "end": 1}],
                        }
                    ],
                }
            )
    manifest = DatasetManifest(
        "smoke",
        "test",
        "rag",
        "smoke",
        "fixture",
        "CC0",
        "test",
        "content",
        "test",
        42,
        (*documents, *questions),
    )
    runtime, tokenizer = object(), object()
    built = []

    def build(*args, **kwargs):
        built.append(kwargs)
        return evaluation.ExactIndex(
            chunks,
            np.array(vectors),
            runtime,
            tokenizer,
            None,
            None,
            "test",
            {},
            0.0,
            {},
        )

    lexical = []

    def bm25(texts, **kwargs):
        lexical.append(tuple(texts))
        return SimpleNamespace(texts=tuple(texts))

    monkeypatch.setattr(evaluation, "build_index", build)
    monkeypatch.setattr(retrieval_run, "BM25", bm25)
    monkeypatch.setattr(retrieval_run, "load_manifest", lambda _: manifest)
    monkeypatch.setattr(retrieval_run, "sha256_file", lambda _: "test")
    monkeypatch.setattr(rag_validation, "verify_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(models, "load_selected_model_lock", lambda _: {})
    monkeypatch.setattr(
        retrieval_run,
        "load_engineer_decision",
        lambda path, **kw: SimpleNamespace(
            source_run_id="test",
            selected_candidates=(
                "rrf|none"
                if path.name == "retrieval.decision"
                else "token-256-32|bge-small",
            ),
        ),
    )
    from pathlib import Path

    from benchmarks.rag.chunking_embedding.protocol import DEFAULT_PROTOCOL_PATH as ce
    from benchmarks.rag.retrieval_reranking.protocol import DEFAULT_PROTOCOL_PATH as rr

    prepared = retrieval_run.prepare_integration(
        {
            "session_ids": list(corpora),
            "manifest": "fixture.json",
            "embedding_selection": "embedding.decision",
            "retrieval_selection": "retrieval.decision",
            "chunking_protocol": ce,
            "retrieval_protocol": rr,
            "model_lock": "fixture.lock",
        },
        Path("."),
        "smoke",
        corpora,
        payload["sessions"],
    )
    assert len(built) == 1 and built[0]["with_bm25"] is False
    assert len(prepared["sessions"]) == 2
    assert all(
        bundle["index"].embedder is runtime and bundle["index"].tokenizer is tokenizer
        for bundle in prepared["sessions"]
    )
    assert lexical == [corpus.texts for corpus in corpora.values()]


def test_bootstrap_retains_null_groups_and_reports_undefined_degenerate_draws():
    from benchmarks.vectordb.metrics import mean

    protocol = replace(
        fixture_protocol(),
        ci_support={
            "test": {
                "minimum_groups": 2,
                "reviewed_by": "fixture",
                "reviewed_date": "2026-11-13",
            }
        },
    )
    statistic = lambda groups: mean([value for group in groups for value in group])
    bounds = interval(
        [[1.0], [2.0], [None]],
        statistic,
        protocol=protocol,
        profile="development",
        metric="test",
        unit="source",
    )
    assert (
        bounds["independent_source_count"] == 3
        and bounds["contributing_source_count"] == 2
    )
    assert (
        bounds["undefined_resamples"] > 0
        and bounds["defined_resamples"] + bounds["undefined_resamples"] == 10000
    )
    equal = interval(
        [[0.0], [0.0]],
        statistic,
        protocol=protocol,
        profile="development",
        metric="test",
        unit="source",
    )
    assert equal["lower"] == equal["upper"] == 0 and equal["degenerate"]


@pytest.mark.parametrize(
    "exception,status",
    [(ConnectionError("outage"), "failed"), (TimeoutError("timeout"), "failed")],
)
def test_configuration_infrastructure_errors_never_become_unsupported(
    monkeypatch, exception, status
):
    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))

    def fail(*args):
        raise exception

    monkeypatch.setattr(benchmark, "check", fail)
    with pytest.raises(CandidateExecutionError) as caught:
        benchmark.evaluate_candidate(
            "chroma|smoke-hnsw",
            payload,
            corpora,
            fixture_protocol(),
            "smoke",
            run.COMPOSE,
            restart=False,
        )
    assert (
        caught.value.artifacts["validation_report"]["conformance"]["fixture"]["status"]
        == status
    )
    assert all(
        row["status"] == "not_executed" for row in caught.value.artifacts["requests"]
    )


def test_native_query_gate_precedes_embedding_and_server_dispatch():
    from benchmarks.rag.evaluation import InputCompatibilityError

    server, embeddings, _ = make_server()
    server.index.embedder.spec.maximum_length = 2
    previous = len(server.adapter.calls)
    evidence = {}
    with pytest.raises(InputCompatibilityError, match="capacity"):
        server.retrieve("long question", evidence=evidence)
    assert not embeddings and len(server.adapter.calls) == previous
    assert evidence["query_input"]["truncated_inputs"] == 0


@pytest.mark.parametrize(
    "vector",
    [
        [0.0, 0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [1.0],
        [[1.0, 0.0]],
        ["x", "y"],
    ],
)
def test_invalid_live_query_vectors_rejected_before_server(vector):
    from benchmarks.rag.evaluation import OutputIntegrityError

    server, _, _ = make_server()
    server.index.embedder.embed_query = lambda query: vector
    previous = len(server.adapter.calls)
    with pytest.raises(OutputIntegrityError):
        server.retrieve("question")
    assert len(server.adapter.calls) == previous


@pytest.mark.parametrize("counts", [[True], [0], [], [1, 2]])
def test_malformed_native_query_count_envelope_rejected(counts):
    from benchmarks.rag.evaluation import InputCompatibilityError

    server, embeddings, _ = make_server()
    server.index.embedder.input_token_counts = lambda texts, role: counts
    with pytest.raises(InputCompatibilityError):
        server.retrieve("question")
    assert not embeddings
