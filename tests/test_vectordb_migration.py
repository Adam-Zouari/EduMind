"""Behavioral vector migration tests, all services/inference controlled in memory."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.benchmarks.common.contracts import CandidateExecutionError
from experiments.benchmarks.vectordb import benchmark, conformance, run, validation
from experiments.benchmarks.vectordb.adapters import Hit, Record
from experiments.benchmarks.vectordb.adapters.base import DimensionMismatch
from experiments.benchmarks.vectordb.integration import (
    SelectedServer,
    verify_inventory,
)
from experiments.benchmarks.vectordb.metrics import (
    aggregate_condition,
    interval,
    repeatability,
)
from experiments.benchmarks.vectordb.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)
from experiments.benchmarks.vectordb.selection import (
    candidate_config,
    resolve_selected_config,
)
from experiments.benchmarks.vectordb.workload import (
    Corpus,
    tie_recall,
    tie_reference,
    validate_hits,
)


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


def test_tie_boundary_alternatives_cannot_replace_strict_neighbor():
    vectors = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [-1, 0]], dtype=float)
    corpus = Corpus(vectors, vectors[:1], np.array([0]), ({},) * 5, "fixture")
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


def test_load_all_failures_zero_qps_and_zero_duration_unavailable(monkeypatch):
    _, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    corpus = corpora["fixture"]
    protocol = fixture_protocol()
    _, config = candidate_config(protocol, "chroma|smoke-hnsw", 2)
    adapter = MemoryAdapter(config)
    adapter.failure = ConnectionError("offline")
    condition = {
        "id": "load",
        "corpus": "fixture",
        "concurrency": 2,
        "mode": "load",
        "filters": {},
    }
    rows, window = benchmark.search_window(
        adapter, corpus, condition, [("q1", 1), ("q2", 1)], 0
    )
    assert all(
        row["status"] == "failed" and row["elapsed_to_failure_seconds"] is not None
        for row in rows
    )
    window["wall_seconds"] = 2
    metrics, _, _ = aggregate_condition(
        [], rows, [window], condition, protocol, "smoke"
    )
    assert metrics["operational.qps.condition.load"] == 0
    assert metrics["diagnostic.request_error_rate.search.condition.load"] == 1
    window["wall_seconds"] = 0
    metrics, _, _ = aggregate_condition(
        [], rows, [window], condition, protocol, "smoke"
    )
    assert metrics["operational.qps.condition.load"] is None


def test_known_forbidden_result_is_execution_success_in_load():
    _, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    condition = {
        "id": "filter",
        "corpus": "fixture",
        "concurrency": 1,
        "mode": "load",
        "filters": {"source_id": "absent"},
    }
    adapter = SimpleNamespace(
        search=lambda *args: [Hit("a", 1, {"source_id": "absent"})]
    )
    rows, window = benchmark.search_window(
        adapter, corpora["fixture"], condition, [("q1", 1)], 0
    )
    assert rows[0]["status"] == "success" and rows[0]["filter_violation"] is True
    metrics, _, _ = aggregate_condition(
        [], rows, [window], condition, fixture_protocol(), "smoke"
    )
    assert metrics["diagnostic.request_error_rate.search.condition.filter"] == 0
    assert metrics["diagnostic.filter_violation_rate.condition.filter"] == 1
    assert metrics["operational.qps.condition.filter"] > 0


def test_repeatability_counts_failed_pairs_and_missing_schedule():
    success = {"status": "success", "ids": ["a"]}
    failed = {"status": "failed", "ids": []}
    assert repeatability([failed, success, success], 3)["value"] == 1 / 3
    assert repeatability([failed] * 3, 3)["value"] == 0
    assert repeatability([success] * 2, 3)["status"] == "incomplete"
    assert repeatability([success], 1)["status"] == "inapplicable"


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
    payload["corpora"][0]["vectors"][0][0] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        validation.verify_inputs(path, "smoke", protocol, report_root=tmp_path)


@pytest.mark.parametrize("change", ["duplicate", "filter"])
def test_validator_rejects_duplicate_conditions_and_unknown_filters(tmp_path, change):
    payload = json.loads(validation.manifest_path("smoke").read_text())
    if change == "duplicate":
        payload["conditions"].append(payload["conditions"][0])
    else:
        payload["conditions"][0]["filters"] = {"unknown": "x"}
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
    from experiments.benchmarks.vectordb.workload import records

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
    payload["conditions"] = [payload["conditions"][0]]
    protocol = fixture_protocol()

    def factory(database, config):
        adapter = MemoryAdapter(config)
        search = adapter.search
        calls = 0

        def fail_first(vector, limit, filters=None):
            nonlocal calls
            calls += 1
            if calls == 2:  # warmup then designated first q1
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
    assert (
        len(requests) == 6
        and requests[0]["status"] == "failed"
        and requests[2]["status"] == "success"
    )
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

    from experiments.benchmarks.common.tracking import benchmark_experiment
    from experiments.benchmarks.vectordb.selection import STAGE, SUITE

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
    from experiments.benchmarks.vectordb.selection import STAGE, SUITE

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
    from experiments.benchmarks.vectordb import docker_metrics

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


def test_postbuild_checkpoint_precedes_all_workload_queries(monkeypatch):
    from experiments.benchmarks.vectordb.docker_metrics import DockerMonitor

    payload, corpora = validation.load_inputs(validation.manifest_path("smoke"))
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


@pytest.mark.parametrize("mismatch", ["digest", "client", "missing_client"])
def test_deployment_lock_rejects_malformed_or_changed_identities(
    tmp_path, monkeypatch, mismatch
):
    from experiments.benchmarks.vectordb import docker_metrics

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
    from experiments.benchmarks.vectordb.workload import records

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
    from experiments.benchmarks.vectordb import retrieval_run

    server, _, _ = make_server()
    index = server.index
    index.tokenizer = SimpleNamespace(spans=lambda text: [(0, len(text))])
    questions = [
        {
            "id": value,
            "question": value,
            "document_id": "doc",
            "answerable": True,
            "evidence": [
                {
                    "id": "E1",
                    "evidence_type": "text",
                    "document_id": "doc",
                    "start": 0,
                    "end": 1,
                }
            ],
        }
        for value in ("q1", "q2")
    ]
    durations = iter([1.0, 100.0, 100.0, 2.0, 2.0, 2.0])

    class FakeServer:
        records = server.records

        def __init__(self, *args, **kwargs):
            pass

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
        "index": index,
        "questions": questions,
        "spec": {"corpus": "fixture", "filters": {}},
        "retrieval": "dense|none",
        "retrieval_protocol": SimpleNamespace(),
        "reranker_factory": lambda: None,
        "source_groups": {"q1": "same", "q2": "same"},
    }
    result = retrieval_run.evaluate_integration(
        "chroma|smoke-hnsw", prepared, fixture_protocol(), "development"
    )
    assert len(result) == 6
    key = (
        "operational.full_stack_retrieval_latency_ms_p50.condition.integration-fixture"
    )
    assert result[2][key] == 51000
    assert (
        result[4][key]["lower"] is None
        and result[4][key]["independent_source_count"] == 1
    )
    assert len(result[5]["integration_requests"]) == 6


def test_bootstrap_retains_null_groups_and_reports_undefined_degenerate_draws():
    from experiments.benchmarks.vectordb.metrics import mean

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


def test_qdrant_and_weaviate_transport_limits_are_applied(monkeypatch):
    import qdrant_client
    import weaviate

    from experiments.benchmarks.vectordb.adapters.qdrant import Qdrant
    from experiments.benchmarks.vectordb.adapters.weaviate import Weaviate

    _, config = candidate_config(fixture_protocol(), "chroma|smoke-hnsw", 2)
    qdrant_calls, weaviate_calls = [], []
    monkeypatch.setattr(
        qdrant_client,
        "QdrantClient",
        lambda **kwargs: qdrant_calls.append(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        weaviate,
        "connect_to_custom",
        lambda **kwargs: weaviate_calls.append(kwargs) or SimpleNamespace(),
    )
    Qdrant(config)
    adapter = Weaviate(config)
    assert qdrant_calls[0]["limits"].max_connections == config.connection_pool_limit
    assert "pool_size" not in qdrant_calls[0]  # Mutually exclusive with HTTP limits.
    assert (
        weaviate_calls[0]["additional_config"].connection.session_pool_maxsize
        == config.connection_pool_limit
    )
    assert adapter._search_slots._initial_value == config.connection_pool_limit


def test_pgvector_rejects_unhonored_collection_override_before_connect():
    from experiments.benchmarks.vectordb.adapters.pgvector import PgVector

    _, config = candidate_config(fixture_protocol(), "pgvector|smoke-hnsw", 2)
    with pytest.raises(ValueError, match="collection"):
        PgVector(replace(config, collection="not_the_table"))


def test_weaviate_missing_distance_never_becomes_perfect_similarity():
    from experiments.benchmarks.vectordb.adapters.weaviate import Weaviate

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
    from experiments.benchmarks.rag.evaluation import InputCompatibilityError

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
    from experiments.benchmarks.rag.evaluation import OutputIntegrityError

    server, _, _ = make_server()
    server.index.embedder.embed_query = lambda query: vector
    previous = len(server.adapter.calls)
    with pytest.raises(OutputIntegrityError):
        server.retrieve("question")
    assert len(server.adapter.calls) == previous


@pytest.mark.parametrize("counts", [[True], [0], [], [1, 2]])
def test_malformed_native_query_count_envelope_rejected(counts):
    from experiments.benchmarks.rag.evaluation import InputCompatibilityError

    server, embeddings, _ = make_server()
    server.index.embedder.input_token_counts = lambda texts, role: counts
    with pytest.raises(InputCompatibilityError):
        server.retrieve("question")
    assert not embeddings
