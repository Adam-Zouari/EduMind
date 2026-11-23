"""Retained-evidence recovery and dependence-aware paired comparison contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from vectordb_fixtures import (
    MemoryAdapter,
    fixture_protocol,
    freeze_test_sessions,
    make_server,
)

from benchmarks.vectordb.protocol import load_protocol
from benchmarks.vectordb.workload import SCOPES


def _paired_workload():
    from benchmarks.vectordb import validation

    workloads, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    workloads["operations"] = []
    return freeze_test_sessions(workloads, corpora, 3)


def _search_side(name, workloads, timing):
    return {
        "candidate": name,
        "requests": [
            {
                **row,
                "session_id": session["id"],
                "session_group_id": session["session_group_id"],
                "schedule_position": position,
                "operation": "search",
                "paired": row["query_id"] in session["selected_sources"],
                "quality_designated": row["attempt"] == 1,
                "status": "success",
                "latency_seconds": timing(session["id"], row),
                "ids": [],
                "session_isolation_violation": False,
                "filter_violation": False
                if row["condition"] == "selected-files"
                else None,
            }
            for session in workloads["sessions"]
            for position, row in enumerate(session["schedule"], 1)
        ],
    }


def test_paired_serial_latency_uses_session_group_draws_and_null_timers():
    from benchmarks.vectordb.comparisons import paired_operational

    workloads = _paired_workload()
    key = "operational.database_search_latency_ms_p50"
    protocol = replace(load_protocol(), ci_support={key: {"minimum_groups": 2}})
    a = _search_side(
        "a", workloads, lambda session, row: 1 if session == "fixture" else 8
    )
    b = _search_side(
        "b", workloads, lambda session, row: 2 if session == "fixture" else 10
    )
    b["requests"][0]["latency_seconds"] = None
    rows = paired_operational(a, b, workloads, protocol, "development")
    point = next(
        row
        for row in rows
        if row["metric"] == key and row["condition"] == "all-session"
    )
    assert point["difference"] == -2000
    assert point["left_completed_count"] == point["right_completed_count"] == 15
    assert point["left_timed_count"] == 15 and point["right_timed_count"] == 14
    assert point["right_missing_timing_count"] == 1 and point["right_failed_count"] == 0
    assert (
        point["independent_source_count"] == 2 and point["defined_resamples"] == 10000
    )


def test_paired_serial_schedule_rejects_duplicate_logical_operations():
    from benchmarks.vectordb.comparisons import paired_operational

    workloads = _paired_workload()
    a = _search_side("a", workloads, lambda *_: 1)
    b = _search_side("b", workloads, lambda *_: 1)
    b["requests"].append(b["requests"][0])
    with pytest.raises(ValueError, match="duplicate"):
        paired_operational(a, b, workloads, load_protocol(), "development")


@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "Missing scheduled"),
        ("duplicate", "Duplicate paired"),
        ("session_id", "Unexpected paired"),
        ("query_id", "Unexpected paired"),
        ("attempt", "Unexpected paired"),
    ],
)
def test_paired_search_ledger_validates_both_sides(scope, change, message):
    from benchmarks.vectordb.benchmark import paired_search_from_ledger

    workloads = _paired_workload()
    requests = _search_side("candidate", workloads, lambda *_: 1)["requests"]
    first, other, *_ = [
        i
        for i, row in enumerate(requests)
        if row["paired"] and row["condition"] == scope
    ]
    if change == "missing":
        del requests[first]
    elif change == "duplicate":
        requests[other] = dict(requests[first])
    else:
        requests[first][change] = 99 if change == "attempt" else "unexpected"
    with pytest.raises(ValueError, match=message):
        paired_search_from_ledger(workloads["sessions"], requests)


def test_paired_search_ledger_keeps_failed_untimed_and_unexecuted_distinct():
    from benchmarks.vectordb.benchmark import paired_search_from_ledger

    workloads = _paired_workload()
    requests = _search_side("candidate", workloads, lambda *_: 0.004)["requests"]
    for row in requests:
        if row["condition"] == "selected-files" and row["query_id"] == "q1":
            row["status"] = ("failed", "success", "not_executed")[row["attempt"] - 1]
            row["latency_seconds"] = None
    summary = paired_search_from_ledger(workloads["sessions"], requests)
    assert summary["unfiltered_scheduled"] == summary["filtered_scheduled"] == 6
    assert summary["filtered_completed"] == 4 and summary["filtered_timed"] == 3
    assert summary["filtered_failed"] == summary["filtered_not_executed"] == 1
    assert summary["filtered_missing_timing"] == 1
    assert summary["filtered_recall_at_20"] is None
    assert summary["filtered_recall_status"] == "incomplete"
    assert summary["filtered_search_latency_ms_p50"] == 4
    assert summary["filtered_search_latency_ms_p95"] == 4


def _run_retained(tmp_path, monkeypatch, *, failure=None, no_pairs=False):
    import yaml

    from benchmarks.vectordb import benchmark, recover, run, validation
    from benchmarks.vectordb.protocol import DEFAULT_PROTOCOL_PATH

    root = yaml.safe_load(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    root["candidates"]["smoke"] = root["candidates"]["smoke"][:1]
    root["measurement"] = {"monitor_storage": False, "monitor_server_memory": False}
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(root), encoding="utf-8")
    protocol = load_protocol(protocol_path)
    report_root = tmp_path / "reports"
    manifest = validation.manifest_path("smoke")
    if no_pairs:
        workloads, corpora = validation.load_inputs(manifest)
        for session in workloads["sessions"]:
            session["selected_sources"] = {}
        freeze_test_sessions(workloads, corpora, protocol.profile("smoke").repetitions)
        manifest = tmp_path / "no-pairs.json"
        manifest.write_text(json.dumps(workloads), encoding="utf-8")
    validation.prepare_report(manifest, "smoke", protocol, report_root=report_root)
    monkeypatch.setattr(run, "image_lock", lambda: {"image:chroma": "fixture"})

    def verify_image(*args, **kwargs):
        if failure == "pre-dispatch":
            raise RuntimeError("image verification failed before dispatch")

    monkeypatch.setattr(run, "verify_image", verify_image)
    monkeypatch.setattr(benchmark, "create", lambda name, config: MemoryAdapter(config))
    monkeypatch.setattr(
        benchmark,
        "check",
        lambda name, config, *args: (MemoryAdapter(config), {"status": "passed"}),
    )
    monkeypatch.setattr(
        benchmark, "startup", lambda *args: {"status": "success", "seconds": 1.0}
    )
    scorer = benchmark.score_server_evidence
    if failure == "scoring":

        def fail(*args):
            raise RuntimeError("offline scorer interrupted after sealed execution")

        monkeypatch.setattr(benchmark, "score_server_evidence", fail)
    if failure in {"snapshot-write", "missing-row"}:
        write_seal = recover.write_seal

        def save(path, kind, payload):
            if kind == "server":
                if failure == "snapshot-write":
                    raise OSError("snapshot write failed after dispatch")
                evidence = payload["evidence"]
                requests = list(evidence["requests"])
                requests.pop(
                    next(i for i, row in enumerate(requests) if row.get("paired"))
                )
                payload = {**payload, "evidence": {**evidence, "requests": requests}}
            return write_seal(path, kind, payload)

        monkeypatch.setattr(recover, "write_seal", save)
    code = run.main(
        [
            "--profile",
            "smoke",
            "--manifest",
            str(manifest),
            "--protocol",
            str(protocol_path),
            "--report-root",
            str(report_root),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--no-mlflow",
        ]
    )
    monkeypatch.setattr(benchmark, "score_server_evidence", scorer)
    assert code == (2 if failure else 0)
    source = next((tmp_path / "artifacts").rglob("retained-*-context.json")).parent
    return source


def _read_paired_export(source):
    path = source.parents[1] / "paired_search_comparison"
    columns = [
        "candidate",
        "status",
        "paired_sessions",
        "paired_questions",
        "scheduled_requests_per_side",
        "unfiltered_recall_at_20",
        "filtered_recall_at_20",
        "unfiltered_search_latency_ms_p50",
        "filtered_search_latency_ms_p50",
        "unfiltered_search_latency_ms_p95",
        "filtered_search_latency_ms_p95",
        "unfiltered_completed",
        "filtered_completed",
        "unfiltered_failed",
        "filtered_failed",
        "unfiltered_timed",
        "filtered_timed",
    ]
    frame = pd.read_parquet(path.with_suffix(".parquet"))
    assert list(frame.columns) == columns
    pd.testing.assert_frame_equal(
        pd.read_csv(path.with_suffix(".csv"), dtype=str, keep_default_na=False),
        frame.fillna("").astype(str),
    )
    return frame.iloc[0].to_dict()


@pytest.mark.parametrize("no_pairs", [False, True])
def test_paired_parent_projection_preserves_detailed_child_artifacts(
    tmp_path, monkeypatch, no_pairs
):
    source = _run_retained(tmp_path, monkeypatch, no_pairs=no_pairs)
    row = _read_paired_export(source)
    detail = json.loads((source / "paired_search.json").read_text(encoding="utf-8"))
    assert list(detail) == sorted(detail)
    assert row["candidate"] == "chroma|smoke-hnsw" and row["status"] == "success"
    scheduled = 0 if no_pairs else 2
    assert row["paired_sessions"] == row["paired_questions"] == scheduled
    assert row["scheduled_requests_per_side"] == scheduled
    assert row["unfiltered_completed"] == row["filtered_completed"] == scheduled
    assert all(value == detail[key] for key, value in row.items() if key in detail)
    assert detail["filtered_recall_status"] == (
        "inapplicable" if no_pairs else "scored"
    )
    assert detail["unfiltered_scheduled"] == detail["filtered_submitted"] == scheduled
    assert detail["filtered_not_executed"] == detail["filtered_missing_timing"] == 0
    requests = pd.read_parquet(source / "requests.parquet")
    assert len(requests[requests["operation"] == "search"]) == 5 + scheduled


def test_paired_parent_known_pre_dispatch_failure_keeps_scheduled_counts(
    tmp_path, monkeypatch
):
    source = _run_retained(tmp_path, monkeypatch, failure="pre-dispatch")
    assert not list(source.glob("retained-*-server.json"))
    row = _read_paired_export(source)
    assert row["status"] == "failed"
    assert row["paired_sessions"] == row["paired_questions"] == 2
    assert row["scheduled_requests_per_side"] == 2
    for side in ("unfiltered", "filtered"):
        for name in ("completed", "failed", "timed"):
            assert row[f"{side}_{name}"] == 0
        for name in ("recall_at_20", "search_latency_ms_p50", "search_latency_ms_p95"):
            assert row[f"{side}_{name}"] is None


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("snapshot-write", "Missing retained paired search execution evidence"),
        ("missing-row", "Missing scheduled paired search ledger rows"),
    ],
)
def test_paired_parent_rejects_missing_evidence(
    tmp_path, monkeypatch, failure, message
):
    source = _run_retained(tmp_path, monkeypatch, failure=failure)
    parent = source.parents[1]
    report = json.loads((parent / "summary.json").read_text(encoding="utf-8"))
    assert any(message in problem for problem in report["completion"]["problems"])
    assert not list(parent.glob("paired_search_comparison.*"))
    if failure == "missing-row":
        assert (source / "paired_search.json").is_file()


def test_recovery_after_offline_failure_never_dispatches_and_keeps_original_bytes(
    tmp_path, monkeypatch
):
    from benchmarks.vectordb import benchmark, recover, retrieval_run
    from edumind.common.artifacts import sha256_file

    source = _run_retained(tmp_path, monkeypatch, failure="scoring")
    row = _read_paired_export(source)
    assert row["status"] == "failed" and row["scheduled_requests_per_side"] == 2
    assert row["filtered_recall_at_20"] == 1 and row["filtered_completed"] == 2
    original = {
        str(path): sha256_file(path) for path in source.rglob("*") if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation-only recovery attempted live execution")

    monkeypatch.setattr(benchmark, "create", forbidden)
    monkeypatch.setattr(benchmark, "startup", forbidden)
    monkeypatch.setattr(retrieval_run, "prepare_integration", forbidden)
    monkeypatch.setattr(retrieval_run, "_execute_integration", forbidden)
    report = recover.recover(source, tmp_path / "recovered")
    assert report["complete"] and report["inference_executed"] is False
    assert report["metrics"]["quality.ann_recall_at_20.condition.all-session"] == 1
    assert {
        str(path): sha256_file(path) for path in source.rglob("*") if path.is_file()
    } == original
    assert (tmp_path / "recovered/requests.csv").is_file()
    with pytest.raises(ValueError, match="new output"):
        recover.recover(source, tmp_path / "recovered")


def test_recovery_rejects_changed_prediction_seal(tmp_path, monkeypatch):
    from benchmarks.vectordb import recover

    source = _run_retained(tmp_path, monkeypatch)
    snapshot = next(source.glob("retained-*-server.json"))
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["payload"]["evidence"]["requests"][0]["ids"] = ["forged"]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seal"):
        recover.recover(source, tmp_path / "recovered")


def test_recovery_rejects_changed_input_bytes_before_scoring(tmp_path, monkeypatch):
    from benchmarks.vectordb import recover

    source = _run_retained(tmp_path, monkeypatch)
    context = next(source.glob("retained-*-context.json"))
    envelope = json.loads(context.read_text(encoding="utf-8"))
    # Keep the actual project fixture untouched: add a test-only immutable dependency.
    dependency = tmp_path / "dependency.json"
    dependency.write_text("original", encoding="utf-8")
    from edumind.common.artifacts import sha256_file, stable_hash

    report = envelope["payload"]["input_report"]
    report["files"][str(dependency)] = sha256_file(dependency)
    report["fingerprint"] = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key
            not in {"fingerprint", "status", "errors", "metadata", "report_checksum"}
        }
    )
    envelope["payload"]["input_validation"] = report["fingerprint"]
    report["report_checksum"] = stable_hash(
        {key: value for key, value in report.items() if key != "report_checksum"}
    )
    # A fresh sealed context is needed because existing execution is immutable.
    new_source = tmp_path / "input-check"
    save, _ = recover.snapshot_writer(new_source, envelope["payload"])
    original = recover.read_seal(next(source.glob("retained-*-server.json")), "server")
    save("server", original["payload"]["evidence"])
    dependency.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="input bytes changed"):
        recover.recover(new_source, tmp_path / "recovered")


def test_selected_server_qualification_envelope_stops_inference_before_both_models():
    from benchmarks.rag.evaluation import InputCompatibilityError

    server, calls, _ = make_server()
    with pytest.raises(InputCompatibilityError, match="qualification"):
        server.retrieve("query", input_envelope={"maximum_query_tokens": 2})
    assert not calls
    rank_calls = []
    server, _, _ = make_server(
        reranker=SimpleNamespace(
            input_token_counts=lambda query, documents: [7] * len(documents),
            rank=lambda *args: rank_calls.append(True) or [0, 1, 2],
        )
    )
    evidence = {}
    with pytest.raises(InputCompatibilityError, match="qualification"):
        server.retrieve(
            "query",
            evidence=evidence,
            input_envelope={"maximum_query_tokens": 10, "maximum_reranker_tokens": 6},
        )
    assert not rank_calls and evidence["reranker_input_token_counts"] == [7, 7, 7]


def test_paired_operations_keep_repetitions_inside_reviewed_source_groups():
    from benchmarks.vectordb.comparisons import paired_operational

    operations = [
        {"id": name, "condition_id": "updates"} for name in ("doc1", "doc2", "doc3")
    ]

    def side(name, scale):
        return {
            "candidate": name,
            "requests": [
                {
                    "condition": "updates",
                    "operation_id": spec["id"],
                    "operation": kind,
                    "source_group": "same" if spec["id"] != "doc3" else "other",
                    "attempt": attempt,
                    "status": "failed" if spec["id"] == "doc3" else "success",
                    "latency_seconds": scale * attempt
                    if spec["id"] != "doc3"
                    else None,
                }
                for spec in operations
                for kind in ("insert", "update", "delete")
                for attempt in range(1, 4)
            ],
        }

    protocol = replace(
        load_protocol(),
        ci_support={
            "diagnostic.request_error_rate.update": {
                "minimum_groups": 2,
                "reviewed_by": "fixture",
                "reviewed_date": "2026-11-13",
            }
        },
    )
    rows = paired_operational(
        side("a", 1.0),
        side("b", 2.0),
        {"sessions": [], "operations": operations},
        protocol,
        "development",
    )
    rate = next(
        row for row in rows if row["metric"] == "diagnostic.request_error_rate.update"
    )
    assert rate["difference"] == 0 and rate["independent_source_count"] == 2
    assert rate["lower"] == rate["upper"] == 0 and rate["degenerate"]


def test_locked_recovery_preserves_failed_first_and_consumption_marker(
    tmp_path, monkeypatch
):
    from benchmarks.common.contracts import CandidateExecutionError
    from benchmarks.vectordb import benchmark, recover, run, validation
    from edumind.common.artifacts import sha256_file, stable_hash

    source = _run_retained(tmp_path, monkeypatch)
    context = recover.read_seal(
        next(source.glob("retained-*-context.json")), "context"
    )["payload"]
    # Controlled unit fixture for the locked boundary; no authoritative run is claimed.
    manifest = tmp_path / "locked-fixture.json"
    data = json.loads(validation.manifest_path("smoke").read_text(encoding="utf-8"))
    data["split"] = "locked-test"
    _, corpora = validation.load_inputs(validation.manifest_path("smoke"))
    freeze_test_sessions(data, corpora, 3)
    context["workloads"] = {name: data[name] for name in ("sessions", "operations")}
    manifest.write_text(json.dumps(data), encoding="utf-8")
    report = context["input_report"]
    old_manifest = context["manifest"]
    report["files"].pop(old_manifest)
    report["files"][str(manifest)] = sha256_file(manifest)
    report.update(
        profile="locked", manifest_path=str(manifest), inventory=[str(manifest)]
    )
    report["fingerprint"] = stable_hash(
        {
            key: value
            for key, value in report.items()
            if key
            not in {"fingerprint", "status", "errors", "metadata", "report_checksum"}
        }
    )
    report["report_checksum"] = stable_hash(
        {key: value for key, value in report.items() if key != "report_checksum"}
    )
    context.update(
        profile="locked", manifest=str(manifest), input_validation=report["fingerprint"]
    )
    marker = tmp_path / "locked-consumption.json"
    marker.write_text(
        json.dumps(
            {"input": context["input_validation"], "candidate": context["candidate"]}
        ),
        encoding="utf-8",
    )
    marker_hash = sha256_file(marker)
    context["locked_reservation"] = {"path": str(marker), "sha256": marker_hash}
    save, _ = recover.snapshot_writer(tmp_path / "locked-source", context)

    def factory(name, config):
        adapter = MemoryAdapter(config)
        search = adapter.search
        calls = 0

        def first_failure(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise TimeoutError("designated first request failed")
            return search(*args, **kwargs)

        adapter.search = first_failure
        return adapter

    monkeypatch.setattr(benchmark, "create", factory)
    payload, corpora = validation.load_inputs(manifest)
    with pytest.raises(CandidateExecutionError):
        benchmark.evaluate_candidate(
            context["candidate"],
            payload,
            corpora,
            load_protocol(),
            "locked",
            run.COMPOSE,
            conformance=False,
            restart=False,
            evidence_sink=lambda value: save("server", value),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("locked inference was rerun")

    monkeypatch.setattr(benchmark, "create", forbidden)
    recovered = recover.recover(
        tmp_path / "locked-source", tmp_path / "locked-evaluation"
    )
    assert not recovered["complete"] and recovered["inference_executed"] is False
    frame = __import__("pandas").read_parquet(
        tmp_path / "locked-evaluation/query_metrics.parquet"
    )
    first = json.loads(frame.loc[0, "metrics"])
    assert (
        first["quality.ann_recall_at_20"] is None
        and first["diagnostic.ranking_repeatability"] == 1 / 3
    )
    assert sha256_file(marker) == marker_hash
    marker.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="reservation"):
        recover.recover(tmp_path / "locked-source", tmp_path / "must-not-run")


def test_paired_application_quality_uses_question_macro():
    from types import SimpleNamespace

    from benchmarks.common.contracts import SampleResult
    from benchmarks.vectordb.comparisons import paired_quality

    key = "quality.evidence_unit_recall_at_5.condition.integration"

    def candidate(name, scores):
        return SimpleNamespace(
            candidate=name,
            samples=[
                SampleResult(
                    str(i),
                    {key: score},
                    None,
                    {
                        "independent_source_id": "group" + str(i // 2),
                        "source_document_ids": ["doc" + str(i // 2)],
                    },
                )
                for i, score in enumerate(scores)
            ],
        )

    rows = paired_quality(
        candidate("a", [1.0, 0.0, 1.0]),
        candidate("b", [0.0, 0.0, 1.0]),
        load_protocol(),
        "development",
    )
    assert rows[0]["left_value"] == pytest.approx(2 / 3)
    assert rows[0]["right_value"] == pytest.approx(1 / 3)
    assert rows[0]["difference"] == pytest.approx(1 / 3) and rows[0]["lower"] is None


def test_partial_paired_schedule_keeps_diagnostics_without_supported_bounds():
    from benchmarks.vectordb.comparisons import paired_operational

    workloads = _paired_workload()
    left, right = [_search_side(name, workloads, lambda *_: 1.0) for name in ("a", "b")]
    right["requests"] = [row for row in right["requests"] if row["query_id"] != "q1"]
    rows = paired_operational(left, right, workloads, load_protocol(), "development")
    assert all(row["status"] == "incomplete" and row["lower"] is None for row in rows)
    assert all(row["reason"] == "required_execution_incomplete" for row in rows)


def test_retained_integration_pool_can_be_scored_without_live_search(
    tmp_path, monkeypatch
):
    import copy

    from benchmarks.common.datasets import (
        load_manifest,
        manifest_content_checksum,
        validate_evidence,
    )
    from benchmarks.vectordb import (
        integration,
        recover,
        retrieval_run,
        validation,
    )

    server, _, _ = make_server()
    index = server.index
    index.tokenizer = SimpleNamespace(
        name="tiktoken:cl100k_base", spans=lambda text: [(0, len(text))]
    )
    questions = [
        {
            "kind": "question",
            "id": key,
            "question": key,
            "answerable": True,
            "evidence_type": "text",
            "evidence": [
                {
                    "id": "E1",
                    "evidence_type": "text",
                    "document_id": "doc",
                    "intervals": [{"start": 0, "end": 1}],
                }
            ],
        }
        for key in ("q1", "q2")
    ]
    for chunk in index.chunks[1:]:
        chunk.document_id = f"doc-{chunk.identifier}"
        chunk.start, chunk.end = 0, 1
    samples = [
        {
            "kind": "document",
            "id": chunk.document_id,
            "text": chunk.text,
            "independent_source_id": chunk.document_id,
        }
        for chunk in index.chunks
    ] + questions
    rag = {
        "name": "controlled-rag-fixture",
        "version": "1",
        "task": "rag",
        "split": "smoke",
        "source": "controlled-fixture",
        "license": "CC0-1.0",
        "revision": "test",
        "checksum": manifest_content_checksum(samples),
        "preprocessing_version": "evidence-units-v2",
        "split_seed": 42,
        "samples": samples,
    }
    (tmp_path / "rag.json").write_text(json.dumps(rag), encoding="utf-8")
    data = json.loads(validation.manifest_path("smoke").read_text(encoding="utf-8"))
    session = data["sessions"][0]
    session["selected_sources"] = {"q1": ["doc-b", "doc-c"]}
    for chunk, record in zip(index.chunks, session["records"], strict=True):
        record["text"] = chunk.text
        record["metadata"] = {
            "source_id": chunk.document_id,
            "start": chunk.start,
            "end": chunk.end,
        }
    spec = {"session_ids": ["fixture"], "manifest": "rag.json"}
    data["integration"] = spec
    manifest = tmp_path / "vectors.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    workloads, corpora = validation.load_inputs(manifest)
    freeze_test_sessions(workloads, corpora, 1)
    sources = validate_evidence(load_manifest(tmp_path / "rag.json"))
    questions = [{**q, **sources[q["id"]].metadata()} for q in questions]
    monkeypatch.setattr(
        integration, "create", lambda name, config: MemoryAdapter(config)
    )
    prepared = {
        "sessions": [
            {
                "index": index,
                "questions": questions,
                "session": workloads["sessions"][0],
                "corpus": corpora["fixture"],
            }
        ],
        "global_ids": {
            identifier for corpus in corpora.values() for identifier in corpus.ids
        },
        "spec": spec,
        "retrieval": "dense|none",
        "retrieval_protocol": SimpleNamespace(pool_size=20),
        "reranker_factory": lambda: None,
    }
    retained = []
    result = retrieval_run.evaluate_integration(
        "chroma|smoke-hnsw",
        prepared,
        fixture_protocol(),
        "smoke",
        evidence_sink=retained.append,
    )
    raw = retained[0]
    recover.validate_integration_ledger(
        raw,
        {**workloads, "manifest_path": str(manifest)},
        corpora,
        fixture_protocol(),
        "smoke",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("recovery attempted live search")

    monkeypatch.setattr(integration, "create", forbidden)
    rescored = retrieval_run.score_integration_evidence(
        raw, fixture_protocol(), "smoke", tokenizer=index.tokenizer
    )
    assert result[2] == rescored[2]
    selected = next(
        row
        for row in rescored[5]["integration_query_metrics"]
        if row["condition"] == "selected-files"
    )
    assert selected["quality_status"] == "inapplicable"
    assert selected["quality_reason"] == "selected_scope_excludes_required_evidence"
    assert selected["metrics"] == {} and selected["latency_seconds"] is not None
    assert (
        result[2]["quality.evidence_unit_recall_at_3.condition.integration-all-session"]
        == 1.0
    )
    broken = copy.deepcopy(raw)
    broken["sessions"][0]["rankings"][0]["evidence"]["ranking"] = [0, 0, 1]
    with pytest.raises(ValueError, match="permutation"):
        recover.validate_integration_ledger(
            broken,
            {**workloads, "manifest_path": str(manifest)},
            corpora,
            fixture_protocol(),
            "smoke",
        )
