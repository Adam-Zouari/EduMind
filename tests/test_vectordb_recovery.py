"""Retained-evidence recovery and dependence-aware paired comparison contracts."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.benchmarks.vectordb.protocol import load_protocol


def _load(side, durations, successes):
    requests, windows = [], []
    for window, (duration, count) in enumerate(zip(durations, successes, strict=True)):
        windows.append(
            {
                "condition": "load",
                "window": window,
                "wall_seconds": duration,
                "successful": count,
                "submitted": 3,
                "complete": True,
            }
        )
        for attempt in range(1, 4):
            requests.append(
                {
                    "condition": "load",
                    "operation": "search",
                    "query_id": "q",
                    "source_group": "source",
                    "window": window,
                    "attempt": attempt,
                    "status": "success" if attempt <= count else "failed",
                    "latency_seconds": duration / 10 if attempt <= count else None,
                    "filter_violation": None,
                }
            )
    return {"candidate": side, "requests": requests, "windows": windows}


def test_paired_load_uses_ratio_of_totals_and_aligned_window_draws():
    from experiments.benchmarks.vectordb.comparisons import paired_operational

    protocol = replace(
        load_protocol(),
        ci_support={
            "operational.qps": {
                "minimum_groups": 2,
                "reviewed_by": "fixture",
                "reviewed_date": "2026-11-13",
            }
        },
    )
    condition = {
        "id": "load",
        "mode": "load",
        "filters": {},
        "windows": 2,
        "query_ids": ["q"],
        "concurrency": 2,
        "independent_window_groups": ["w1", "w2"],
    }
    left = _load("left", [1.0, 3.0], [3, 3])
    right = _load("right", [2.0, 6.0], [3, 3])
    rows = paired_operational(
        left,
        right,
        {"conditions": [condition], "operations": []},
        protocol,
        "development",
    )
    qps = next(row for row in rows if row["metric"] == "operational.qps")
    assert qps["difference"] == 0.75  # 6/4 - 6/8, not mean window QPS differences.
    assert qps["independent_source_count"] == 2 and qps["defined_resamples"] == 10000
    assert qps["lower"] is not None and qps["upper"] is not None


def test_paired_load_missing_review_keeps_point_but_null_bounds():
    from experiments.benchmarks.vectordb.comparisons import paired_operational

    condition = {
        "id": "load",
        "mode": "load",
        "filters": {},
        "windows": 2,
        "query_ids": ["q"],
        "concurrency": 2,
    }
    rows = paired_operational(
        _load("a", [1.0, 1.0], [0, 0]),
        _load("b", [1.0, 1.0], [3, 3]),
        {"conditions": [condition], "operations": []},
        load_protocol(),
        "development",
    )
    qps = next(row for row in rows if row["metric"] == "operational.qps")
    assert qps["difference"] == -3 and qps["lower"] is None
    assert qps["reason"] in {
        "ci_support_not_frozen",
        "load_window_dependence_design_not_frozen",
    }


def test_paired_load_rejects_duplicate_or_unaligned_logical_operations():
    from experiments.benchmarks.vectordb.comparisons import paired_operational

    condition = {
        "id": "load",
        "mode": "load",
        "filters": {},
        "windows": 2,
        "query_ids": ["q"],
        "concurrency": 2,
    }
    left, right = _load("a", [1.0, 1.0], [3, 3]), _load("b", [1.0, 1.0], [3, 3])
    right["requests"].append(right["requests"][0])
    with pytest.raises(ValueError, match="duplicate"):
        paired_operational(
            left,
            right,
            {"conditions": [condition], "operations": []},
            load_protocol(),
            "development",
        )


def _run_retained(tmp_path, monkeypatch, *, fail_scoring=False):
    import yaml
    from test_vectordb_migration import MemoryAdapter

    from experiments.benchmarks.vectordb import benchmark, run, validation
    from experiments.benchmarks.vectordb.protocol import DEFAULT_PROTOCOL_PATH

    root = yaml.safe_load(DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    root["candidates"]["smoke"] = root["candidates"]["smoke"][:1]
    root["measurement"] = {"monitor_storage": False, "monitor_server_memory": False}
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(root), encoding="utf-8")
    protocol = load_protocol(protocol_path)
    report_root = tmp_path / "reports"
    validation.prepare_report(
        validation.manifest_path("smoke"), "smoke", protocol, report_root=report_root
    )
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
    scorer = benchmark.score_server_evidence
    if fail_scoring:

        def fail(*args):
            raise RuntimeError("offline scorer interrupted after sealed execution")

        monkeypatch.setattr(benchmark, "score_server_evidence", fail)
    code = run.main(
        [
            "--profile",
            "smoke",
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
    assert code == (2 if fail_scoring else 0)
    source = next((tmp_path / "artifacts").rglob("retained-*-context.json")).parent
    return source


def test_recovery_after_offline_failure_never_dispatches_and_keeps_original_bytes(
    tmp_path, monkeypatch
):
    from edumind.common.artifacts import sha256_file
    from experiments.benchmarks.vectordb import benchmark, recover, retrieval_run

    source = _run_retained(tmp_path, monkeypatch, fail_scoring=True)
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
    assert report["metrics"]["quality.ann_recall_at_20.condition.serial"] == 1
    assert {
        str(path): sha256_file(path) for path in source.rglob("*") if path.is_file()
    } == original
    assert (tmp_path / "recovered/requests.csv").is_file()
    with pytest.raises(ValueError, match="new output"):
        recover.recover(source, tmp_path / "recovered")


def test_recovery_rejects_changed_prediction_seal(tmp_path, monkeypatch):
    from experiments.benchmarks.vectordb import recover

    source = _run_retained(tmp_path, monkeypatch)
    snapshot = next(source.glob("retained-*-server.json"))
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["payload"]["evidence"]["requests"][0]["ids"] = ["forged"]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="seal"):
        recover.recover(source, tmp_path / "recovered")


def test_recovery_rejects_changed_input_bytes_before_scoring(tmp_path, monkeypatch):
    from experiments.benchmarks.vectordb import recover

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
    from test_vectordb_migration import make_server

    from experiments.benchmarks.rag.evaluation import InputCompatibilityError

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
    from experiments.benchmarks.vectordb.comparisons import paired_operational

    operations = [
        {"id": name, "condition_id": "updates"} for name in ("doc1", "doc2", "doc3")
    ]

    def side(name, scale):
        return {
            "candidate": name,
            "windows": [],
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
        {"conditions": [], "operations": operations},
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
    from test_vectordb_migration import MemoryAdapter

    from edumind.common.artifacts import sha256_file, stable_hash
    from experiments.benchmarks.common.contracts import CandidateExecutionError
    from experiments.benchmarks.vectordb import benchmark, recover, run, validation

    source = _run_retained(tmp_path, monkeypatch)
    context = recover.read_seal(
        next(source.glob("retained-*-context.json")), "context"
    )["payload"]
    # Controlled unit fixture for the locked boundary; no authoritative run is claimed.
    manifest = tmp_path / "locked-fixture.json"
    data = json.loads(validation.manifest_path("smoke").read_text(encoding="utf-8"))
    data["split"] = "locked-test"
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
            if calls == 2:
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


def test_paired_quality_uses_document_macro_not_question_mean():
    from types import SimpleNamespace

    from experiments.benchmarks.common.contracts import SampleResult
    from experiments.benchmarks.vectordb.comparisons import paired_quality

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
                        "source_group": "group" + str(i // 2),
                        "document_id": "doc" + str(i // 2),
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
    assert rows[0]["left_value"] == 0.75 and rows[0]["right_value"] == 0.5
    assert rows[0]["difference"] == 0.25 and rows[0]["lower"] is None


def test_partial_paired_schedule_keeps_diagnostics_without_supported_bounds():
    from experiments.benchmarks.vectordb.comparisons import paired_operational

    condition = {
        "id": "load",
        "mode": "load",
        "filters": {},
        "windows": 2,
        "query_ids": ["q"],
        "independent_window_groups": ["one", "two"],
    }
    left, right = _load("a", [1.0, 1.0], [3, 3]), _load("b", [1.0, 1.0], [3, 3])
    right["requests"].pop()
    rows = paired_operational(
        left,
        right,
        {"conditions": [condition], "operations": []},
        load_protocol(),
        "development",
    )
    assert all(row["status"] == "incomplete" and row["lower"] is None for row in rows)
    assert all(row["reason"] == "required_execution_incomplete" for row in rows)


def test_retained_integration_pool_can_be_scored_without_live_search(
    tmp_path, monkeypatch
):
    import copy

    from test_vectordb_migration import MemoryAdapter, fixture_protocol, make_server

    from experiments.benchmarks.common.datasets import manifest_content_checksum
    from experiments.benchmarks.vectordb import (
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
            "document_id": "doc",
            "answerable": True,
            "evidence_type": "text",
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
        for key in ("q1", "q2")
    ]
    samples = [{"kind": "document", "id": "doc", "text": "abc"}, *questions]
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
    for i, record in enumerate(data["corpora"][0]["records"]):
        record["text"] = "abc"[i]
        record["metadata"] = {"source_id": "doc", "start": i, "end": i + 1}
    spec = {"corpus": "fixture", "manifest": "rag.json", "filters": {}}
    data["integration"] = spec
    manifest = tmp_path / "vectors.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    workloads, corpora = validation.load_inputs(manifest)
    monkeypatch.setattr(
        integration, "create", lambda name, config: MemoryAdapter(config)
    )
    prepared = {
        "index": index,
        "questions": questions,
        "spec": spec,
        "retrieval": "dense|none",
        "retrieval_protocol": SimpleNamespace(pool_size=20),
        "reranker_factory": lambda: None,
        "source_groups": {"q1": "doc", "q2": "doc"},
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
    broken = copy.deepcopy(raw)
    broken["rankings"][0]["evidence"]["ranking"] = [0, 0, 1]
    with pytest.raises(ValueError, match="permutation"):
        recover.validate_integration_ledger(
            broken,
            {**workloads, "manifest_path": str(manifest)},
            corpora,
            fixture_protocol(),
            "smoke",
        )
