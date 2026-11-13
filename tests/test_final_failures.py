"""Final failure/recovery regressions using committed smoke data and test doubles.

Three-attempt tests exercise a locked-shaped ledger only: no held-out data,
model weights, services, real judge, or locked invocation is used.
"""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_generation_benchmark import Backend, FixtureJudge, Monitor
from test_generation_recovery import calibrated_inputs as base_calibrated_inputs
from test_generation_recovery import fixture_transport

from edumind.common.artifacts import atomic_write_json, sha256_file
from experiments.benchmarks.common import runner as common_runner
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.rag.evaluation import Chunk
from experiments.benchmarks.rag.final import benchmark, run, worker
from experiments.benchmarks.rag.final.validation import manifest_path
from experiments.benchmarks.rag.generation.evidence import (
    load_execution,
    save_evaluation,
)
from experiments.benchmarks.rag.generation.judge import SemanticJudge
from experiments.benchmarks.rag.generation.metrics import (
    PRIMARY_METRICS,
    REPEAT_KEYS,
    parse_response,
    question_statistics,
)
from experiments.benchmarks.rag.generation.validation import questions_from_manifest

GENERATOR = "qwen3-0.6b-direct"
STREAM_MARKER = "test-only native stream retained after inference"


class FaultBackend(Backend):
    def __init__(self, fault):
        super().__init__()
        self.fault = fault
        self.fired = False
        self.returned = []

    def infer(self, prepared, seed):
        produced = super().infer(prepared, seed)
        produced["raw_text"] = f"{STREAM_MARKER}; call={len(self.calls)}; seed={seed}"
        if self.fault == "malformed" and len(self.calls) > 1:
            produced["visible_text"] = "test-only malformed response, not JSON"
        self.returned.append(deepcopy(produced))
        return produced

    def placement(self):
        if self.fault == "placement" and len(self.calls) == 2 and not self.fired:
            self.fired = True
            return {"status": "offload_detected", "detail": "test-only late placement"}
        return super().placement()

    def synchronize(self):
        if self.fault == "synchronize" and len(self.calls) == 2 and not self.fired:
            self.fired = True
            raise RuntimeError("test-only late synchronization failure")

    def close(self):
        super().close()
        if self.fault == "cleanup":
            raise RuntimeError("test-only generator cleanup failure")


@pytest.fixture
def final_harness(tmp_path, monkeypatch):
    """Run real worker assembly and parent-owned sealing with external doubles."""
    manifest = load_manifest(manifest_path("smoke"))
    protocols = {key: loader() for key, loader in worker.LOADERS.items()}
    chunks = [
        Chunk(row["id"] + ":whole", row["id"], row["text"], 0, len(row["text"]), 10)
        for row in manifest.samples
        if row.get("kind") == "document"
    ]
    index = SimpleNamespace(
        chunks=chunks,
        vectors=np.ones((len(chunks), 2)),
        embedder=None,
    )
    composition_path = tmp_path / "composition.json"
    atomic_write_json(composition_path, {"test_only": True})
    database_decision = tmp_path / "database-decision.json"
    atomic_write_json(database_decision, {"test_only": True})
    composition = SimpleNamespace(
        path=composition_path,
        upstream={"database": {"path": str(database_decision)}},
        checksum="test-only-final-composition",
        protocols={key: value.meta.checksum for key, value in protocols.items()},
        system={
            "chunking_embedding": "token-256-32|Alibaba-NLP/gte-modernbert-base",
            "retrieval": "dense|none",
            "generator": GENERATOR,
            "database": "chroma|test-only",
            "residency": "resident",
            "top_k": 3,
        },
    )
    monkeypatch.setattr(benchmark, "release_runtime_memory", lambda: None)
    parent_worker = run._worker

    def execute_case(
        backend=None,
        *,
        profile="smoke",
        fail_retrieval_calls=(),
        directory=None,
        selected_protocols=None,
        selected_plan=None,
        model_lock=None,
    ):
        selected_protocols = dict(selected_protocols or protocols)
        if profile == "locked":
            # Keep this test-only three-seed ledger cheap. Both composed execution
            # profiles explicitly agree with the plan; production YAML is untouched.
            for name in ("final_rag", "generation"):
                protocol = selected_protocols[name]
                selected_protocols[name] = replace(
                    protocol,
                    meta=replace(
                        protocol.meta,
                        profiles={
                            **protocol.meta.profiles,
                            "locked": replace(
                                protocol.profile("locked"), bootstrap_resamples=0
                            ),
                        },
                    ),
                )
        generation = selected_protocols["generation"]
        execution = selected_protocols["final_rag"].profile(profile)
        plan = selected_plan or BenchmarkPlan(
            "rag",
            "final",
            profile,
            manifest.name,
            ("selected-system",),
            selected_protocols["final_rag"].meta.seed,
            execution.repetitions,
            execution.bootstrap_resamples,
            execution.warmups,
            {"device": execution.device, "dtype": execution.dtype},
        )
        backend = backend or Backend()
        directory = Path(directory or tmp_path / "execution")
        calls = []

        class Server:
            identity = "test-only-selected-live-server"
            reranker = None
            closed = False

            def __init__(self):
                self.index = index

            def prepare(self):
                return {
                    "status": "passed",
                    "identity": self.identity,
                    "test_only": True,
                }

            def retrieve(self, question, *, evidence):
                calls.append(question["id"])
                if len(calls) in fail_retrieval_calls:
                    raise RuntimeError("test-only selected-server request failure")
                evidence.update(
                    pool=list(range(len(chunks))),
                    ranking=list(range(len(chunks))),
                    database_latency_seconds=0.001,
                    identity=self.identity,
                )

            def close(self):
                self.closed = True

        server = Server()
        monkeypatch.setattr(
            worker,
            "benchmark_objects_from_payload",
            lambda _: (
                "selected-system",
                manifest,
                model_lock or {},
                plan,
            ),
        )
        monkeypatch.setattr(
            worker,
            "LOADERS",
            {
                key: lambda _path, value=value: value
                for key, value in selected_protocols.items()
            },
        )
        monkeypatch.setattr(worker, "load_composition", lambda *_: composition)
        monkeypatch.setattr(worker, "build_index", lambda *a, **kw: index)
        monkeypatch.setattr(worker, "candidate_config", lambda *a: ("chroma", object()))
        monkeypatch.setattr(worker, "reranker_for", lambda *a, **kw: None)
        monkeypatch.setattr(worker, "SelectedServer", lambda *a, **kw: server)

        def execute_with_test_runtime(*args, **kwargs):
            return benchmark.execute(
                *args,
                backend_factory=lambda *a: backend,
                monitor_factory=Monitor,
                **kwargs,
            )

        monkeypatch.setattr(worker, "execute", execute_with_test_runtime)

        def execute_with_test_supervision(_script, payload, **kwargs):
            result = worker.execute_payload(payload)
            result["_worker_supervision"] = {
                "test_only": True,
                "peak_vram_mb": 1900 if plan.settings["device"] == "cuda" else None,
            }
            return result

        monkeypatch.setattr(run, "run_json_worker", execute_with_test_supervision)
        raw = parent_worker(
            manifest,
            model_lock or {},
            plan,
            selected_protocols,
            composition,
            directory,
        )
        path = directory / "generation_execution.json"
        saved, binding = load_execution(path, generation)
        assert saved == raw
        assert binding["prediction_sha256"] == sha256_file(path)
        checkpoint = json.loads(
            (directory / "generation_progress.json").read_text(encoding="utf-8")
        )
        for key in ("records", "error", "schedule_checksum"):
            assert checkpoint[key] == raw[key]
        assert saved["supervisor_resource_evidence"]["test_only"] is True
        assert (
            saved["operational"]["peak_ram_mib"]
            == checkpoint["operational"]["peak_ram_mib"]
        )
        if plan.settings["device"] == "cuda":
            assert saved["operational"]["peak_vram_mib"] == max(
                1900,
                checkpoint["operational"]["peak_vram_mib"],
            )
        else:
            assert saved["operational"]["peak_vram_mib"] is None
        assert server.closed
        return SimpleNamespace(
            raw=saved,
            binding=binding,
            path=path,
            backend=backend,
            server=server,
            server_calls=calls,
            protocol=generation,
            retrieval_protocol=selected_protocols["retrieval"],
            manifest=manifest,
        )

    return SimpleNamespace(
        execute=execute_case,
        manifest=manifest,
        protocols=protocols,
        composition=composition,
        tmp_path=tmp_path,
    )


def _persist_scoring(case, *, judge=None):
    path = case.path.parent / "test-only-evaluation.json"
    try:
        result = benchmark.score(
            case.raw,
            case.manifest,
            case.protocol,
            case.retrieval_protocol,
            judge=judge,
        )
    except CandidateExecutionError as error:
        save_evaluation(path, error=error, provenance=case.binding)
    else:
        save_evaluation(path, result=result, provenance=case.binding)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fault", ["placement", "synchronize"])
def test_late_postchecks_retain_failed_stream_in_sealed_execution(final_harness, fault):
    backend = FaultBackend(fault)
    case = final_harness.execute(backend)
    failed = case.raw["records"][0]["attempts"][0]
    produced = backend.returned[1]  # Warmup precedes the failed measured call.
    assert failed["request_status"] == "failed"
    assert failed["failure_stage"] == "post_inference_integrity"
    assert failed["raw_text"] == produced["raw_text"]
    assert STREAM_MARKER in failed["raw_text"]
    assert failed["visible_text"] == produced["visible_text"]
    assert failed["workload"] == produced["workload"]
    assert failed["timing"]["ttft_seconds"] == produced["timing"]["ttft_seconds"]
    assert len(failed["token_event_seconds"]) == len(produced["token_event_seconds"])
    assert failed["retrieval"]["ranking"]
    assert failed["error"]
    assert failed.get("request_latency_seconds") is None
    if fault == "placement":
        assert case.raw["error"]
        assert len(backend.calls) == 2
        later = case.raw["records"][1]["attempts"][0]
        assert later["outcome"] == "not_executed"

    scored = _persist_scoring(case)
    assert scored["status"] == "incomplete"
    retained = scored["artifacts"]["generation_evidence"]["records"][0]["attempts"][0]
    assert retained["raw_text"] == produced["raw_text"]
    assert retained["visible_text"] == produced["visible_text"]
    assert retained["request_status"] == "failed"
    exported = json.loads(scored["artifacts"]["final_attempts"][0]["evidence"])
    assert exported["raw_text"] == produced["raw_text"]


def test_cleanup_error_preserves_sealed_predictions_and_measurements(final_harness):
    backend = FaultBackend("cleanup")
    case = final_harness.execute(backend)
    assert "test-only generator cleanup failure" in case.raw["cleanup_error"]
    assert case.raw["error"]
    assert case.raw["monitoring_complete"] is True
    assert case.raw["operational"]["peak_ram_mib"] == 100
    assert case.raw["resource_samples"]
    for record in case.raw["records"]:
        item = record["attempts"][0]
        assert item["request_status"] == "completed"
        assert STREAM_MARKER in item["raw_text"]
        assert item["request_latency_seconds"] >= 0
    scored = _persist_scoring(case)
    assert scored["status"] == "incomplete"
    assert scored["operational"]["peak_ram_mib"] == 100
    assert "cleanup failure" in scored["error"]
    retained = scored["artifacts"]["generation_evidence"]
    assert retained["cleanup_error"] == case.raw["cleanup_error"]
    assert (
        retained["records"][0]["attempts"][0]["raw_text"]
        == backend.returned[1]["raw_text"]
    )


def test_malformed_completion_is_persisted_request_failure_not_fast_success(
    final_harness,
):
    case = final_harness.execute(FaultBackend("malformed"))
    for record in case.raw["records"]:
        item = record["attempts"][0]
        assert (
            item["outcome"] == "completed"
        )  # Model completion is independently known.
        assert parse_response(item["visible_text"])[0] is None
        assert item["request_status"] == "failed"
        assert item["failure_stage"] == "output_schema"
        assert item.get("request_latency_seconds") is None
        assert item["request_elapsed_seconds"] >= 0
        assert STREAM_MARKER in item["raw_text"]
    scored = _persist_scoring(case)
    assert scored["metrics"]["reliability.request_failure_rate"] == 1
    assert scored["metrics"]["validity.malformed_output_rate"] == 1
    assert scored["operational"]["request_p50_seconds"] is None
    assert scored["operational"]["request_p95_seconds"] is None
    assert scored["operational"]["retrieval_p50_seconds"] is not None
    assert scored["operational"]["server_call_p50_seconds"] == 0.001
    assert all(
        row["request_status"] == "failed"
        for row in scored["artifacts"]["final_attempts"]
    )


@pytest.mark.parametrize("answerable", [True, False])
@pytest.mark.parametrize("failed_count", [1, 3])
def test_upstream_failures_keep_scheduled_validity_and_zero_failed_pairs(
    final_harness,
    answerable,
    failed_count,
):
    first_call = 2 if answerable else 5  # One warmup, then three requests per question.
    case = final_harness.execute(
        profile="locked",
        fail_retrieval_calls=range(first_call, first_call + failed_count),
    )
    scored = _persist_scoring(case, judge=FixtureJudge())
    records = scored["artifacts"]["generation_evidence"]["records"]
    record = next(row for row in records if row["question"]["answerable"] is answerable)
    validity = (
        "validity.response_validity_rate"
        if answerable
        else "validity.refusal_validity_rate"
    )
    stats, support = question_statistics(record)
    assert stats[validity] == pytest.approx((3 - failed_count) / 3)
    assert support[validity]["eligible_observations"] == 3
    assert support[validity]["contributing_observations"] == 3
    assert scored["parameters"]["execution_complete"] is True
    for item in record["attempts"][:failed_count]:
        assert item["request_status"] == "failed"
        assert item["generation_status"] == "not_started"
        assert item["generation_eligible"] is False
        assert item["metrics"][validity]["value"] == 0
        for key in (
            "reliability.runtime_error_rate",
            "reliability.timeout_rate",
            "reliability.context_limit_reached_rate",
        ):
            assert item["metrics"][key]["status"] == "inapplicable"
            assert item["metrics"][key]["value"] is None
        assert (
            item["metrics"]["validity.malformed_output_rate"]["status"]
            == "inapplicable"
        )
    for key in (
        "reliability.runtime_error_rate",
        "reliability.timeout_rate",
        "reliability.context_limit_reached_rate",
    ):
        assert support[key]["eligible_observations"] == 3 - failed_count
        assert stats[key] == (None if failed_count == 3 else 0)
    failed_seeds = {item["seed"] for item in record["attempts"][:failed_count]}
    assert len(record["pairs"]) == 3
    for pair in record["pairs"]:
        if failed_seeds.intersection(pair["seeds"]):
            for key in REPEAT_KEYS:
                if key == "repeatability.citation_agreement" and not answerable:
                    assert pair["metrics"][key]["status"] == "inapplicable"
                else:
                    assert pair["metrics"][key]["status"] == "scored"
                    assert pair["metrics"][key]["value"] == 0
    sample = next(
        row for row in scored["samples"] if row["sample_id"] == record["question"]["id"]
    )
    assert sample["metrics"][validity] == pytest.approx((3 - failed_count) / 3)


def _fail_external(*args, **kwargs):
    pytest.fail(
        "Evaluation-only recovery must not load models, infer, reserve locked data, or start a worker"
    )


def test_final_cli_resume_reuses_accepted_original_journal_without_inference(
    final_harness, monkeypatch
):
    harness = final_harness
    base = harness.protocols["generation"]
    questions = questions_from_manifest(harness.manifest)
    generation, *_ = base_calibrated_inputs.__wrapped__(
        (base, harness.manifest, questions), harness.tmp_path
    )
    protocols = {**harness.protocols, "generation": generation}
    harness.composition.protocols = {
        key: value.meta.checksum for key, value in protocols.items()
    }
    case = harness.execute(selected_protocols=protocols)
    transport_calls = []
    response_transport = fixture_transport(questions[0])

    def transport(request, **kwargs):
        transport_calls.append(request)
        return response_transport(request, **kwargs)

    original = SemanticJudge(generation.judge_contract_path, transport).with_journal(
        case.path.parent / "judge-journal",
        case.raw["schedule_checksum"],
    )
    initial = benchmark.score(
        case.raw, case.manifest, generation, case.retrieval_protocol, judge=original
    )
    assert initial[3]["execution_complete"] is True
    assert len(transport_calls) == 1
    journal_files = sorted(original.journal_directory.glob("*.json"))
    assert len(journal_files) == 2  # Identity plus one accepted response.
    accepted = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in journal_files
        if path.name != "identity.json"
    )
    assert accepted["entry"]["status"] == "evaluated"
    execution_bytes = case.path.read_bytes()
    schedule_bytes = case.path.with_name("generation_schedule.json").read_bytes()

    def never_transport(request, **kwargs):
        pytest.fail(
            "Accepted Final judge evaluation was requested again during recovery"
        )

    monkeypatch.setattr(run, "load_composition", lambda *_: harness.composition)
    # This is a journal-recovery test, not an upstream advancement/lineage test.
    monkeypatch.setattr(
        run,
        "selected_candidates",
        lambda *a, **kw: (harness.composition.system["database"],),
    )
    monkeypatch.setattr(
        run,
        "verify_inputs",
        lambda *a, **kw: (harness.tmp_path / "test-only-report.json", {}),
    )
    monkeypatch.setattr(
        run,
        "_judge",
        lambda *a: SemanticJudge(generation.judge_contract_path, never_transport),
    )
    for name in (
        "_worker",
        "load_selected_model_lock",
        "reserve_locked",
        "run_benchmark",
        "_verify_model_lineage",
    ):
        monkeypatch.setattr(run, name, _fail_external)
    assert (
        run.main(
            [
                "--profile",
                "smoke",
                "--device",
                "cpu",
                "--no-mlflow",
                "--generation-protocol",
                str(generation.meta.source_path),
                "--resume-evaluation",
                str(case.path),
            ]
        )
        == 0
    )
    assert len(transport_calls) == 1
    assert case.path.read_bytes() == execution_bytes
    assert (
        case.path.with_name("generation_schedule.json").read_bytes() == schedule_bytes
    )
    (recovered_path,) = case.path.parent.glob("final_evaluation-*.json")
    recovered = json.loads(recovered_path.read_text(encoding="utf-8"))
    assert recovered["status"] == "complete"
    assert (
        recovered["evaluation_provenance"]["prediction_sha256"]
        == case.binding["prediction_sha256"]
    )
    for key in PRIMARY_METRICS:
        assert recovered["metrics"][key] == initial[2][key]


def test_complete_final_smoke_cli_persists_one_system_and_equivalent_tables(
    final_harness, monkeypatch, capsys
):
    harness = final_harness
    monkeypatch.setattr(run, "PROJECT_ROOT", harness.tmp_path)
    model_path = harness.tmp_path / "data/benchmarks/models/selected.json"
    model_lock = {
        "Alibaba-NLP/gte-modernbert-base": {"revision": "test-only-fixture"},
        harness.protocols["generation"].model_id(GENERATOR): {
            "revision": "test-only-fixture"
        },
    }
    atomic_write_json(model_path, {"models": model_lock, "test_only": True})
    report_path = harness.tmp_path / "test-only-data-report.json"
    atomic_write_json(report_path, {"status": "passed", "test_only": True})
    monkeypatch.setattr(run, "load_composition", lambda *_: harness.composition)
    # Keep component-authority validation outside this execution/artifact test.
    monkeypatch.setattr(
        run,
        "selected_candidates",
        lambda *a, **kw: (harness.composition.system["database"],),
    )
    monkeypatch.setattr(run, "_verify_model_lineage", lambda *a, **kw: None)
    monkeypatch.setattr(
        run, "verify_inputs", lambda *a, **kw: (report_path, {"status": "passed"})
    )
    monkeypatch.setattr(run, "load_selected_model_lock", lambda *a, **kw: model_lock)
    monkeypatch.setattr(run, "_judge", lambda *a: None)
    monkeypatch.setattr(run, "reserve_locked", _fail_external)
    monkeypatch.setattr(
        common_runner, "hardware_summary", lambda: {"test_only": True, "device": "cpu"}
    )
    monkeypatch.setattr(
        common_runner,
        "git_provenance",
        lambda _: {"commit": "test-only", "dirty": True},
    )
    worker_calls, runner_results = [], []

    def selected_worker(manifest, lock, plan, protocols, composition, directory):
        worker_calls.append((manifest.fingerprint, plan, composition.checksum))
        case = harness.execute(
            selected_protocols=protocols,
            selected_plan=plan,
            model_lock=lock,
            directory=directory,
        )
        assert len(case.server_calls) == 1 + len(case.raw["records"])
        assert case.backend.closed and case.server.closed
        return case.raw

    actual_runner = common_runner.run_benchmark

    def run_in_temporary_artifacts(*args, **kwargs):
        result = actual_runner(
            *args, artifact_root=harness.tmp_path / "artifacts", **kwargs
        )
        runner_results.append(result)
        return result

    monkeypatch.setattr(run, "_worker", selected_worker)
    monkeypatch.setattr(run, "run_benchmark", run_in_temporary_artifacts)
    status = run.main(["--profile", "smoke", "--device", "cpu", "--no-mlflow"])
    if status != 0:
        failures = []
        for result in runner_results:
            summary = json.loads(
                (result.artifact_directory / "summary.json").read_text(encoding="utf-8")
            )
            failures.append(
                {
                    "summary": str(result.artifact_directory / "summary.json"),
                    "completion": summary["completion"],
                    "candidate_errors": [
                        row.get("error") for row in summary["candidates"]
                    ],
                }
            )
        pytest.fail(f"Final smoke CLI returned {status}: {json.dumps(failures)}")
    output = json.loads(capsys.readouterr().out)
    assert len(output["runs"]) == len(runner_results) == len(worker_calls) == 1
    result = runner_results[0]
    assert result.complete
    assert result.plan.candidates == ("selected-system",)
    directory = result.artifact_directory
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    child = json.loads(
        (directory / "candidates/selected-system.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["completion"]["planned_candidates"] == 1
    assert len(summary["candidates"]) == 1
    assert child["candidate"] == "selected-system" and child["status"] == "success"
    assert summary["plan"]["profile"] == "smoke"
    assert summary["paired_comparisons"] == []
    assert summary["selection"]["made_by_runner"] is False
    expected_protocols = set(harness.protocols)
    assert len(expected_protocols) == 5
    assert set(summary["protocols"]) == expected_protocols
    assert set(summary["provenance"]["protocols"]) == expected_protocols
    assert set(child["parameters"]["protocols"]) == expected_protocols
    for key, protocol in harness.protocols.items():
        assert (
            summary["provenance"]["protocols"][key]["checksum"]
            == protocol.meta.checksum
        )
        assert (
            child["parameters"]["protocols"][key]["checksum"] == protocol.meta.checksum
        )
        assert (directory / f"{key}_protocol.json").is_file()
    assert set(summary["provenance"]["engineer_decisions"]) == {"composition"}
    assert "human_review" not in summary["provenance"]["input_artifacts"]
    child_directory = directory / "candidates/selected-system"
    parquet = pd.read_parquet(child_directory / "final_attempts.parquet")
    csv = pd.read_csv(
        child_directory / "final_attempts.csv", dtype=str, keep_default_na=False
    )
    pd.testing.assert_frame_equal(parquet.fillna("").astype(str), csv)
    assert len(csv) == sum(
        row.get("kind") == "question" for row in harness.manifest.samples
    )
    assert set(csv["request_status"]) == {"completed"}
    assert set(csv["seed"]) == {"42"}
    for serialized in csv["evidence"]:
        attempt = json.loads(serialized)
        document_count = sum(
            row.get("kind") == "document" for row in harness.manifest.samples
        )
        assert attempt["retrieval"]["ranking"] == list(range(document_count))
        assert attempt["citation_identities"]
        assert attempt["visible_text"]
    sealed = child_directory / "retained-execution/generation_execution.json"
    assert sealed.is_file()
    assert (
        json.loads(sealed.read_text(encoding="utf-8"))["server_preparation"]["status"]
        == "passed"
    )
