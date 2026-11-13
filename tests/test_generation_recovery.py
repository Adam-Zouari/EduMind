"""Behavioral migration regressions; every backend/judge here is test-only."""

import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_generation_benchmark import (
    Backend,
    FixtureJudge,
    Monitor,
    answer,
    attempt,
    execute,
)
from test_generation_benchmark import inputs as base_inputs

from edumind.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
)
from experiments.benchmarks.common.contracts import CandidateExecutionError
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
    benchmark_objects_from_payload,
)
from experiments.benchmarks.rag.generation import evaluate, run
from experiments.benchmarks.rag.generation.evidence import load_execution
from experiments.benchmarks.rag.generation.judge import (
    LABEL_SCHEMA,
    TASKS,
    SemanticJudge,
    configured_judge,
    validate_response_labels,
)
from experiments.benchmarks.rag.generation.metrics import (
    PRIMARY_METRICS,
    RATE_KEYS,
    aggregate,
    cell,
    score_attempt,
    score_pairs,
)
from experiments.benchmarks.rag.generation.models import LocalBackend
from experiments.benchmarks.rag.generation.protocol import load_protocol
from experiments.benchmarks.rag.generation.validation import questions_from_manifest


@pytest.fixture
def inputs():
    return base_inputs.__wrapped__()


@pytest.mark.parametrize("damage", ["one_seed", "same_seed", "question", "empty"])
def test_rescoring_retains_independent_schedule(inputs, monkeypatch, damage):
    inputs = (
        replace(
            inputs[0],
            minimum_ci_sources=dict.fromkeys(run.required_ci_policy_keys(inputs[2]), 2),
        ),
        *inputs[1:],
    )
    raw, _ = execute(inputs, monkeypatch, Backend())
    assert (
        evaluate.score_execution(raw, inputs[0], "development", judge=FixtureJudge())[
            3
        ]["result_status"]
        == "complete"
    )
    if damage == "one_seed":
        for row in raw["records"]:
            row["attempts"] = row["attempts"][:1]
    elif damage == "same_seed":
        for row in raw["records"]:
            for item in row["attempts"]:
                item["seed"] = 42
    elif damage == "question":
        raw["records"].pop()
    else:
        raw["records"] = []
    with pytest.raises((CandidateExecutionError, ValueError)):
        evaluate.score_execution(raw, inputs[0], "development", judge=FixtureJudge())


def test_one_seed_smoke_is_complete(inputs, monkeypatch):
    raw, _ = execute(inputs, monkeypatch, Backend(), profile="smoke")
    result = evaluate.score_execution(raw, inputs[0], "smoke")
    assert result[3]["result_status"] == "complete"
    assert result[3]["measured_seeds"] == [42]


def test_citation_subset_is_distinct_from_full_context(inputs):
    q = inputs[2][0]
    response = answer(citations=[])
    labels = deepcopy(FixtureJudge().response(q, response))
    claim = labels["labels"]["claims"][0]
    claim.update(
        cited_supported=False,
        cited_support_sources=[],
        cited_support_reason="No cited passages; evidence is present only in full context",
    )
    scored = score_attempt(q, attempt(response=response), labels)
    assert scored["metrics"]["quality.faithfulness"]["value"] == 1
    assert scored["metrics"]["quality.citation_support"]["value"] == 0
    assert len(PRIMARY_METRICS) == 4


def test_preinfer_prompt_drift_is_input_failure(inputs, monkeypatch):
    class DriftBackend(Backend):
        def prepare(self, question):
            value = super().prepare(question)
            if self.calls:
                value["prompt_tokens"] += 1
            return value

    raw, _ = execute(inputs, monkeypatch, DriftBackend())
    assert raw["phase"] != "finished" and raw["error"]
    assert raw["failure_stage"] == "input_validation"
    first = raw["records"][0]["attempts"][0]
    assert first["outcome"] == "input_rejected"
    with pytest.raises(CandidateExecutionError) as caught:
        evaluate.score_execution(raw, inputs[0], "development")
    assert caught.value.metrics["reliability.runtime_error_rate"] is None


@pytest.fixture
def calibrated_inputs(inputs, tmp_path):
    """Synthetic calibration solely for interface tests, never real readiness."""
    contract = {
        "schema_version": 2,
        "identity": {"model_id": "test-only", "revision": "fixture", "decoding": {}},
        "prompts": {"response": "test-only response", "pair": "test-only pair"},
        "rubric": {
            "response": "test-only response scopes",
            "pair": "test-only semantic comparison",
        },
        "label_schema": LABEL_SCHEMA,
        "retry": {"maximum_attempts": 2, "timeout_seconds": 2},
        "calibration": {"path": "calibration.json", "sha256": "pending"},
    }
    checksum = stable_hash({k: v for k, v in contract.items() if k != "calibration"})
    human = tmp_path / "human.json"
    atomic_write_json(
        human,
        {
            "schema_version": 2,
            "label_schema": LABEL_SCHEMA,
            "split": "development",
            "items": [{"test_only": True}],
        },
    )
    calibration = tmp_path / "calibration.json"
    atomic_write_json(
        calibration,
        {
            "schema_version": 2,
            "label_schema": LABEL_SCHEMA,
            "split": "development",
            "contract_checksum": checksum,
            "reviewed_by": "test-only",
            "selection_rationale": "test-only",
            "cost_review": {"test_only": True},
            "human_labels": {"path": human.name, "sha256": sha256_file(human)},
            "criteria": {
                task: {"criterion": "test-only", "observed": 1, "passed": True}
                for task in TASKS
            },
        },
    )
    contract["calibration"]["sha256"] = sha256_file(calibration)
    contract_path = tmp_path / "judge.json"
    atomic_write_json(contract_path, contract)
    import yaml

    source = yaml.safe_load(inputs[0].meta.source_path.read_text(encoding="utf-8"))
    source["judge"]["contract_path"] = str(contract_path)
    source["statistics"]["minimum_ci_sources"] = dict.fromkeys(
        run.required_ci_policy_keys(inputs[2]), 2
    )
    protocol_path = tmp_path / "protocol.yaml"
    atomic_write_text(protocol_path, yaml.safe_dump(source))
    return load_protocol(protocol_path), *inputs[1:]


def fixture_transport(question):
    def transport(request, *, timeout_seconds):
        if request["task"] == "pair":
            return FixtureJudge().pair(question, request["left"], request["right"])[
                "labels"
            ]
        assert request["cited_evidence"] == [
            b
            for b in request["supplied_evidence"]
            if b["id"] in request["response"]["citations"]
        ]
        assert not {"candidate", "model", "mode", "seed", "reasoning"} & request.keys()
        return FixtureJudge().response(question, request["response"])["labels"]

    return transport


@pytest.mark.parametrize(
    "failure",
    [
        WorkerMeasurementError("test monitoring"),
        WorkerExecutionError("test deadline", {}),
        OSError("test backend"),
    ],
)
def test_typed_inflight_failure_categories(inputs, monkeypatch, failure):
    class FaultBackend(Backend):
        def infer(self, prepared, seed):
            if seed == 43:
                raise failure
            return super().infer(prepared, seed)

    raw, _ = execute(inputs, monkeypatch, FaultBackend())
    first, failed, last = raw["records"][0]["attempts"]
    assert first["outcome"] == "completed" and first["token_event_seconds"]
    if isinstance(failure, WorkerExecutionError):
        assert failed["outcome"] == "infrastructure_error"
        assert failed["finish_reason"] is None
        assert last["outcome"] == "not_executed"
        assert raw["phase"] != "finished"
    else:
        assert failed["outcome"] == "runtime_error"
        assert last["outcome"] == "completed"
        assert raw["phase"] == "finished"
    assert failed["failure"]["type"] == type(failure).__name__
    with pytest.raises(CandidateExecutionError) as caught:
        evaluate.score_execution(raw, inputs[0], "development", judge=FixtureJudge())
    expected = (
        None if isinstance(failure, WorkerExecutionError) else pytest.approx(1 / 3)
    )
    assert caught.value.metrics["reliability.runtime_error_rate"] == expected


@pytest.mark.parametrize(
    "token_time,eos,expected",
    [(0.5, False, "context_limit"), (1.2, False, "timeout"), (0.5, True, "completed")],
)
def test_first_terminal_event_beats_delayed_backend_return(
    inputs, monkeypatch, token_time, eos, expected
):
    clock = SimpleNamespace(value=0)
    monkeypatch.setattr(evaluate.time, "perf_counter", lambda: clock.value)
    backend = object.__new__(LocalBackend)
    backend.device = "cpu"
    backend.protocol = replace(inputs[0], streamer_timeout_seconds=1)
    backend.resolved_decoder = {}
    backend.torch = SimpleNamespace(
        manual_seed=lambda _: None, inference_mode=nullcontext
    )
    backend._separate = lambda *args: ("", None, None)

    class Tensor:
        shape = (1, 10)

        def to(self, *args):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *args):
            return self

        def tolist(self):
            return [1 if eos else 2]

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": Tensor()}

        def decode(self, *args, **kwargs):
            return ""

    def generate(**kwargs):
        kwargs["streamer"].put(Tensor())
        clock.value = token_time
        kwargs["streamer"].put(Tensor())
        clock.value = 1.5

    backend.tokenizer = Tokenizer()
    backend.model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=1), generate=generate
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            LogitsProcessor=object,
            LogitsProcessorList=list,
            StoppingCriteria=object,
            StoppingCriteriaList=list,
        ),
    )
    result = backend.infer(
        {
            "prompt": "test",
            "prompt_tokens": 10,
            "context_tokens": 5,
            "max_new_tokens": 1,
        },
        42,
    )
    assert result["outcome"] == expected
    assert result["token_event_seconds"] == [token_time]


@pytest.mark.parametrize("citations", [[], ["UNKNOWN"], ["E2"]])
def test_wrong_unknown_empty_citations_never_supply_support(inputs, citations):
    q = deepcopy(inputs[2][0])
    q["blocks"].append({"id": "E2", "text": "Unrelated context.", "gold_unit_ids": []})
    response = answer(citations=citations)
    labels = FixtureJudge().response(q, response)
    scored = score_attempt(q, attempt(response=response), labels)
    assert scored["metrics"]["quality.citation_support"]["value"] == 0
    assert scored["metrics"]["quality.faithfulness"]["value"] == 1


def test_reference_only_parser_does_not_inject_gold_context(inputs):
    rows = deepcopy(inputs[1].samples)
    for row in rows:
        if row.get("kind") == "question":
            row.pop("supplied_evidence")
    manifest = replace(inputs[1], samples=rows)
    with pytest.raises(ValueError, match="supplied_evidence"):
        questions_from_manifest(manifest)
    parsed = questions_from_manifest(manifest, require_supplied_evidence=False)
    assert parsed[0]["gold_claims"] and parsed[0]["gold_unit_ids"]
    assert parsed[0]["verification_material"] and not parsed[0]["blocks"]


def test_evaluator_only_correct_fact_cannot_rescue_context_support(inputs):
    q, response = deepcopy(inputs[2][0]), answer(citations=[])
    q["blocks"] = []
    labels = FixtureJudge().response(q, response)["labels"]
    claim = labels["claims"][0]
    claim.update(
        supported=False,
        support_sources=[],
        support_reason="Evaluator-only verification",
    )
    checked = validate_response_labels(labels, q, response)
    scored = score_attempt(
        q, attempt(response=response), {"status": "evaluated", "labels": checked}
    )
    assert scored["metrics"]["quality.faithfulness"]["value"] == 0
    assert scored["metrics"]["quality.citation_support"]["value"] == 0
    assert scored["metrics"]["quality.factual_correctness_precision"]["value"] == 1


def test_contradictory_subset_labels_retry_and_journal(calibrated_inputs, tmp_path):
    q, response = calibrated_inputs[2][0], answer()
    bad = FixtureJudge().response(q, response)["labels"]
    bad["claims"][0]["supported"] = False
    replies = iter([bad, FixtureJudge().response(q, response)["labels"]])
    judge = SemanticJudge(
        calibrated_inputs[0].judge_contract_path,
        lambda *a, **k: next(replies),
        journal_directory=tmp_path / "journal",
        evaluation_identity={"test_prediction": "fixed"},
    )
    judged = judge.response(q, response)
    assert judged["status"] == "evaluated" and len(judged["calls"]) == 2
    assert (
        judged["calls"][0]["raw"] == bad
        and "contradicts" in judged["calls"][0]["error"]
    )
    assert judged["calls"][1]["state"] == "accepted"


def test_interrupted_judging_resumes_only_missing_evaluations(
    calibrated_inputs, monkeypatch, tmp_path
):
    protocol = calibrated_inputs[0]
    raw, _ = execute(calibrated_inputs, monkeypatch, Backend())
    directory = tmp_path / "saved"
    path = evaluate._retain_execution(raw, directory)
    original = sha256_file(path)
    journal = directory / "judge-journal"
    calls = []
    good = fixture_transport(calibrated_inputs[2][0])

    def interrupt(request, **kwargs):
        calls.append(request["task"])
        if request["task"] == "pair":
            raise KeyboardInterrupt("test-only interruption")
        return good(request, **kwargs)

    judge = SemanticJudge(
        protocol.judge_contract_path,
        interrupt,
        journal_directory=journal,
        evaluation_identity=evaluate._evaluation_identity(raw),
    )
    with pytest.raises(KeyboardInterrupt):
        evaluate.score_execution(raw, protocol, "development", judge=judge)
    assert calls == ["response", "pair"]  # identical saved responses share a request
    recovered = []

    def resume(request, **kwargs):
        recovered.append(request["task"])
        return good(request, **kwargs)

    judge = SemanticJudge(
        protocol.judge_contract_path,
        resume,
        journal_directory=journal,
        evaluation_identity=evaluate._evaluation_identity(raw),
    )
    result = evaluate.score_execution(
        load_execution(path, protocol)[0], protocol, "development", judge=judge
    )
    assert result[3]["result_status"] == "complete"
    assert "response" not in recovered and recovered  # accepted replies reused
    assert sha256_file(path) == original
    assert result[5]["metric_support"]["quality.citation_support"]["evidence_complete"]


@pytest.mark.parametrize("damage", ["predictions", "schedule", "binding", "evaluator"])
def test_saved_execution_rejects_changed_provenance(
    calibrated_inputs, monkeypatch, tmp_path, damage
):
    protocol = calibrated_inputs[0]
    raw, _ = execute(calibrated_inputs, monkeypatch, Backend())
    path = evaluate._retain_execution(raw, tmp_path / "sealed")
    target = (
        path
        if damage == "predictions"
        else path.with_name("generation_schedule.json")
        if damage == "schedule"
        else path.with_name("execution_binding.json")
        if damage == "binding"
        else protocol.judge_contract_path
    )
    value = json.loads(target.read_text(encoding="utf-8"))
    if damage == "evaluator":
        value["rubric"]["response"] = "changed"
    else:
        value["tampered"] = True
    atomic_write_json(target, value)
    with pytest.raises(ValueError):
        load_execution(path, protocol)


def test_journal_rejects_other_prediction_and_retains_exhausted_raw(
    calibrated_inputs, tmp_path
):
    protocol, q = calibrated_inputs[0], calibrated_inputs[2][0]
    directory = tmp_path / "journal"
    judge = SemanticJudge(
        protocol.judge_contract_path,
        lambda *a, **k: "malformed raw marker",
        journal_directory=directory,
        evaluation_identity={"prediction": "original"},
    )
    result = judge.response(q, answer())
    assert result["status"] == "unavailable" and len(result["calls"]) == 2
    assert all(call["raw"] == "malformed raw marker" for call in result["calls"])
    with pytest.raises(ValueError, match="identity mismatch"):
        judge.with_journal(directory, {"prediction": "different"})
    retry = judge.with_journal(directory, {"prediction": "original"})
    retry.transport = fixture_transport(q)
    accepted = retry.response(q, answer())
    assert accepted["status"] == "evaluated" and len(accepted["calls"]) == 3
    assert accepted["calls"][-1]["round"] == 2


def test_scoring_only_cli_never_resolves_models_or_reserves_locked(
    calibrated_inputs, monkeypatch, tmp_path, capsys
):
    protocol = calibrated_inputs[0]
    raw, _ = execute(calibrated_inputs, monkeypatch, Backend(), profile="locked")
    path = evaluate._retain_execution(raw, tmp_path / "original")
    original = sha256_file(path)
    for name in (
        "load_selected_model_lock",
        "load_manifest",
        "resolve_preflight_report",
        "run_in_fresh_process",
    ):
        monkeypatch.setattr(
            run,
            name,
            lambda *a, **k: pytest.fail("Recovery attempted inference/preparation"),
        )
    monkeypatch.setattr(
        run,
        "configured_judge",
        lambda *a: SemanticJudge(
            protocol.judge_contract_path, fixture_transport(calibrated_inputs[2][0])
        ),
    )
    output = tmp_path / "evaluation-version"
    assert (
        run.main(
            [
                "--protocol",
                str(protocol.meta.source_path),
                "--resume-execution",
                str(path),
                "--judge-transport",
                "explicit-test-adapter.json",
                "--evaluation-output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert report["evaluation_provenance"]["evaluation_only"]
    assert report["evaluation_provenance"]["prediction_sha256"] == original
    assert sha256_file(path) == original
    assert "evaluation_only" in capsys.readouterr().out


def test_known_request_skip_hooks_do_not_fabricate_generator_error(inputs, monkeypatch):
    raw, _ = execute(inputs, monkeypatch, Backend(), profile="smoke")
    raw["records"][0]["attempts"][0] = {
        **evaluate._empty_attempt(42, "known upstream failure"),
        "request_failed": True,
    }

    def score(q, a, j):
        row = score_attempt(q, a, j)
        if a.get("request_failed"):
            for key in RATE_KEYS:
                row["metrics"][key] = (
                    cell(status="inapplicable", reason="known upstream skip")
                    if key.startswith("reliability.")
                    else cell(0)
                )
        return row

    result = evaluate.score_execution(
        raw,
        inputs[0],
        "smoke",
        attempt_scorer=score,
        execution_predicate=lambda r, a: (
            a.get("request_failed") or a["outcome"] == "completed"
        ),
        pair_failure_predicate=lambda a: a.get("request_failed", False),
    )
    assert result[3]["execution_complete"]
    assert (
        result[5]["generation_evidence"]["records"][0]["attempts"][0]["outcome"]
        == "not_executed"
    )
    assert (
        result[2]["reliability.runtime_error_rate"] == 0
    )  # other completed question only
    rows = [
        score(
            inputs[2][0],
            {**evaluate._empty_attempt(seed), "request_failed": True},
            None,
        )
        for seed in (42, 43)
    ]
    pairs = score_pairs(
        inputs[2][0], rows, executed_failure=lambda a: a["request_failed"]
    )
    assert pairs[0]["metrics"]["repeatability.status_agreement"]["value"] == 0


def test_citation_support_has_slices_completeness_and_source_intervals(inputs):
    q = inputs[2][0]
    records = []
    for source in ("A", "B"):
        question = {**q, "id": source, "document_id": source, "source_id": source}
        rows = [
            score_attempt(
                question, attempt(seed), FixtureJudge().response(question, answer())
            )
            for seed in (42, 43, 44)
        ]
        records.append(
            {
                "question": question,
                "attempts": rows,
                "pairs": score_pairs(question, rows, FixtureJudge()),
            }
        )
    result = aggregate(
        records,
        profile="development",
        resamples=30,
        minimum_ci_sources={"quality.citation_support": 2},
    )
    metric = result["quality.citation_support"]
    assert (
        metric["value"]
        == metric["interval"]["lower"]
        == metric["interval"]["upper"]
        == 1
    )
    assert metric["interval"]["degenerate"] and metric["eligible_sources"] == 2
    assert result["slice.text.quality.citation_support"]["value"] == 1


def test_known_failed_request_does_not_hide_missing_other_pair_outcome(inputs):
    q = inputs[2][0]
    left = score_attempt(q, {**evaluate._empty_attempt(42), "request_failed": True})
    right = score_attempt(
        q, {**evaluate._empty_attempt(43), "outcome": "missing_outcome"}
    )
    pair = score_pairs(
        q, [left, right], executed_failure=lambda a: a.get("request_failed", False)
    )[0]
    assert pair["metrics"]["repeatability.status_agreement"]["status"] == "incomplete"
    assert pair["metrics"]["repeatability.status_agreement"]["value"] is None


@pytest.mark.parametrize(
    "damage", ["candidate", "question", "unexpected_question", "seed", "schedule"]
)
def test_direct_rescoring_rejects_mismatched_identity(inputs, monkeypatch, damage):
    raw, _ = execute(inputs, monkeypatch, Backend(), profile="smoke")
    if damage == "candidate":
        raw["candidate"] = "qwen3-0.6b-reasoning"
    elif damage == "question":
        raw["records"][0]["question"]["verification_material"] = []
    elif damage == "unexpected_question":
        raw["records"][0]["question"]["id"] = "unexpected"
    elif damage == "seed":
        raw["records"][0]["attempts"][0]["seed"] = True
    else:
        raw["schedule"]["seeds"] = [43]
    with pytest.raises(ValueError):
        evaluate.score_execution(raw, inputs[0], "smoke")


def test_claim_free_support_is_evaluated_inapplicable_not_missing(inputs):
    response = answer(
        "The provided evidence is insufficient to answer the question.",
        [],
        "insufficient_evidence",
    )
    row = score_attempt(
        inputs[2][0],
        attempt(response=response),
        FixtureJudge().response(inputs[2][0], response),
    )
    metric = row["metrics"]["quality.citation_support"]
    assert metric["value"] is None and metric["status"] == "inapplicable"
    assert metric["reason"] == "no_generated_claims"


def test_configured_adapter_binds_frozen_identity_without_provider_default(
    calibrated_inputs, monkeypatch, tmp_path
):
    protocol, question = calibrated_inputs[0], calibrated_inputs[2][0]
    path = tmp_path / "transport.json"
    atomic_write_json(
        path,
        {"schema_version": 1, "kind": "command", "command": ["explicit-test-adapter"]},
    )
    invoked = []

    def adapter(command, **kwargs):
        assert command == ["explicit-test-adapter"] and kwargs["shell"] is False
        envelope = json.loads(kwargs["input"])
        assert envelope["identity"]["model_id"] == "test-only"
        invoked.append(envelope)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                fixture_transport(question)(envelope["request"], timeout_seconds=2)
            ),
        )

    monkeypatch.setattr(
        "experiments.benchmarks.rag.generation.judge.subprocess.run", adapter
    )
    result = configured_judge(protocol.judge_contract_path, path).response(
        question, answer()
    )
    assert result["status"] == "evaluated" and len(invoked) == 1


def test_semantic_pair_has_its_own_rubric_not_response_evidence_scopes(
    calibrated_inputs,
):
    protocol, q = calibrated_inputs[0], calibrated_inputs[2][0]
    seen = []

    def transport(request, **kwargs):
        seen.append(request)
        return fixture_transport(q)(request, **kwargs)

    judge = SemanticJudge(protocol.judge_contract_path, transport)
    assert judge.response(q, answer())["status"] == "evaluated"
    assert judge.pair(q, answer(), answer())["status"] == "evaluated"
    assert seen[0]["rubric"] != seen[1]["rubric"]
    assert "supplied_evidence" in seen[0] and "supplied_evidence" not in seen[1]
    assert "cited_evidence" not in seen[1] and "verification_material" not in seen[1]


def test_received_raw_reply_recovers_across_validation_interruption(
    calibrated_inputs, monkeypatch, tmp_path
):
    from experiments.benchmarks.rag.generation import judge as judge_module

    protocol, question = calibrated_inputs[0], calibrated_inputs[2][0]
    original = judge_module.atomic_write_json
    interrupted = False

    def checkpoint(path, payload):
        nonlocal interrupted
        original(path, payload)
        calls = payload.get("entry", {}).get("calls", [])
        if calls and calls[-1]["state"] == "received" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("Interrupted after durable raw reply")

    monkeypatch.setattr(judge_module, "atomic_write_json", checkpoint)
    directory = tmp_path / "journal"
    judge = SemanticJudge(
        protocol.judge_contract_path,
        fixture_transport(question),
        journal_directory=directory,
        evaluation_identity={"prediction": "same"},
    )
    with pytest.raises(KeyboardInterrupt):
        judge.response(question, answer())
    resumed = SemanticJudge(
        protocol.judge_contract_path,
        lambda *a, **k: pytest.fail("Received raw reply was not reused"),
        journal_directory=directory,
        evaluation_identity={"prediction": "same"},
    )
    assert resumed.response(question, answer())["status"] == "evaluated"


@pytest.mark.parametrize("tracking", [False, True])
def test_first_run_cli_generates_then_judges_and_retains_all_candidates(
    calibrated_inputs, monkeypatch, tmp_path, request, tracking
):
    client = experiment = None
    if tracking:
        mlflow = pytest.importorskip("mlflow")
        previous = mlflow.get_tracking_uri()
        request.addfinalizer(lambda: mlflow.set_tracking_uri(previous))
        uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.create_experiment(
            "EduMind / Generation", artifact_location=(tmp_path / "mlruns").as_uri()
        )
    protocol = calibrated_inputs[0]
    manifest_file = run.default_manifest_path("generation", "smoke")
    model_lock = tmp_path / "data/benchmarks/models/selected.json"
    transport_path = tmp_path / "test-adapter.json"
    atomic_write_json(transport_path, {"test_only": True})
    report = tmp_path / "report.json"
    atomic_write_json(model_lock, {"test_only": True})
    atomic_write_json(report, {"status": "passed", "test_only": True})
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run, "verify_inputs", lambda *a, **k: (report, {}))
    monkeypatch.setattr(run, "load_selected_model_lock", lambda *a, **k: {})
    backends = []

    def worker(script, payload, **kwargs):
        candidate, manifest, lock, plan = benchmark_objects_from_payload(payload)
        backend = Backend()
        backends.append(backend)
        return evaluate.execute_generation(
            candidate,
            manifest,
            lock,
            plan,
            protocol,
            backend_factory=lambda *a: backend,
            monitor_factory=Monitor,
            evidence_directory=payload["evidence_directory"],
        )

    calls = []

    def transport(request, **kwargs):
        assert backends[-1].closed
        assert len(backends[-1].calls) == 3
        calls.append(request["task"])
        return fixture_transport(calibrated_inputs[2][0])(request, **kwargs)

    monkeypatch.setattr(evaluate, "run_json_worker", worker)
    monkeypatch.setattr(
        run,
        "configured_judge",
        lambda *a: SemanticJudge(protocol.judge_contract_path, transport),
    )
    benchmark = run.run_benchmark
    results = []

    def run_local(*args, **kwargs):
        result = benchmark(*args, **kwargs, artifact_root=tmp_path / "artifacts")
        results.append(result)
        return result

    monkeypatch.setattr(run, "run_benchmark", run_local)
    assert (
        run.main(
            [
                "--protocol",
                str(protocol.meta.source_path),
                "--profile",
                "smoke",
                "--device",
                "cpu",
                "--manifest",
                str(manifest_file),
                "--judge-transport",
                str(transport_path),
                *([] if tracking else ["--no-mlflow"]),
            ]
        )
        == 0
    )
    assert len(backends) == 7 and calls == ["response"] * 7
    assert (
        len(list(results[0].artifact_directory.rglob("generation_execution.json"))) == 7
    )
    assert len(list(results[0].artifact_directory.rglob("execution_binding.json"))) == 7
    if tracking:
        runs = client.search_runs([experiment])
        assert len(runs) == 8 and all(row.info.status == "FINISHED" for row in runs)
        children = [row for row in runs if row.data.tags.get("mlflow.parentRunId")]
        assert len(children) == 7
        assert all(
            row.data.metrics["quality.citation_support"] == 1 for row in children
        )
        assert all(row.data.metrics["quality.faithfulness"] == 1 for row in children)
        for child in children:
            names = {
                item.path.rsplit("/", 1)[-1]
                for item in client.list_artifacts(child.info.run_id)
            }
            assert (
                "generation_evidence.json" in names and "metric_support.json" in names
            )
