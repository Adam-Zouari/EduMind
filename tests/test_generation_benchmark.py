"""Offline generation contracts. Fixture labels below are TEST DATA, not a judge."""

import json
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from typing import ClassVar

import pytest

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from experiments.benchmarks.common.data_validation import manifest_path
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.preflight import _probe_candidate
from experiments.benchmarks.common.process import WorkerResourceLimitError
from experiments.benchmarks.rag.generation import evaluate
from experiments.benchmarks.rag.generation.judge import (
    LABEL_SCHEMA,
    TASKS,
    SemanticJudge,
    load_judge_contract,
    replay_judge,
)
from experiments.benchmarks.rag.generation.metrics import (
    REFUSAL,
    aggregate,
    parse_response,
    score_attempt,
    score_pairs,
)
from experiments.benchmarks.rag.generation.models import LocalBackend
from experiments.benchmarks.rag.generation.protocol import load_protocol
from experiments.benchmarks.rag.generation.run import _selected_candidates
from experiments.benchmarks.rag.generation.validation import (
    prepare_report,
    questions_from_manifest,
    verify_inputs,
)


@pytest.fixture
def inputs():
    protocol = load_protocol()
    manifest = load_manifest(manifest_path("generation", "smoke"))
    return protocol, manifest, questions_from_manifest(manifest)


def answer(text="chemical energy", citations=None, status="answered"):
    return {
        "status": status,
        "answer": text,
        "citations": ["E1"] if citations is None else citations,
    }


def attempt(seed=42, response=None, outcome="completed"):
    return {
        "seed": seed,
        "outcome": outcome,
        "finish_reason": "eos" if outcome == "completed" else outcome,
        "visible_text": json.dumps(answer() if response is None else response),
        "raw_text": "saved stream",
        "token_event_seconds": [0.1, 0.3],
        "timing": {
            "ttft_seconds": 0.1,
            "first_token_seconds": 0.1,
            "last_token_seconds": 0.3,
            "generation_seconds": 0.4,
            "latency_seconds": 0.5,
        },
        "workload": {
            "prompt_tokens": 10,
            "context_tokens": 5,
            "reasoning_tokens": None,
            "visible_answer_tokens": 2,
            "total_output_tokens": 2,
        },
    }


class FixtureJudge:
    contract: ClassVar = {"test_only": True}

    def response(self, question, response):
        claims = (
            []
            if response["answer"] == REFUSAL
            else [
                {
                    "id": "claim-1",
                    "text": "Photosynthesis produces chemical energy.",
                    "spans": [[0, len(response["answer"])]],
                    "supported": True,
                    "correct": True,
                    "support_sources": [{"id": "E1", "quote": "chemical energy"}],
                    "verification_sources": [
                        {"id": "source-1", "quote": "chemical energy"}
                    ],
                    "support_reason": "Fixture source entails claim",
                    "correctness_reason": "Fixture reference verifies claim",
                    "gold_matches": [
                        {
                            "id": "energy-conversion",
                            "justification": "Same required fact",
                        }
                    ],
                }
            ]
        )
        return {
            "status": "evaluated",
            "labels": {
                "claims": claims,
                "relevancy": {
                    "level": 2 if claims else 0,
                    "justification": "Fixture rubric",
                },
            },
        }

    def pair(self, question, left, right):
        return {
            "status": "evaluated",
            "labels": {
                "equivalent": left["answer"] == right["answer"],
                "justification": "Fixture pair",
            },
        }


def record(question, attempts, judge=None):
    rows = [
        score_attempt(
            question,
            a,
            judge.response(question, parse_response(a["visible_text"])[0])
            if judge and a["outcome"] == "completed"
            else None,
        )
        for a in attempts
    ]
    return {
        "question": question,
        "attempts": rows,
        "pairs": score_pairs(question, rows, judge),
    }


class Monitor:
    vram_measurement_method = "nvml-device-total"
    gpu_identity: ClassVar = {"baseline_used_mb": 500}

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def samples(self):
        return [{"vram_mb": 1200}, {"vram_mb": 1800}]

    def metrics(self):
        return {"peak_process_tree_ram_mb": 100, "peak_vram_mb": 1800}


class Backend:
    def __init__(self, *, failure=None, placement="qualified", prompt_tokens=10):
        self.failure, self.placement_status, self.prompt_tokens = (
            failure,
            placement,
            prompt_tokens,
        )
        self.calls, self.closed, self.loaded = [], False, False
        self.resolved_decoder, self.inherited_generation_config = (
            {"do_sample": True},
            {"top_k": 20},
        )

    def prepare(self, question):
        return {
            "prompt_tokens": self.prompt_tokens,
            "context_tokens": 5,
            "question": question,
            "max_new_tokens": 8192 - self.prompt_tokens,
        }

    def load(self):
        self.loaded = True

    def synchronize(self):
        pass

    def placement(self):
        return {"status": self.placement_status}

    def infer(self, prepared, seed):
        self.calls.append(seed)
        outcome = self.failure if self.failure and seed != 0 else "completed"
        q = prepared["question"]
        response = (
            answer()
            if q["answerable"]
            else answer(REFUSAL, [], "insufficient_evidence")
        )
        return attempt(seed, response, outcome)

    def close(self):
        self.closed = True


def execute(inputs, monkeypatch, backend, *, profile="development", preflight=False):
    protocol, manifest, _ = inputs
    ticks = count()
    monkeypatch.setattr(evaluate.time, "perf_counter", lambda: next(ticks))
    if preflight:
        manifest = replace(
            manifest,
            samples=tuple(
                {**row, "preflight_stress": True}
                if row.get("kind") == "question"
                else row
                for row in manifest.samples
            ),
        )
    execution = protocol.preflight if preflight else protocol.profile(profile)
    plan = BenchmarkPlan(
        "rag",
        "generation",
        profile,
        manifest.name,
        ("qwen3-0.6b-direct",),
        protocol.meta.seed,
        execution.repetitions,
        execution.bootstrap_resamples,
        execution.warmups,
        {
            "device": "cpu" if profile == "smoke" else "cuda",
            "dtype": "float32" if profile == "smoke" else "float16",
            "qualified_input_envelopes": {
                "qwen3-0.6b-direct": {
                    "maximum_prompt_tokens": 8191,
                    "context_tokens": protocol.context_tokens,
                }
            },
        },
    )
    raw = evaluate.execute_generation(
        "qwen3-0.6b-direct",
        manifest,
        {},
        plan,
        protocol,
        backend_factory=lambda *a: backend,
        monitor_factory=Monitor,
        preflight=preflight,
    )
    return raw, plan


@pytest.mark.parametrize("tested_limit", [7000, None])
def test_preflight_envelope_reuse_rejects_untested_prompt_before_loading(
    inputs, monkeypatch, tested_limit
):
    protocol, manifest, _ = inputs
    preflight, plan = execute(
        inputs, monkeypatch, Backend(prompt_tokens=7000), preflight=True
    )
    envelope = preflight["tested_input_envelope"]
    assert envelope["maximum_prompt_tokens"] == 7000
    envelope = {**envelope, "maximum_prompt_tokens": tested_limit}
    execution = protocol.profile("development")
    plan = replace(
        plan,
        warmups=execution.warmups,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        settings={
            **plan.settings,
            "qualified_input_envelopes": {"qwen3-0.6b-direct": envelope},
        },
    )
    backend = Backend(prompt_tokens=8000)
    raw = evaluate.execute_generation(
        "qwen3-0.6b-direct",
        manifest,
        {},
        plan,
        protocol,
        backend_factory=lambda *a: backend,
        monitor_factory=Monitor,
    )
    assert "preflight_input_envelope_" in raw["error"]
    assert not backend.loaded and backend.calls == []
    assert backend.closed


def test_preflight_envelope_reuse_accepts_tested_prompt(inputs, monkeypatch):
    protocol, manifest, _ = inputs
    raw, plan = execute(inputs, monkeypatch, Backend(prompt_tokens=7000))
    plan = replace(
        plan,
        settings={
            **plan.settings,
            "qualified_input_envelopes": {
                "qwen3-0.6b-direct": {
                    "maximum_prompt_tokens": 7000,
                    "context_tokens": protocol.context_tokens,
                }
            },
        },
    )
    backend = Backend(prompt_tokens=7000)
    raw = evaluate.execute_generation(
        "qwen3-0.6b-direct",
        manifest,
        {},
        plan,
        protocol,
        backend_factory=lambda *a: backend,
        monitor_factory=Monitor,
    )
    assert raw["error"] is None and backend.loaded
    assert backend.calls == [0, 42, 43, 44, 42, 43, 44]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{}",
        "[]",
        '{"status":"answered","answer":"x","citations":[],"extra":0}',
        '{"status":"answered","answer":"x","citations":[1]}',
        '{"status":"answered","status":"answered","answer":"x","citations":[]}',
    ],
)
def test_exact_schema(text):
    assert parse_response(text)[0] is None


def test_citation_unknown_ids_deduplicate_without_erasing_semantics(inputs):
    q = inputs[2][0]
    a = attempt(response=answer(citations=["E1", "E1", "UNKNOWN"]))
    row = score_attempt(q, a, FixtureJudge().response(q, answer()))
    assert row["schema_valid"] is True
    assert row["metrics"]["quality.citation_precision"]["value"] == 0.5
    assert row["metrics"]["quality.citation_f1"]["value"] == pytest.approx(2 / 3)
    assert row["metrics"]["quality.faithfulness"]["value"] == 1
    assert row["metrics"]["validity.response_validity_rate"]["value"] == 0


def test_claim_free_and_no_judge_are_not_the_same(inputs):
    q = inputs[2][0]
    response = answer(REFUSAL, [], "insufficient_evidence")
    a = attempt(response=response)
    known = score_attempt(q, a, FixtureJudge().response(q, response))
    unknown = score_attempt(q, a)
    assert known["metrics"]["quality.faithfulness"]["reason"] == "no_generated_claims"
    assert known["metrics"]["quality.factual_correctness_f1"]["value"] == 0
    assert unknown["metrics"]["quality.factual_correctness_f1"]["value"] is None
    assert known["metrics"]["validity.malformed_output_rate"]["value"] == 0


def test_three_seed_quality_and_scheduled_failure_pairs(inputs):
    q = inputs[2][0]
    rows = record(
        q,
        [attempt(42, outcome="runtime_error"), attempt(43), attempt(44)],
        FixtureJudge(),
    )
    summary = aggregate([rows], profile="development", resamples=0)
    assert summary["quality.faithfulness"]["value"] == 1
    assert summary["quality.faithfulness"]["partial"] is True
    assert summary["repeatability.status_agreement"]["value"] == pytest.approx(1 / 3)
    assert summary["reliability.runtime_error_rate"]["value"] == pytest.approx(1 / 3)
    assert summary["validity.malformed_output_rate"]["value"] == 0
    assert summary["validity.malformed_output_rate"]["evidence_complete"] is True


def test_known_noncompletion_is_conditional_not_missing_schema(inputs):
    q = inputs[2][0]
    row = score_attempt(q, attempt(outcome="timeout"))
    assert row["metrics"]["validity.malformed_output_rate"]["status"] == "inapplicable"
    summary = aggregate([record(q, [attempt(outcome="timeout")])], profile="smoke")
    assert summary["validity.malformed_output_rate"]["evidence_complete"] is True
    assert summary["operational.ttft_p50_seconds"]["value"] == 0.1
    assert summary["operational.latency_p50_seconds"]["value"] is None


def test_document_macro_and_unfrozen_ci_support(inputs):
    q = inputs[2][0]
    records = [
        record({**q, "id": f"q{i}"}, [attempt(response=answer(citations=[]))])
        for i in range(4)
    ]
    records.append(
        record({**q, "id": "qB", "document_id": "B", "source_id": "B"}, [attempt()])
    )
    summary = aggregate(records, profile="development", resamples=30)
    point = summary["quality.citation_f1"]
    assert point["value"] == 0.5
    assert point["interval"]["lower"] is None
    assert point["interval"]["reason"] == "ci_support_not_frozen"
    assert point["contributing_sources"] == 2


def test_pooled_decode_and_supported_grouped_bootstrap(inputs):
    q = inputs[2][0]
    a, b = attempt(), attempt()
    b["workload"]["total_output_tokens"] = 5
    b["token_event_seconds"] = [0.1, 0.2, 0.3, 0.4, 0.5]
    b["timing"]["last_token_seconds"] = 0.5
    records = [record(q, [a]), record({**q, "document_id": "B", "source_id": "B"}, [b])]
    summary = aggregate(
        records,
        profile="development",
        resamples=40,
        minimum_ci_sources={
            "quality.citation_f1": 2,
            "operational.decode_tokens_per_second": 2,
        },
    )
    assert summary["operational.decode_tokens_per_second"]["value"] == pytest.approx(
        5 / 0.6
    )
    assert summary["quality.citation_f1"]["interval"]["lower"] == 1
    assert summary["quality.citation_f1"]["interval"]["degenerate"] is True


def test_compatibility_gate_prevents_load_and_invented_failures(inputs, monkeypatch):
    backend = Backend(prompt_tokens=8192)
    raw, _ = execute(inputs, monkeypatch, backend)
    assert not backend.loaded and backend.closed
    assert raw["input_compatibility"]["status"] == "failed"
    assert all(
        a["outcome"] == "not_executed" for r in raw["records"] for a in r["attempts"]
    )


def test_warmup_seeds_and_request_event_origin_do_not_change_model_ttft(
    inputs, monkeypatch
):
    backend = Backend()
    raw, _ = execute(inputs, monkeypatch, backend)
    assert backend.calls == [0, 42, 43, 44, 42, 43, 44]
    assert backend.closed and raw["phase"] == "finished"
    first = raw["records"][0]["attempts"][0]
    assert first["timing"]["ttft_seconds"] == 0.1
    assert first["timing"]["first_token_seconds"] == 1.1
    assert first["timing"]["generation_call_offset_seconds"] == 1
    assert first["timing"]["last_token_seconds"] - first["timing"][
        "first_token_seconds"
    ] == pytest.approx(0.2)


def test_stress_timeout_cannot_qualify(inputs, monkeypatch):
    raw, plan = execute(inputs, monkeypatch, Backend(failure="timeout"), preflight=True)
    assert "stress_inference_incomplete" in raw["error"]
    monkeypatch.setattr(evaluate, "run_json_worker", lambda *a, **k: raw)
    report = _probe_candidate(
        "qwen3-0.6b-direct",
        lambda _: evaluate.run_in_fresh_process(
            "qwen3-0.6b-direct", inputs[1], {}, plan, inputs[0], preflight=True
        ),
    )
    assert report["status"] == "blocked"


def test_typed_offload_is_hardware_exclusion(inputs, monkeypatch):
    backend = Backend(placement="offload_detected")
    raw, plan = execute(inputs, monkeypatch, backend, preflight=True)
    assert raw["failure_class"] == "offload_detected"
    assert not backend.calls
    monkeypatch.setattr(evaluate, "run_json_worker", lambda *a, **k: raw)
    report = _probe_candidate(
        "qwen3-0.6b-direct",
        lambda _: evaluate.run_in_fresh_process(
            "qwen3-0.6b-direct", inputs[1], {}, plan, inputs[0], preflight=True
        ),
    )
    assert (report["status"], report["reason_code"]) == ("excluded", "offload_detected")


def test_supervisor_peak_merges_instead_of_disappearing():
    raw = evaluate._merge_supervision(
        {
            "operational": {"peak_vram_mib": 1800, "peak_ram_mib": 100},
            "_worker_supervision": {
                "peak_vram_mb": 2400,
                "resource_samples": [{"vram_mb": 2400}],
            },
        }
    )
    assert raw["operational"]["peak_vram_mib"] == 2400
    assert raw["operational"]["peak_ram_mib"] == 100
    assert raw["supervisor_resource_samples"]


def test_missing_instrumentation_cannot_complete_smoke(inputs, monkeypatch):
    raw, _ = execute(inputs, monkeypatch, Backend(), profile="smoke")
    raw["records"][0]["attempts"][0].pop("token_event_seconds")
    with pytest.raises(
        CandidateExecutionError, match="instrumentation incomplete"
    ) as caught:
        evaluate.score_execution(raw, inputs[0], "smoke")
    assert caught.value.artifacts["instrumentation_issues"]


def test_partial_quality_is_artifact_only_not_normal_scalar(inputs, monkeypatch):
    raw, _ = execute(inputs, monkeypatch, Backend())
    raw["records"][0]["attempts"][0]["outcome"] = "runtime_error"
    with pytest.raises(CandidateExecutionError) as caught:
        evaluate.score_execution(raw, inputs[0], "development", judge=FixtureJudge())
    exc = caught.value
    assert exc.metrics["quality.faithfulness"] is None
    assert exc.metrics["slice.answerable.quality.faithfulness"] is None
    assert exc.artifacts["metric_support"]["quality.faithfulness"]["value"] == 1
    assert exc.artifacts["metric_support"]["quality.faithfulness"]["partial"] is True


def test_standalone_report_verification_and_missing_judge(inputs, tmp_path):
    path = manifest_path("generation", "smoke")
    report_path, report = prepare_report(path, "smoke", inputs[0], report_root=tmp_path)
    assert report["status"] == "passed"
    assert (
        verify_inputs(path, "smoke", inputs[0], report_root=tmp_path)[0] == report_path
    )
    with pytest.raises(ValueError, match="judge_calibration_pending"):
        load_judge_contract(None)


def test_missing_claims_are_invalid_reference_not_empty_score(inputs):
    manifest = deepcopy(inputs[1])
    rows = [dict(row) for row in manifest.samples]
    rows[1].pop("required_gold_claims")
    with pytest.raises(ValueError, match="required_gold_claims"):
        questions_from_manifest(replace(manifest, samples=tuple(rows)))


def test_cold_load_reloads_tokenizer_and_retains_inherited_config(monkeypatch):
    events = []
    backend = object.__new__(LocalBackend)
    backend.candidate, backend.dtype, backend.device = (
        "qwen3-0.6b-direct",
        "float16",
        "cpu",
    )
    backend.path, backend.resolved_decoder = "local-only", {"do_sample": True}
    backend.tokenizer, backend.processor = "compatibility-preloaded", None

    def tokenizer():
        assert backend.tokenizer is None
        events.append("tokenizer-cold-load")
        backend.tokenizer = "reloaded"

    model = SimpleNamespace(
        generation_config=SimpleNamespace(to_dict=lambda: {"top_k": 20})
    )
    model.to = lambda device: model
    model.eval = lambda: model

    def weights(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        events.append("weights-cold-load")
        return model

    backend._tokenizer = tokenizer
    loader = SimpleNamespace(from_pretrained=weights)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16="float16"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=loader,
            AutoModelForImageTextToText=loader,
            GenerationConfig=object,
        ),
    )
    backend.load()
    assert events == ["tokenizer-cold-load", "weights-cold-load"]
    assert backend.inherited_generation_config == {"top_k": 20}


def test_exception_after_eos_callback_is_runtime_error(monkeypatch, inputs):
    class Tensor:
        shape = (1, 10)

        def __getattr__(self, name):
            return lambda *args: self

        def tolist(self):
            return [1]

    backend = object.__new__(LocalBackend)
    backend.device, backend.protocol, backend.resolved_decoder = (
        "cpu",
        inputs[0],
        {"do_sample": True},
    )
    backend.torch = SimpleNamespace(
        manual_seed=lambda seed: None, inference_mode=nullcontext
    )
    backend.tokenizer = lambda *args, **kwargs: {"input_ids": Tensor()}
    backend._separate = lambda *args: ("", None, None)

    class Tokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": Tensor()}

        def decode(self, *args, **kwargs):
            return ""

    def generate(**kwargs):
        kwargs["streamer"].put(Tensor())  # prompt callback
        kwargs["streamer"].put(Tensor())  # EOS callback
        raise RuntimeError("synchronization/backend failure after EOS")

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
            "prompt": "native",
            "prompt_tokens": 10,
            "context_tokens": 5,
            "max_new_tokens": 8182,
        },
        42,
    )
    assert result["outcome"] == "runtime_error"
    assert result["workload"]["total_output_tokens"] == 1


def test_cpu_vram_is_inapplicable_and_supervisor_ram_is_not_worker_ram(
    inputs, monkeypatch
):
    raw, _ = execute(inputs, monkeypatch, Backend(), profile="smoke")
    assert raw["device"] == "cpu"
    assert raw["operational"]["peak_vram_mib"] is None
    merged = evaluate._merge_supervision(
        {
            **raw,
            "_worker_supervision": {
                "peak_vram_mb": 2400,
                "peak_process_tree_ram_mb": 9999,
            },
        }
    )
    assert merged["operational"]["peak_ram_mib"] == 100
    assert merged["operational"]["peak_vram_mib"] is None
    assert merged["supervisor_resource_evidence"]["peak_process_tree_ram_mb"] == 9999


def test_terminated_worker_recovers_completed_predictions_and_inflight_outcome(
    inputs, monkeypatch, tmp_path
):
    raw, plan = execute(inputs, monkeypatch, Backend(), profile="smoke")
    protocol, manifest, _ = inputs
    plan = replace(plan, settings={"device": "cuda", "dtype": "float16"})
    raw.update(
        phase="measured",
        device="cuda",
        dtype="float16",
        monitoring_complete=False,
        resource_status="unavailable",
        operational={},
    )
    last = raw["records"][-1]
    last["attempts"][0] = evaluate._empty_attempt(42)
    raw["in_flight"] = {"question_id": last["question"]["id"], "seed": 42}

    def terminated(_script, payload, **_):
        atomic_write_json(tmp_path / "generation_progress.json", raw)
        raise WorkerResourceLimitError(3600, 3584, [{"vram_mb": 3600}])

    monkeypatch.setattr(evaluate, "run_json_worker", terminated)
    with pytest.raises(WorkerResourceLimitError) as captured:
        evaluate.run_in_fresh_process(
            "qwen3-0.6b-direct",
            manifest,
            {},
            plan,
            protocol,
            judge=FixtureJudge(),
            evidence_directory=tmp_path,
        )
    failure = captured.value
    saved = failure.artifacts["generation_evidence"]
    assert saved["records"][0]["attempts"][0]["outcome"] == "completed"
    assert saved["records"][-1]["attempts"][0]["outcome"] == "runtime_error"
    assert saved["records"][-1]["attempts"][0]["raw_text"] is None
    assert failure.parameters["measurement.resource_scope"] == "partial"
    assert failure.parameters["prediction_evidence"]["sha256"]
    assert failure.operational["peak_vram_mib"] == 3600
    assert len(failure.samples) == 2


def test_generation_citation_coverage_requires_all_unit_intervals(inputs):
    _, manifest, _ = inputs
    rows = deepcopy(manifest.samples)
    question = rows[1]
    text = rows[0]["text"]
    intervals = [
        {"document_id": rows[0]["id"], "start": left, "end": right}
        for left, right in ((0, 41), (41, len(text)))
    ]
    question["evidence"] = [
        {"id": "photosynthesis-energy", "evidence_type": "text", "intervals": intervals}
    ]
    block = question["supplied_evidence"][0]
    block.update(intervals=intervals[:1], text=text[:41])
    with pytest.raises(ValueError, match="sufficient supplied evidence"):
        questions_from_manifest(replace(manifest, samples=rows))
    block.update(intervals=intervals, text=text[:41] + "\n" + text[41:])
    validated = questions_from_manifest(replace(manifest, samples=rows))
    assert validated[0]["blocks"][0]["gold_unit_ids"] == ["photosynthesis-energy"]


def test_falcon_resolves_official_decoder_without_assuming_sampling(
    monkeypatch, tmp_path
):
    atomic_write_json(tmp_path / "generation_config.json", {"do_sample": False})
    backend = object.__new__(LocalBackend)
    backend.path, backend.dtype, backend.device = tmp_path, "float32", "cpu"
    backend.candidate = "falcon-h1-tiny-r-90m-control"
    backend.resolved_decoder = {"source": "pinned-generation-config"}
    backend._tokenizer = lambda: None
    model = SimpleNamespace(
        to=lambda _: model,
        eval=lambda: model,
        generation_config=SimpleNamespace(to_dict=lambda: {"do_sample": False}),
    )
    loader = SimpleNamespace(from_pretrained=lambda *_, **__: model)
    config = SimpleNamespace(
        from_pretrained=lambda *_, **__: SimpleNamespace(do_sample=False)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=loader,
            AutoModelForImageTextToText=loader,
            GenerationConfig=config,
        ),
    )
    backend.load()
    assert backend.resolved_decoder == {"do_sample": False}
    assert backend.decoder_source["sha256"]


def test_all_interrupted_malformed_rate_keeps_scheduled_support(inputs):
    question = inputs[2][0]
    result = aggregate(
        [record(question, [attempt(seed, outcome="timeout") for seed in (42, 43, 44)])],
        profile="development",
        resamples=0,
    )["validity.malformed_output_rate"]
    assert result["value"] is None and result["status"] == "unavailable"
    assert result["reason"] == "no_completed_outputs"
    assert result["eligible_questions"] == 1
    assert result["scheduled_attempts"] == result["failed_attempts"] == 3


def test_missing_ttft_instrumentation_suppresses_only_affected_intervals(inputs):
    question = inputs[2][0]
    records = []
    for index in range(2):
        item = deepcopy(question)
        item.update(id=str(index), document_id=str(index), source_id=str(index))
        records.append(record(item, [attempt(seed) for seed in (42, 43, 44)]))
    records[0]["attempts"][0]["timing"]["ttft_seconds"] = None
    keys = ("operational.ttft_p50_seconds", "operational.latency_p50_seconds")
    result = aggregate(
        records,
        profile="development",
        resamples=100,
        minimum_ci_sources={key: 2 for key in keys},
    )
    assert result[keys[0]]["value"] == 0.1
    assert result[keys[0]]["partial"]
    assert result[keys[0]]["interval"]["lower"] is None
    assert result[keys[0]]["interval"]["reason"] == "incomplete_evidence"
    assert result[keys[1]]["evidence_complete"]
    assert result[keys[1]]["interval"]["lower"] == 0.5


def test_calibrated_judge_contract_transport_replay_and_staleness(inputs, tmp_path):
    # Synthetic calibration exercises sealing only; it is not selection evidence.
    contract = {
        "schema_version": 1,
        "identity": {
            "model_id": "test-only",
            "revision": "fixture",
            "decoding": {"temperature": 0},
        },
        "prompts": {"response": "Fixture labeling", "pair": "Fixture comparison"},
        "rubric": "Fixture only",
        "label_schema": LABEL_SCHEMA,
        "retry": {"maximum_attempts": 1, "timeout_seconds": 2},
        "calibration": {"path": "calibration.json", "sha256": "pending"},
    }
    checksum = stable_hash(
        {key: value for key, value in contract.items() if key != "calibration"}
    )
    human_path = tmp_path / "human.json"
    atomic_write_json(
        human_path, {"split": "development", "items": [{"test_only": True}]}
    )
    calibration = {
        "split": "development",
        "contract_checksum": checksum,
        "reviewed_by": "test-only",
        "selection_rationale": "Fixture sealing check",
        "cost_review": {"test_only": True},
        "human_labels": {"path": human_path.name, "sha256": sha256_file(human_path)},
        "criteria": {
            task: {"criterion": "Fixture assertion", "observed": 1.0, "passed": True}
            for task in TASKS
        },
    }
    calibration_path = tmp_path / "calibration.json"
    atomic_write_json(calibration_path, calibration)
    contract["calibration"]["sha256"] = sha256_file(calibration_path)
    path = tmp_path / "judge.json"
    atomic_write_json(path, contract)
    question, response = inputs[2][0], answer()
    replies = {}

    def transport(request, *, timeout_seconds):
        assert timeout_seconds == 2
        assert not {"candidate", "model", "seed", "mode"} & request.keys()
        reply = FixtureJudge().response(question, response)["labels"]
        replies[stable_hash(request)] = reply
        return reply

    judged = SemanticJudge(path, transport).response(question, response)
    assert judged["status"] == "evaluated"
    records = tmp_path / "replies.json"
    atomic_write_json(records, {"contract_checksum": checksum, "responses": replies})
    assert (
        replay_judge(path, records).response(question, response)["labels"]
        == judged["labels"]
    )
    contract["rubric"] = "Changed fixture rubric"
    atomic_write_json(path, contract)
    with pytest.raises(ValueError, match="bind this exact contract"):
        load_judge_contract(path)


@pytest.mark.parametrize("current_judge", ["same", "changed"])
def test_generation_advancement_binds_judge_bytes_not_just_protocol_path(
    inputs, tmp_path, current_judge
):
    protocol = inputs[0]
    alias = "qwen3-0.6b-direct"
    summary_path = tmp_path / "summary.json"
    atomic_write_json(
        summary_path,
        {
            "run_id": "source",
            "complete": True,
            "plan": {
                "suite": "rag",
                "stage": "generation",
                "profile": "development",
                "settings": {"preflight_fingerprint": "gpu"},
            },
            "protocols": {"generation": {"checksum": protocol.meta.checksum}},
            "provenance": {"input_artifacts": {"judge_contract": {"sha256": "same"}}},
            "candidates": [{"candidate": alias, "status": "success"}],
        },
    )
    decision = tmp_path / "decision.json"
    atomic_write_json(
        decision,
        {
            "schema_version": 1,
            "source_summary": "summary.json",
            "source_run_id": "source",
            "selected_candidates": [alias],
            "selected_by": "test reviewer",
            "selected_date": "2026-10-20",
            "reason": "Fixture only",
        },
    )
    if current_judge == "changed":
        with pytest.raises(ValueError, match="judge contract"):
            _selected_candidates(
                "validation",
                decision,
                protocol,
                qualification_fingerprint="gpu",
                judge_contract_sha256=current_judge,
            )
    else:
        assert _selected_candidates(
            "validation",
            decision,
            protocol,
            qualification_fingerprint="gpu",
            judge_contract_sha256=current_judge,
        ) == (alias,)
