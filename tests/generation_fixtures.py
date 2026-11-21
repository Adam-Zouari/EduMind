"""Shared offline Generation and Final fixtures; all labels are test data."""

import json
from dataclasses import replace
from itertools import count
from typing import ClassVar

import yaml

from benchmarks.common.contracts import BenchmarkPlan
from benchmarks.common.data_validation import manifest_path
from benchmarks.common.datasets import load_manifest
from benchmarks.rag.generation import evaluate, run
from benchmarks.rag.generation.judge import LABEL_SCHEMA, TASKS
from benchmarks.rag.generation.metrics import REFUSAL
from benchmarks.rag.generation.protocol import load_protocol
from benchmarks.rag.generation.validation import questions_from_manifest
from edumind.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
)


def smoke_generation_inputs():
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
                    "cited_supported": "E1" in response["citations"],
                    "cited_support_sources": [{"id": "E1", "quote": "chemical energy"}]
                    if "E1" in response["citations"]
                    else [],
                    "cited_support_reason": "Fixture cited-subset check",
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


def make_calibrated_inputs(inputs, tmp_path):
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
