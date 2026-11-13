"""Readiness gates use test-only inputs, never held-out execution or real models."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.contracts import BenchmarkPlan
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.preflight import _probe_candidate
from experiments.benchmarks.rag.final import run
from experiments.benchmarks.rag.generation import metrics as generation_metrics
from experiments.benchmarks.rag.generation import run as generation_run
from experiments.benchmarks.rag.generation.evidence import retain_schedule
from experiments.benchmarks.rag.generation.validation import questions_from_manifest

REAL_WORKER = run._worker


class StopBeforeReservation(Exception):
    """Stop without reserving or executing any real locked input."""


@pytest.fixture
def ready_inputs(monkeypatch, tmp_path):
    protocols = {key: loader() for key, loader in run.LOADERS.items()}
    manifest = load_manifest(run.manifest_path("smoke"))
    # Reviewed policy values here are fixture-only, not real support decisions.
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    protocols["generation"] = replace(
        protocols["generation"],
        minimum_ci_sources=dict.fromkeys(
            generation_run.required_ci_policy_keys(questions), 2
        ),
    )
    protocols["retrieval"] = replace(
        protocols["retrieval"],
        minimum_quality_ci_documents=2,
        minimum_latency_ci_documents=2,
    )
    composition = SimpleNamespace(
        path=tmp_path / "test-only-composition.json",
        checksum="test-only-composition",
        protocols={key: value.meta.checksum for key, value in protocols.items()},
        upstream={"database": {"path": "test-only-database-decision.json"}},
        system={
            "chunking_embedding": "token-256-32|Alibaba-NLP/gte-modernbert-base",
            "retrieval": "dense|none",
            "generator": "qwen3-0.6b-direct",
            "database": "chroma|test-only",
            "residency": "resident",
            "top_k": 3,
        },
    )
    qualification = {
        "candidates": [
            {
                "candidate": "selected-system",
                "status": "qualified",
                "tested_input_envelope": {
                    "maximum_prompt_tokens": 8191,
                    "context_tokens": 8192,
                },
            }
        ],
        "mlflow_run_id": "test-only-qualification",
        "qualification_fingerprint": "test-only-fingerprint",
    }

    def loader_for(name):
        def loader(_path):
            return protocols[name]

        return loader

    loaders = {name: loader_for(name) for name in protocols}
    monkeypatch.setattr(run, "LOADERS", loaders)
    monkeypatch.setattr(run, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run, "load_composition", lambda *a: composition)
    monkeypatch.setattr(
        run, "selected_candidates", lambda *a: (composition.system["database"],)
    )
    monkeypatch.setattr(run, "load_manifest", lambda *a: manifest)
    monkeypatch.setattr(
        run, "verify_inputs", lambda *a, **k: (tmp_path / "test-only-report.json", {})
    )
    monkeypatch.setattr(
        run, "_judge", lambda *a: SimpleNamespace(contract={"test_only": True})
    )
    monkeypatch.setattr(run, "load_selected_model_lock", lambda *a, **k: {})
    monkeypatch.setattr(run, "_verify_model_lineage", lambda *a: None)
    monkeypatch.setattr(run, "_stress", lambda value: value)
    monkeypatch.setattr(run, "model_lock_fingerprints", lambda *a: {})
    monkeypatch.setattr(
        run,
        "current_qualification_fingerprint",
        lambda **k: ("test-only-fingerprint", {}),
    )
    monkeypatch.setattr(
        run,
        "resolve_preflight_report",
        lambda **k: (tmp_path / "test-only-preflight.json", qualification),
    )
    monkeypatch.setattr(
        run, "eligible_candidates", lambda *a, **k: ("selected-system",)
    )
    reserve = Mock(side_effect=StopBeforeReservation)
    worker = Mock(side_effect=AssertionError("Readiness must finish before inference"))
    benchmark = Mock(
        side_effect=AssertionError("Readiness must finish before execution")
    )
    monkeypatch.setattr(run, "reserve_locked", reserve)
    monkeypatch.setattr(run, "_worker", worker)
    monkeypatch.setattr(run, "run_benchmark", benchmark)
    return SimpleNamespace(
        protocols=protocols,
        manifest=manifest,
        composition=composition,
        reserve=reserve,
        worker=worker,
        benchmark=benchmark,
    )


@pytest.mark.parametrize(
    "missing", ["generation", "retrieval_quality", "retrieval_latency"]
)
def test_unfrozen_policy_blocks_before_locked_reservation(ready_inputs, missing):
    case = ready_inputs
    if missing == "generation":
        case.protocols["generation"].minimum_ci_sources.pop(
            "slice.answerable.quality.citation_support"
        )
    else:
        field = (
            "minimum_quality_ci_documents"
            if missing == "retrieval_quality"
            else "minimum_latency_ci_documents"
        )
        case.protocols["retrieval"] = replace(
            case.protocols["retrieval"], **{field: None}
        )
    with pytest.raises(ValueError, match="ci_support_policy_pending"):
        run.main(["--profile", "locked", "--confirm-locked-test", "--no-mlflow"])
    case.reserve.assert_not_called()
    case.worker.assert_not_called()
    case.benchmark.assert_not_called()


def test_frozen_policies_allow_reservation_without_requiring_supplied_gold(
    ready_inputs, monkeypatch
):
    case = ready_inputs
    rows = deepcopy(case.manifest.samples)
    for row in rows:
        if row.get("kind") == "question":
            row.pop("supplied_evidence", None)
    manifest = replace(case.manifest, samples=rows)
    monkeypatch.setattr(run, "load_manifest", lambda *a: manifest)
    with pytest.raises(StopBeforeReservation):
        run.main(["--profile", "locked", "--confirm-locked-test", "--no-mlflow"])
    case.reserve.assert_called_once()
    case.worker.assert_not_called()
    case.benchmark.assert_not_called()


def test_generation_cli_preserves_public_helper_alias():
    assert (
        generation_run.required_ci_policy_keys
        is generation_metrics.required_ci_policy_keys
    )


def empty_worker_result(
    manifest, lock, plan, protocols, composition, *_args, **_kwargs
):
    generation = protocols["generation"]
    candidate = composition.system["generator"]
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    schedule = retain_schedule(candidate, manifest, lock, plan, generation, questions)
    schedule.update(
        final_composition=composition.checksum, final_protocols=composition.protocols
    )
    return {
        "candidate": candidate,
        "mode": generation.mode(candidate),
        "model_id": generation.model_id(candidate),
        "device": plan.settings["device"],
        "dtype": plan.settings["dtype"],
        "schedule": schedule,
        "schedule_checksum": stable_hash(schedule),
        "composition_checksum": composition.checksum,
        "records": [],
        "error": None,
        "phase": "finished",
        "monitoring_complete": True,
        "resource_status": "observed",
        "vram_measurement_method": "nvml-device-total",
        "operational": {"peak_ram_mib": 100, "peak_vram_mib": 1800},
        "resource_samples": [{"vram_mb": 1800, "ram_mb": 100}],
        "input_compatibility": {
            "status": "passed",
            "questions": [{"prompt_tokens": 100}],
        },
    }


def test_empty_preflight_records_block_actual_qualification(ready_inputs, monkeypatch):
    case = ready_inputs
    monkeypatch.setattr(run, "_worker", lambda *a, **k: empty_worker_result(*a, **k))
    observed = {}

    def preflight(**options):
        observed.update(_probe_candidate("selected-system", options["probe"]))
        return SimpleNamespace(ready_for_development=observed["status"] == "qualified")

    monkeypatch.setattr(run, "run_preflight", preflight)
    assert run.main(["--profile", "preflight", "--no-mlflow"]) == 2
    assert observed["status"] == "blocked"
    assert "omitted required question outcomes" in observed["error"]
    case.reserve.assert_not_called()


def test_empty_worker_records_rejected_before_sealing(
    ready_inputs, monkeypatch, tmp_path
):
    case = ready_inputs
    plan = BenchmarkPlan(
        "rag",
        "final",
        "locked",
        case.manifest.name,
        ("selected-system",),
        42,
        1,
        0,
        0,
        {"device": "cuda", "dtype": "float16"},
    )
    raw = empty_worker_result(case.manifest, {}, plan, case.protocols, case.composition)
    monkeypatch.setattr(run, "_run_worker", lambda *a, **k: raw)
    seal = Mock()
    monkeypatch.setattr(run, "seal_execution", seal)
    with pytest.raises(ValueError, match="omitted required question outcomes"):
        REAL_WORKER(
            case.manifest,
            {},
            plan,
            case.protocols,
            case.composition,
            tmp_path,
            preflight=True,
        )
    seal.assert_not_called()
