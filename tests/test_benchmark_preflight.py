from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import stable_hash
from edumind.common.model_placement import inspect_model_placement
from experiments.benchmarks.common import process as benchmark_process
from experiments.benchmarks.common.arguments import (
    default_decision_path,
    execution_devices,
)
from experiments.benchmarks.common.preflight import (
    eligible_candidates,
    model_lock_fingerprints,
    qualification_fingerprint,
    run_preflight,
)
from experiments.benchmarks.common.preflight_reports import (
    find_local_preflight_report,
    load_preflight_report,
)
from experiments.benchmarks.common.process import (
    WorkerMeasurementError,
    WorkerResourceLimitError,
)
from experiments.benchmarks.common.tracking import benchmark_experiment
from experiments.benchmarks.extraction.audio.protocol import load_protocol as audio
from experiments.benchmarks.extraction.document.benchmark import _preflight_items
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol as document,
)
from experiments.benchmarks.extraction.video.protocol import load_protocol as video
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as chunking,
)
from experiments.benchmarks.rag.final.protocol import load_protocol as final_rag
from experiments.benchmarks.rag.generation.protocol import (
    load_protocol as generation,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    load_protocol as retrieval,
)
from experiments.benchmarks.vectordb.protocol import load_protocol as vectordb

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("suite", "stage", "expected"),
    (
        ("extraction", "document-configuration-pdf", "EduMind / Document"),
        ("extraction", "audio-development", "EduMind / ASR"),
        ("extraction", "video-development", "EduMind / Video"),
        ("rag", "chunking-embedding", "EduMind / Chunking–Embedding"),
        ("rag", "retrieval-reranking", "EduMind / Retrieval–Reranking"),
        ("rag", "generation", "EduMind / Generation"),
        ("rag", "final", "EduMind / Final RAG"),
        ("vectordb-server-v4", "ann", "EduMind / Vector Database"),
    ),
)
def test_each_benchmark_has_one_mlflow_experiment(suite, stage, expected) -> None:
    assert benchmark_experiment(suite, stage) == expected


def test_smoke_devices_and_authoritative_device_rules() -> None:
    assert execution_devices(
        "smoke", None, smoke_devices=("cpu", "cuda"), authoritative_device="cuda"
    ) == ("cpu", "cuda")
    assert execution_devices(
        "smoke", "cuda", smoke_devices=("cpu", "cuda"), authoritative_device="cuda"
    ) == ("cuda",)
    assert execution_devices(
        "preflight", None, smoke_devices=("cpu", "cuda"), authoritative_device="cuda"
    ) == ("cuda",)
    assert execution_devices(
        "development",
        None,
        smoke_devices=("cpu", "cuda"),
        authoritative_device="cuda",
    ) == ("cuda",)
    with pytest.raises(ValueError, match="requires the protocol device cuda"):
        execution_devices(
            "validation",
            "cpu",
            smoke_devices=("cpu", "cuda"),
            authoritative_device="cuda",
        )
    with pytest.raises(ValueError, match="valid only with --profile smoke"):
        execution_devices(
            "locked",
            "both",
            smoke_devices=("cpu", "cuda"),
            authoritative_device="cuda",
        )


def test_protocols_declare_the_expected_smoke_and_locked_profiles() -> None:
    for protocol in (document(), audio(), video(), chunking(), retrieval()):
        assert protocol.profile("smoke").devices == ("cpu", "cuda")
        assert protocol.profile("development").device == "cuda"
        assert protocol.profile("validation").device == "cuda"
    assert chunking().profile("locked").device == "cuda"
    assert retrieval().profile("locked").device == "cuda"
    assert chunking().profile("smoke").dtype_for("cpu") == "float32"
    assert chunking().profile("smoke").dtype_for("cuda") == "float16"
    assert retrieval().profile("smoke").dtype_for("cuda") == "float16"
    assert generation().profile("smoke").devices == ("cpu", "cuda")
    assert generation().profile("smoke").dtype_for("cpu") == "auto"
    assert generation().profile("smoke").dtype_for("cuda") == "float16"
    assert final_rag().profile("smoke").devices == ("cpu", "cuda")
    assert vectordb().profile("smoke").devices == ("cpu",)


def test_default_transition_decisions_are_project_owned() -> None:
    assert default_decision_path("audio", "development") is None
    assert default_decision_path("audio", "validation") == (
        ROOT / "data/benchmarks/decisions/audio-validation.json"
    )


def test_development_filters_exclusions_but_later_profiles_reject_them() -> None:
    report = {"qualified_candidates": ["a", "c"]}
    assert eligible_candidates(
        ("a", "b", "c"), report, profile="development", label="Test"
    ) == ("a", "c")
    with pytest.raises(ValueError, match="did not pass"):
        eligible_candidates(("a", "b"), report, profile="validation", label="Test")
    assert default_decision_path("retrieval-reranking", "locked") == (
        ROOT / "data/benchmarks/decisions/retrieval-reranking-locked.json"
    )


def test_model_fingerprint_uses_revisions_and_every_checksum() -> None:
    first = {
        "model": {
            "revision": "abc",
            "selection_revision": "selected",
            "model_cache_manifest_sha256": "one",
            "component_cache_sha256": "two",
            "model_path": "machine-specific-a",
        }
    }
    moved = {"model": {**first["model"], "model_path": "machine-specific-b"}}
    changed = {"model": {**first["model"], "component_cache_sha256": "different"}}
    assert model_lock_fingerprints(first) == model_lock_fingerprints(moved)
    assert model_lock_fingerprints(first) != model_lock_fingerprints(changed)


def _qualified() -> dict[str, object]:
    return {
        "placement": {"status": "qualified"},
        "peak_vram_mb": 100.0,
        "vram_measurement_method": "nvml-process-tree",
    }


def test_preflight_continues_after_definitive_hardware_exclusions(tmp_path) -> None:
    visited = []

    def probe(candidate: str):
        visited.append(candidate)
        if candidate == "limit":
            raise WorkerResourceLimitError(3600.0, 3584.0, [{"vram_mb": 3600.0}])
        if candidate == "oom":
            raise RuntimeError("CUDA out of memory")
        if candidate == "offload":
            return {
                **_qualified(),
                "placement": {"status": "offload_detected"},
            }
        return _qualified()

    candidates = ("limit", "qualified", "oom", "offload")
    result = run_preflight(
        benchmark="audio",
        candidates=candidates,
        fingerprint="exact-fingerprint",
        context={"protocols": {}},
        probe=probe,
        no_mlflow=True,
        artifact_root=tmp_path,
    )
    assert visited == list(candidates)
    assert result.ready_for_development
    assert result.qualified_candidates == ("qualified",)
    assert result.excluded_candidates == ("limit", "oom", "offload")
    payload = json.loads(
        (result.artifact_directory / "preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    reasons = {row["candidate"]: row["reason_code"] for row in payload["candidates"]}
    assert reasons == {
        "limit": "vram_limit_exceeded",
        "qualified": "qualified",
        "oom": "gpu_oom",
        "offload": "offload_detected",
    }
    assert payload["candidates"][0]["resource_samples"] == [{"vram_mb": 3600.0}]
    assert len(list((result.artifact_directory / "candidates").glob("*.json"))) == 4


def test_preflight_requires_every_declared_component_group(tmp_path) -> None:
    def probe(candidate: str):
        if candidate == "asr":
            raise RuntimeError("CUDA out of memory")
        return _qualified()

    result = run_preflight(
        benchmark="video",
        candidates=("asr", "visual"),
        fingerprint="grouped",
        context={"protocols": {}},
        probe=probe,
        required_groups={"frozen-asr": ("asr",), "visual": ("visual",)},
        no_mlflow=True,
        artifact_root=tmp_path,
    )
    assert not result.ready_for_development
    assert result.group_readiness == {"frozen-asr": False, "visual": True}


def test_worker_supervisor_stops_only_the_process_over_the_vram_limit(
    tmp_path, monkeypatch
) -> None:
    events = []

    class Process:
        pid = 123
        returncode = -9

        def poll(self):
            return None

        def communicate(self):
            return "", ""

    class Monitor:
        interval_seconds = 0.05
        peak_vram_mb = 3600.0
        vram_measurement_method = "nvml-process-tree"

        def __init__(self, **_options):
            pass

        def __enter__(self):
            events.append("monitor-entered")
            return self

        def __exit__(self, *_args):
            return None

        def sample_now(self):
            pass

        def samples(self):
            return [{"vram_mb": self.peak_vram_mb}]

    terminated = []
    def popen(*_args, **_kwargs):
        events.append("worker-started")
        return Process()

    monkeypatch.setattr(benchmark_process.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark_process, "ResourceMonitor", Monitor)
    monkeypatch.setattr(benchmark_process, "_terminate_process_tree", terminated.append)
    with pytest.raises(WorkerResourceLimitError, match="exceeded"):
        benchmark_process.run_json_worker(
            ROOT / "experiments/benchmarks/extraction/audio/worker.py",
            {},
            device="cuda",
            prefix="supervisor-test-",
            error_label="test",
            temporary_root=tmp_path,
            vram_limit_mb=3584.0,
        )
    assert terminated == [123]
    assert events == ["monitor-entered", "worker-started"]


def test_process_tree_termination_tolerates_an_already_exited_worker(
    monkeypatch,
) -> None:
    import psutil

    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.NoSuchProcess(123)),
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    benchmark_process._terminate_process_tree(123)


@pytest.mark.parametrize(
    "failure",
    (
        WorkerMeasurementError("NVML unavailable"),
        RuntimeError("worker could not start"),
    ),
)
def test_unresolved_preflight_errors_block_development(tmp_path, failure) -> None:
    def probe(candidate: str):
        del candidate
        raise failure

    result = run_preflight(
        benchmark="audio",
        candidates=("candidate",),
        fingerprint=stable_hash("blocked"),
        context={"protocols": {}},
        probe=probe,
        no_mlflow=True,
        artifact_root=tmp_path,
    )
    assert not result.ready_for_development
    assert result.blocked_candidates == ("candidate",)


def test_preflight_report_requires_exact_fingerprint_and_complete_roster(
    tmp_path,
) -> None:
    result = run_preflight(
        benchmark="audio",
        candidates=("a", "b"),
        fingerprint="matching",
        context={"protocols": {}},
        probe=lambda _candidate: _qualified(),
        no_mlflow=True,
        artifact_root=tmp_path,
    )
    report = result.artifact_directory / "preflight_report.json"
    assert load_preflight_report(
        report, fingerprint="matching", declared_candidates=("a", "b")
    )["ready_for_development"]
    assert (
        find_local_preflight_report(tmp_path, "audio", "matching", ("a", "b")) == report
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_preflight_report(
            report, fingerprint="different", declared_candidates=("a", "b")
        )
    with pytest.raises(ValueError, match="complete declared candidate roster"):
        load_preflight_report(
            report, fingerprint="matching", declared_candidates=("a",)
        )


class _Tensor:
    def __init__(self, device: str) -> None:
        self.device = device


class _Model:
    def __init__(self, devices, *, device_map=None, hook=None) -> None:
        self._devices = devices
        self.hf_device_map = device_map
        self._hf_hook = hook

    def parameters(self):
        return [_Tensor(device) for device in self._devices]

    def buffers(self):
        return []

    def named_modules(self):
        return [("", self)]


def test_model_placement_detects_cuda_offload_and_unverifiable_models() -> None:
    assert inspect_model_placement(_Model(["cuda:0"]))["status"] == "qualified"
    assert (
        inspect_model_placement(_Model(["cpu"]), expected_device="cpu")["status"]
        == "qualified"
    )
    assert (
        inspect_model_placement(_Model(["cuda:0", "cpu"]))["status"]
        == "offload_detected"
    )
    assert (
        inspect_model_placement(_Model([], device_map={"layer": "disk"}))["status"]
        == "offload_detected"
    )
    hook = SimpleNamespace(offload=True, weights_map=None)
    assert (
        inspect_model_placement(_Model(["cuda:0"], hook=hook))["status"]
        == "offload_detected"
    )
    assert inspect_model_placement(_Model([]))["status"] == "placement_unverifiable"

    nested = SimpleNamespace(runtime={"model": _Model(["cuda:0"])})
    report = inspect_model_placement(nested)
    assert report["status"] == "qualified"
    assert report["inspected_object_count"] > 1


def test_qualification_fingerprint_changes_with_source_and_stress_input() -> None:
    protocol = audio().meta
    common = {
        "benchmark": "audio",
        "candidates": ("candidate",),
        "protocols": {"audio": protocol},
        "revisions": {"candidate": "revision"},
        "hardware": {"gpu": "gpu"},
        "execution": {"device": "cuda"},
        "dependency_locks": {"requirements": "lock"},
    }
    first = qualification_fingerprint(
        **common,
        input_envelope={"stress_manifest_checksum": "a"},
        source_provenance={"source_tree_sha256": "one"},
    )
    changed_source = qualification_fingerprint(
        **common,
        input_envelope={"stress_manifest_checksum": "a"},
        source_provenance={"source_tree_sha256": "two"},
    )
    changed_input = qualification_fingerprint(
        **common,
        input_envelope={"stress_manifest_checksum": "b"},
        source_provenance={"source_tree_sha256": "one"},
    )
    assert len({first, changed_source, changed_input}) == 3


def test_document_preflight_uses_only_candidate_applicable_sources() -> None:
    stress = tuple({"id": kind, "kind": kind} for kind in ("pdf", "image", "docx"))
    assert [
        item["kind"] for item in _preflight_items("docling-standard-native", stress)
    ] == ["docx"]
    assert [
        item["kind"]
        for item in _preflight_items(
            "docling-standard|ocr=tesseract|mode=pdf_aware_layout_regions|"
            "table=fast|formula=off",
            stress,
        )
    ] == ["pdf"]
    assert [item["kind"] for item in _preflight_items("paddleocr-vl-1.6", stress)] == [
        "pdf",
        "image",
    ]
