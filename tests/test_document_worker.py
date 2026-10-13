"""Document adapter/worker regressions, independent of orchestration and scoring."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.extraction.contracts import (
    ExtractionProfile,
    ExtractionRequest,
    SourceKind,
)
from edumind.extraction.errors import ExtractionBackendError
from edumind.extraction.structured import (
    build_docling_document,
    build_structured_document,
)
from experiments.benchmarks.extraction.document import worker
from experiments.benchmarks.extraction.document.adapters import (
    ExperimentalDocumentExtractor,
    _paddle_blocks,
    _paddle_box,
    _parse_table_html,
)
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol,
    protocol_from_mapping,
)


def canonical_payload():
    profile = ExtractionProfile("test", "docling-standard", "revision")
    request = ExtractionRequest(Path("fixture.pdf"), "checksum", profile=profile)
    return build_structured_document(
        request,
        SourceKind.PDF,
        profile,
        [
            {"text": "alpha", "element_id": "a", "page_number": 1},
            {"text": "beta", "element_id": "b", "parent_id": "a", "page_number": 1},
        ],
    ).to_dict()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("text"),
        lambda p: p.update(segments=["bad"]),
        lambda p: p.update(warnings=["bad"]),
        lambda p: p["segments"][0].pop("text"),
        lambda p: p["segments"][0].update(start=True),
        lambda p: p["segments"][0].update(page_number=0),
        lambda p: p["segments"][0].update(bounding_box=[0.9, 0, 0.1, 1]),
        lambda p: p["segments"][0].update(bounding_box=[0, 0, float("nan"), 1]),
        lambda p: p["segments"][1].update(element_id="a"),
        lambda p: p["segments"][1].update(parent_id="missing"),
        lambda p: p["segments"][0].update(parent_id="b"),
        lambda p: p.update(extraction_seconds=float("inf")),
        lambda p: p.update(cache_hit=True),
        lambda p: p["profile"].update(device=123),
        lambda p: p["metadata"].update(bad=object()),
        lambda p: p["segments"][0].update(kind="formula", structured_content={}),
        lambda p: p["segments"][0].update(
            kind="table", structured_content={"rows": [[None]]}
        ),
        lambda p: p["segments"][0].update(end=2),
    ],
)
def test_strict_payload_validator_rejects_malformed_json(mutation):
    payload = canonical_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        worker.validate_document_payload(payload)


def test_validator_accepts_valid_empty_and_json_roundtrip():
    payload = canonical_payload()
    assert (
        worker.validate_document_payload(json.loads(json.dumps(payload))).text
        == "alpha\n\nbeta"
    )
    payload.update(text="", segments=(), warnings=())
    assert worker.validate_document_payload(payload).text == ""


def test_validator_import_does_not_load_models_or_scoring():
    code = """
import sys
from experiments.benchmarks.extraction.document.worker import validate_document_payload
assert not {'torch', 'paddle', 'docling', 'experiments.benchmarks.extraction.document.metrics'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_fingerprint_retains_structure_but_ignores_incidental_ids_and_timings():
    from experiments.benchmarks.extraction.document.metrics import _document_fingerprint

    original = canonical_payload()
    changed = deepcopy(original)
    changed["segments"][0]["element_id"] = "different-a"
    changed["segments"][1].update(element_id="different-b", parent_id="different-a")
    changed["extraction_seconds"] = 999
    a = _document_fingerprint(worker.validate_document_payload(original))
    assert _document_fingerprint(worker.validate_document_payload(changed)) == a
    changed["segments"][1]["parent_id"] = None
    assert _document_fingerprint(worker.validate_document_payload(changed)) != a
    changed = deepcopy(original)
    changed["segments"][0]["bounding_box"] = [0.01, 0.01, 0.2, 0.2]
    assert _document_fingerprint(worker.validate_document_payload(changed)) != a


@pytest.fixture
def lifecycle(monkeypatch):
    events, monitors = [], []
    root = deepcopy(load_protocol().meta.resolved)
    root["profiles"]["smoke"]["repetitions"] = 3
    protocol = protocol_from_mapping(root)
    payload = {
        "protocol": protocol.meta.worker_payload(),
        "profile": "smoke",
        "device": "cpu",
        "candidate": "docling-standard",
        "model_lock": {"docling-standard": {"revision": "test"}},
        "items": [
            {
                "id": "one",
                "kind": "pdf",
                "source_path": "unused.pdf",
                "asset_sha256": "checksum",
                "physical_page_count": 2,
            }
        ],
    }

    class Monitor:
        vram_measurement_method = "not-applicable"
        metrics_error = None

        def __init__(self, **options):
            self.options = options
            self.gpu_identity = {"uuid": "test-gpu"}
            monitors.append(self)

        def __enter__(self):
            events.append("monitor_start")
            return self

        def __exit__(self, *args):
            events.append("monitor_stop")
            if args[1]:
                args[1].resource_evidence = {"marker": "failure-resources"}

        def metrics(self):
            if self.metrics_error:
                raise self.metrics_error
            return {
                "peak_process_tree_ram_mb": 12,
                "peak_vram_mb": None,
                "peak_temporary_disk_mb": 2,
            }

        def samples(self):
            return [{"elapsed_seconds": 1, "process_tree_ram_mb": 12}]

    class Runtime:
        engine = "docling-standard"
        calls = 0

        def __init__(self):
            self.behaviors = {}
            self._runtimes = {}

        def extract(self, request, kind):
            self.calls += 1
            events.append(f"extract-{self.calls}")
            behavior = self.behaviors.get(self.calls)
            if isinstance(behavior, Exception):
                raise behavior
            document = build_structured_document(
                request,
                kind,
                request.profile,
                [
                    {
                        "text": "alpha",
                        "element_id": f"incidental-{self.calls}",
                        "page_number": None if kind is SourceKind.DOCX else 1,
                    },
                ],
            )
            return behavior(document) if callable(behavior) else document

    runtime = Runtime()
    monkeypatch.setattr(worker, "ResourceMonitor", Monitor)
    monkeypatch.setattr(worker, "seed_deterministically", lambda seed: None)
    monkeypatch.setattr(worker, "initialize", lambda *args: events.append("load"))
    monkeypatch.setattr(worker, "synchronize", lambda *args: events.append("sync"))
    monkeypatch.setattr(
        worker,
        "build_experiment_registry",
        lambda: SimpleNamespace(create=lambda *args: runtime),
    )
    monkeypatch.setattr(
        worker,
        "inspect_model_placement",
        lambda *args, **kwargs: {"status": "qualified"},
    )
    return SimpleNamespace(
        payload=payload,
        runtime=runtime,
        events=events,
        monitors=monitors,
        Monitor=Monitor,
    )


def test_worker_times_validation_and_fingerprinting_and_preserves_physical_pages(
    lifecycle, monkeypatch
):
    from experiments.benchmarks.extraction.document import metrics

    validate, fingerprint = (
        worker.validate_document_payload,
        metrics._document_fingerprint,
    )

    def checked(data):
        lifecycle.events.append("validate")
        return validate(data)

    def hashed(document):
        lifecycle.events.append("fingerprint")
        return fingerprint(document)

    monkeypatch.setattr(worker, "validate_document_payload", checked)
    monkeypatch.setattr(metrics, "_document_fingerprint", hashed)
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"] is None
    assert lifecycle.events.count("validate") == 4  # one warmup and three requests
    assert lifecycle.events.count("fingerprint") == 3
    rows = result["attempts"]
    assert all(r["executed"] and r["success"] and r["page_count"] == 2 for r in rows)
    assert all(r["latency_seconds"] >= 0 for r in rows)
    assert len({r["output_fingerprint"] for r in rows}) == 1
    assert lifecycle.events[-1] == "monitor_stop"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("stage", ["load", "warmup"])
def test_setup_failures_preserve_explicit_unexecuted_rows(
    lifecycle, monkeypatch, stage
):
    if stage == "load":

        def failed_load(*args):
            raise RuntimeError("load failed")

        monkeypatch.setattr(worker, "initialize", failed_load)
    else:
        lifecycle.runtime.behaviors[1] = RuntimeError("warmup failed")
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"]["phase"] == stage
    assert result["execution_error"]["phase_elapsed_seconds"] >= 0
    assert result["execution_error"]["worker_elapsed_seconds"] >= 0
    assert result["measured_batch_seconds"] is None
    assert result["operational"]["first_item_latency_seconds"] is None
    if stage == "load":
        assert result["operational"]["cold_model_load_seconds"] is None
    assert result["resource_evidence"]["marker"] == "failure-resources"
    assert all(
        not r["executed"]
        and r["success"] is None
        and r["error"] is None
        and r["document"] is None
        and r["latency_seconds"] is None
        and r["not_executed_reason"] == f"{stage}_failed"
        for r in result["attempts"]
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure_kind", ["schema", "serialization", "sync", "request"])
def test_recoverable_attempt_failures_are_json_safe_and_remaining_attempts_run(
    lifecycle, monkeypatch, failure_kind
):
    if failure_kind == "schema":

        def malformed(document):
            data = document.to_dict()
            data["warnings"] = ["bad"]
            return SimpleNamespace(to_dict=lambda: data)

        lifecycle.runtime.behaviors[2] = malformed
    elif failure_kind == "serialization":

        def fail_dict():
            raise ValueError("bad metadata")

        lifecycle.runtime.behaviors[2] = lambda document: SimpleNamespace(
            to_dict=fail_dict
        )
    elif failure_kind == "sync":

        def sync(*args):
            if lifecycle.runtime.calls == 2:
                raise RuntimeError("synchronization failed")

        monkeypatch.setattr(worker, "synchronize", sync)
    else:
        lifecycle.runtime.behaviors[2] = RuntimeError("request failed")
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"] is None
    assert [r["success"] for r in result["attempts"]] == [False, True, True]
    assert all(r["executed"] for r in result["attempts"])
    assert result["attempts"][0]["document"] is None
    assert result["attempts"][0]["output_fingerprint"] is None
    assert result["attempts"][0]["error"]
    assert lifecycle.runtime.calls == 4
    json.dumps(result, allow_nan=False)


def test_after_placement_failure_preserves_attempts_outside_resource_scope(
    lifecycle, monkeypatch
):
    lifecycle.payload["device"] = "cuda"
    calls = []

    def placement(*args, **kwargs):
        calls.append(len(calls))
        lifecycle.events.append("placement")
        return {"status": "qualified" if len(calls) == 1 else "offload_detected"}

    monkeypatch.setattr(worker, "inspect_model_placement", placement)
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"]["phase"] == "placement_after"
    assert all(r["success"] for r in result["attempts"])
    assert result["measured_batch_seconds"] >= 0
    assert result["placement_after"]["status"] == "offload_detected"
    assert lifecycle.events[-2:] == ["monitor_stop", "placement"]
    assert result["operational"]["peak_process_tree_ram_mb"] == 12


def test_cleanup_error_cannot_mask_original_load_failure(lifecycle, monkeypatch):
    class FailedCleanup:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return "."

        def __exit__(self, *args):
            raise PermissionError("cleanup failed")

    def failed_load(*args):
        raise RuntimeError("original load failed")

    monkeypatch.setattr(worker.tempfile, "TemporaryDirectory", FailedCleanup)
    monkeypatch.setattr(worker, "initialize", failed_load)
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"]["phase"] == "load"
    assert "original load failed" in result["execution_error"]["error"]
    assert result["execution_error"]["additional_errors"][0]["phase"] == "cleanup"
    assert result["resource_evidence"]["marker"] == "failure-resources"


def test_preflight_placement_reason_is_not_masked(lifecycle, monkeypatch):
    lifecycle.payload.update(mode="preflight", device="cuda")
    monkeypatch.setattr(
        worker, "inspect_model_placement", lambda *a, **k: {"status": "wrong_device"}
    )
    result = worker.execute(lifecycle.payload)
    assert result["placement"]["status"] == "wrong_device"
    assert result["execution_error"]["phase"] == "placement_before"
    assert not any(r["executed"] for r in result["attempts"])


def test_preflight_retains_wrapped_oom_error_and_attempt_evidence(lifecycle):
    lifecycle.payload.update(mode="preflight", device="cuda")
    lifecycle.runtime.behaviors[1] = ExtractionBackendError(
        "backend failed", detail="CUDA out of memory"
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory") as failure:
        worker.execute(lifecycle.payload)
    evidence = failure.value.resource_evidence
    assert evidence["attempts"][0]["executed"]
    assert evidence["attempts"][0]["success"] is False
    assert "CUDA out of memory" in evidence["attempts"][0]["error"]
    assert evidence["resource_samples"]


def test_missing_telemetry_preserves_partial_peaks_without_qualifying_measurement(
    lifecycle,
):
    lifecycle.payload["device"] = "cuda"
    lifecycle.Monitor.vram_measurement_method = "unavailable"
    error = RuntimeError("CUDA telemetry missing")
    error.resource_evidence = {
        "peak_process_tree_ram_mb": 456,
        "peak_vram_mb": 123,
        "resource_samples": [{"vram_mb": 123}],
    }
    lifecycle.Monitor.metrics_error = error
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"]["phase"] == "resources"
    assert all(r["success"] for r in result["attempts"])
    assert result["operational"]["peak_process_tree_ram_mb"] == 456
    assert result["operational"]["peak_vram_mb"] == 123
    assert result["operational"]["peak_temporary_disk_mb"] is None
    assert result["parameters"]["vram_measurement_method"] == "unavailable"
    assert result["resource_evidence"]["peak_vram_mb"] == 123


@pytest.mark.parametrize("peak", [None, -1, True, "123", float("nan"), float("inf")])
def test_missing_telemetry_never_promotes_invalid_peak_evidence(lifecycle, peak):
    lifecycle.payload["device"] = "cuda"
    lifecycle.Monitor.vram_measurement_method = "unavailable"
    error = RuntimeError("CUDA telemetry missing")
    error.resource_evidence = {
        "peak_process_tree_ram_mb": 0,
        "peak_vram_mb": peak,
        "peak_temporary_disk_mb": 2.5,
    }
    lifecycle.Monitor.metrics_error = error
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"]["phase"] == "resources"
    assert result["operational"]["peak_process_tree_ram_mb"] == 0
    assert result["operational"]["peak_vram_mb"] is None
    assert result["operational"]["peak_temporary_disk_mb"] == 2.5
    assert result["parameters"]["vram_measurement_method"] == "unavailable"


@pytest.mark.parametrize("failure_kind", ["measurement", "oom"])
def test_preflight_failure_class_survives_json_boundary(
    lifecycle,
    monkeypatch,
    tmp_path,
    failure_kind,
):
    from experiments.benchmarks.common import process
    from experiments.benchmarks.common.preflight import _probe_candidate

    lifecycle.payload.update(mode="preflight", device="cuda")
    if failure_kind == "measurement":
        lifecycle.Monitor.vram_measurement_method = "unavailable"
        error = RuntimeError("CUDA telemetry missing")
        error.resource_evidence = {
            "peak_process_tree_ram_mb": 456,
            "peak_vram_mb": 123,
        }
        lifecycle.Monitor.metrics_error = error
    else:
        lifecycle.runtime.behaviors[1] = ExtractionBackendError(
            "backend failed",
            detail="CUDA out of memory",
        )
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(lifecycle.payload), encoding="utf-8")
    monkeypatch.setattr(
        process.sys, "argv", [worker.__file__, str(input_path), str(output_path)]
    )
    assert process.json_worker_main(worker.execute) == 1
    serialized = output_path.read_text(encoding="utf-8")
    output = json.loads(serialized)
    expected_type = (
        "WorkerMeasurementError"
        if failure_kind == "measurement"
        else "WorkerExecutionError"
    )
    assert output["worker_error_type"] == expected_type

    class FailedProcess:
        returncode = 1

        def __init__(self, command, **kwargs):
            Path(command[-1]).write_text(serialized, encoding="utf-8")

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(process.subprocess, "Popen", FailedProcess)

    def probe(candidate):
        return process.run_json_worker(
            Path(worker.__file__),
            lifecycle.payload,
            device="cpu",
            prefix="document-classification-",
            error_label="document preflight",
            temporary_root=tmp_path,
        )

    report = _probe_candidate("docling-standard", probe)
    if failure_kind == "measurement":
        assert report["status"] == "blocked"
        assert report["reason_code"] == report["outcome"] == "measurement_unavailable"
        assert report["operational"]["peak_process_tree_ram_mb"] == 456
        assert report["operational"]["peak_vram_mb"] == 123
        assert all(row["success"] for row in report["attempts"])
        assert report["vram_measurement_method"] == "unavailable"
    else:
        assert report["status"] == "excluded"
        assert report["reason_code"] == report["outcome"] == "gpu_oom"
        assert "CUDA out of memory" in report["attempts"][0]["error"]


def test_native_cuda_invocation_uses_cpu_without_required_vram(lifecycle, monkeypatch):
    lifecycle.payload.update(candidate="docling-standard-native", device="cuda")
    lifecycle.payload["items"][0]["kind"] = "docx"
    sync_devices = []
    monkeypatch.setattr(
        worker, "synchronize", lambda device, engine: sync_devices.append(device)
    )
    result = worker.execute(lifecycle.payload)
    assert result["execution_error"] is None
    assert set(sync_devices) == {"cpu"}
    assert lifecycle.monitors[0].options["device"] == "cpu"
    assert lifecycle.monitors[0].options["require_vram"] is False
    assert all(r["page_count"] is None for r in result["attempts"])
    assert result["operational"]["peak_vram_mb"] is None


@pytest.mark.parametrize(
    "child_peak, expected_peak",
    [
        (None, 200),
        (150, 200),
        (250, 250),
        (-1, 200),
        (True, 200),
        ("900", 200),
        (float("nan"), 200),
        (float("inf"), 200),
    ],
)
def test_nonzero_worker_exit_preserves_supervisor_and_child_resource_evidence(
    monkeypatch,
    tmp_path,
    child_peak,
    expected_peak,
):
    from experiments.benchmarks.common import process

    supervisor_samples = [{"vram_mb": 200, "source": "supervisor"}]
    child_samples = [{"vram_mb": None, "source": "worker"}]
    placement = {"status": "wrong_device"}
    validation = {"status": "invalid_reference"}
    child_result = {
        "worker_error": "CUDA telemetry missing",
        "worker_error_type": "WorkerMeasurementError",
        "resource_evidence": {
            "peak_process_tree_ram_mb": 500,
            "peak_vram_mb": child_peak,
            "peak_temporary_disk_mb": None,
            "resource_samples": child_samples,
            "vram_measurement_method": "unavailable",
            "placement": placement,
            "operational": {"peak_temporary_disk_mb": 2.5},
        },
        "failure_evidence": {"input_validation": validation, "peak_vram_mb": None},
    }

    class Monitor:
        peak_vram_mb = 200
        vram_measurement_method = "nvml-device-total"

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def samples(self):
            return supervisor_samples

        def metrics(self):
            return {"peak_process_tree_ram_mb": 400, "peak_vram_mb": 200}

    class FailedProcess:
        returncode = 1

        def __init__(self, command, **kwargs):
            Path(command[-1]).write_text(json.dumps(child_result), encoding="utf-8")

        def poll(self):
            return self.returncode

        def communicate(self):
            return "", ""

    monkeypatch.setattr(process, "ResourceMonitor", Monitor)
    monkeypatch.setattr(process.subprocess, "Popen", FailedProcess)
    with pytest.raises(process.WorkerMeasurementError) as failure:
        process.run_json_worker(
            Path(worker.__file__),
            {},
            device="cuda",
            prefix="document-evidence-",
            error_label="document preflight",
            temporary_root=tmp_path,
            require_vram_measurement=True,
        )
    evidence = failure.value.resource_evidence
    assert evidence["peak_vram_mb"] == expected_peak
    assert evidence["peak_process_tree_ram_mb"] == 500
    assert evidence["peak_temporary_disk_mb"] == 2.5
    assert evidence["resource_samples"] == supervisor_samples
    assert evidence["placement"] == placement
    assert evidence["input_validation"] == validation
    assert evidence["vram_measurement_method"] == "unavailable"
    assert evidence["supervisor_resource_evidence"]["peak_vram_mb"] == 200
    assert (
        evidence["supervisor_resource_evidence"]["resource_samples"]
        == supervisor_samples
    )
    assert (
        evidence["worker_result"]["resource_evidence"]["resource_samples"]
        == child_samples
    )
    assert (
        evidence["worker_result"]["failure_evidence"]["input_validation"] == validation
    )
    assert failure.value.operational["peak_vram_mb"] == expected_peak


def test_pipeline_parameters_only_log_constructed_pipelines_and_paddle_defaults():
    unused = SimpleNamespace(
        pipeline_options=SimpleNamespace(model_dump=lambda **k: {"unused": True})
    )
    pipeline = SimpleNamespace(
        pipeline_options=SimpleNamespace(model_dump=lambda **k: {"images_scale": 4})
    )
    runtime = SimpleNamespace(
        format_to_options={"unused-format": unused},
        initialized_pipelines={(object, "hash"): pipeline},
    )
    paddle = SimpleNamespace(
        _merged_paddlex_config={"batch_size": 64, "use_queues": True}
    )
    result = worker.pipeline_parameters(
        SimpleNamespace(_runtimes={"docling": runtime, "paddle": paddle})
    )
    assert list(result.values()) == [
        {"images_scale": 4},
        {"batch_size": 64, "use_queues": True},
    ]


def test_protocol_and_adapter_reject_unsupported_granite_quantization():
    root = deepcopy(load_protocol().meta.resolved)
    root["parser"]["granite"]["load_in_8bit"] = True
    with pytest.raises(ValueError, match="load_in_8bit=True"):
        protocol_from_mapping(root)
    request = ExtractionRequest(
        Path("unused"), "checksum", options={"load_in_8bit": True}
    )
    with pytest.raises(ValueError, match="load_in_8bit=True"):
        ExperimentalDocumentExtractor(
            "docling-vlm-granite-258m", "revision"
        )._docling_converter(request)


def test_paddle_native_label_vocabulary_and_exact_latex():
    labels = {
        "doc_title": "title",
        "paragraph_title": "heading",
        "figure_title": "caption",
        "header": "page_header",
        "footer": "page_footer",
        "chart": "figure",
        "display_formula": "formula",
        "inline_formula": "formula",
    }
    content = r" \frac{x}{y} "
    blocks = _paddle_blocks(
        {
            "page_index": 0,
            "parsing_res_list": [
                {"block_label": label, "block_content": content} for label in labels
            ],
        }
    )
    assert [b["kind"] for b in blocks] == list(labels.values())
    assert all(
        b["structured_content"]["latex"] == content
        for b in blocks
        if b["kind"] == "formula"
    )


def test_paddle_null_content_and_invalid_geometry_are_recoverable():
    warnings = []
    blocks = _paddle_blocks(
        {
            "page_index": 0,
            "width": 200,
            "height": 100,
            "parsing_res_list": [
                {"block_content": None},
                {"block_content": "kept", "block_bbox": [100, 10, 10, 50]},
            ],
        },
        warnings=warnings,
    )
    assert [b["text"] for b in blocks] == ["kept"]
    assert blocks[0]["bounding_box"] is None
    assert [w.code for w in warnings] == [
        "paddle_block_conversion_failed",
        "paddle_bbox_malformed",
    ]
    assert _paddle_box([0, 0, 1, 1], (200, 100)) == [0, 0, 0.005, 0.01]
    for box in ([0, 0, float("inf"), 1], [1, 0, 0, 1], [0, 0, 201, 100]):
        with pytest.raises(ValueError):
            _paddle_box(box, (200, 100))


def test_paddle_ids_are_page_scoped_and_warning_offsets_are_global():
    warnings = []
    page = {
        "page_index": None,
        "parsing_res_list": [{"block_id": 0, "block_content": "ok"}, "bad"],
    }
    a = _paddle_blocks(page, warnings=warnings, fallback_page_number=1)
    b = _paddle_blocks(
        page, warnings=warnings, fallback_page_number=2, segment_offset=len(a)
    )
    assert a[0]["element_id"] != b[0]["element_id"]
    assert [a[0]["page_number"], b[0]["page_number"]] == [1, 2]
    assert [w.segment_index for w in warnings] == [1, 2]


def test_table_optional_cell_closures_and_breaks_preserve_text_and_headers():
    rows, cells = _parse_table_html(
        "<table><tr><th>A</tr><tr><td>alpha<br>beta</tr></table>"
    )
    assert rows == [["A"], ["alpha beta"]]
    assert [c["header"] for c in cells] == [True, False]
    assert _parse_table_html("<table><tr><td>A")[0] == [["A"]]


def test_table_span_allocation_guard_is_recoverable_and_preserves_following_blocks():
    warnings = []
    blocks = _paddle_blocks(
        {
            "parsing_res_list": [
                {
                    "block_label": "table",
                    "block_content": '<table><tr><td rowspan="1000000000">A</td></tr></table>',
                },
                {"block_content": "kept"},
            ]
        },
        warnings=warnings,
    )
    assert [b["text"] for b in blocks] == ["kept"]
    assert warnings[0].code == "paddle_block_conversion_failed"
    assert "budget" in warnings[0].message
    with pytest.raises(ValueError, match="budget"):
        _parse_table_html(
            '<table><tr><td rowspan="1000" colspan="1000">A</td></tr></table>'
        )


def test_real_native_docling_hierarchy_maps_ancestors_without_synthetic_groups():
    core = pytest.importorskip("docling_core.types.doc")
    native = core.DoclingDocument(name="hierarchy")
    heading = native.add_heading(text="Heading", level=1)
    group = native.add_list_group(parent=heading)
    first = native.add_list_item(text="First", parent=group)
    nested_group = native.add_list_group(parent=first)
    native.add_list_item(text="Nested", parent=nested_group)
    profile = ExtractionProfile("native", "docling-standard", "revision")
    request = ExtractionRequest(Path("native.docx"), "checksum", profile=profile)
    document = build_docling_document(request, SourceKind.DOCX, profile, native)
    assert [s.text for s in document.segments] == ["Heading", "First", "Nested"]
    assert [s.parent_id for s in document.segments] == [
        None,
        heading.self_ref,
        first.self_ref,
    ]
    assert [s.metadata["hierarchy_level"] for s in document.segments] == [1, 2, 3]
    assert len(document.segments) == 3
    worker.validate_document_payload(document.to_dict())


def test_pipeline_parameters_preserve_concrete_ocr_and_table_options():
    options_module = pytest.importorskip("docling.datamodel.pipeline_options")
    options = options_module.PdfPipelineOptions(
        ocr_options=options_module.EasyOcrOptions(
            lang=["en"], use_gpu=True, download_enabled=False
        )
    )
    options.table_structure_options.mode = options_module.TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = False
    extractor = SimpleNamespace(
        _runtimes={
            "runtime": SimpleNamespace(
                initialized_pipelines={
                    (object, "hash"): SimpleNamespace(pipeline_options=options)
                }
            )
        }
    )
    logged = next(iter(worker.pipeline_parameters(extractor).values()))
    assert logged["ocr_options"]["use_gpu"] is True
    assert logged["ocr_options"]["download_enabled"] is False
    assert logged["table_structure_options"] == {
        "mode": "accurate",
        "do_cell_matching": False,
    }
