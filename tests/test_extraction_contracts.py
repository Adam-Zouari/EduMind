"""Extraction contract regressions shared across the three benchmark families."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.extraction import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    SegmentKind,
    SourceKind,
)
from experiments.benchmarks.extraction.document.metrics import (
    aggregate_evaluations,
    score_document,
    validate_reference,
)
from experiments.benchmarks.extraction.document.protocol import load_protocol
from experiments.benchmarks.extraction.scoring import attempt_rates, bootstrap_sources

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "benchmark,kind", [("document", "pdf"), ("audio", "audio"), ("video", "video")]
)
def test_validation_inventory_rejects_mislabeled_held_out_samples(
    monkeypatch, benchmark, kind
):
    from experiments.benchmarks.common import data_validation

    manifests = {
        Path(split): SimpleNamespace(
            split=split,
            samples=[
                {
                    "kind": kind,
                    "id": split,
                    "asset_sha256": split,
                    "independent_source_id": split,
                    "split": "development" if split == "validation" else split,
                }
            ],
        )
        for split in ("development", "validation", "locked-test")
    }
    monkeypatch.setattr(data_validation, "load_manifest", manifests.__getitem__)
    with pytest.raises(ValueError, match="contradicts its manifest"):
        data_validation.validate_inventory(
            benchmark, "development", Path("development"), manifests
        )


def test_worker_timeout_terminates_the_tree_and_preserves_failure(
    tmp_path, monkeypatch
):
    import subprocess

    from experiments.benchmarks.common import process

    class Child:
        pid = 123
        returncode = -9

        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return "partial output", "partial error"

    terminated = []
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_args, **_kwargs: Child())
    monkeypatch.setattr(process, "_terminate_process_tree", terminated.append)
    with pytest.raises(process.WorkerExecutionError, match="timeout") as failure:
        process.run_json_worker(
            ROOT / "experiments/benchmarks/extraction/audio/worker.py",
            {},
            device="cpu",
            prefix="timeout-",
            error_label="test",
            temporary_root=tmp_path,
            timeout_seconds=1,
        )
    assert terminated == [123]
    assert failure.value.artifacts["worker_error"]["stdout"] == "partial output"


def test_worker_peak_cannot_hide_continuous_measurement_gaps(tmp_path, monkeypatch):
    from experiments.benchmarks.common import process

    class Monitor:
        peak_vram_mb = 1.0

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def metrics(self):
            error = RuntimeError("missing device-memory observations")
            error.resource_evidence = {"resource_samples": [{"vram_mb": None}]}
            raise error

    child = SimpleNamespace(
        pid=123, returncode=0, poll=lambda: 0, communicate=lambda: ("", "")
    )
    monkeypatch.setattr(process.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(process, "ResourceMonitor", Monitor)
    with pytest.raises(process.WorkerMeasurementError, match="missing") as failure:
        process.run_json_worker(
            ROOT / "experiments/benchmarks/extraction/audio/worker.py",
            {},
            device="cuda",
            prefix="measurement-",
            error_label="test",
            temporary_root=tmp_path,
            require_vram_measurement=True,
        )
    assert failure.value.artifacts["worker_error"]["resource_samples"] == [
        {"vram_mb": None}
    ]


def test_official_scorer_timeout_removes_only_its_own_container(tmp_path, monkeypatch):
    import subprocess

    from experiments.benchmarks.extraction.document import official_metrics as official

    source = tmp_path / "data/benchmarks/evaluators/OmniDocBench"
    source.mkdir(parents=True)
    (source / ".edumind-revision").write_text(official.OMNIDOCBENCH_REVISION)
    (source / official.OMNIDOCBENCH_IMAGE_LOCK).write_text("{}")
    monkeypatch.setattr(official, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        official, "official_image_digest", lambda: "image@sha256:fixture"
    )
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(official.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timeout"):
        official.score_official_metrics([("table", "table")], [], timeout_seconds=1)
    name = commands[0][commands[0].index("--name") + 1]
    assert name.startswith("edumind-scorer-")
    assert commands[1] == ["docker", "rm", "--force", name]


def test_table_html_preserves_spanning_cell_coordinates_and_recovers_missing_tags():
    from experiments.benchmarks.extraction.document.adapters import _parse_table_html

    rows, cells = _parse_table_html(
        '<table><tr><th rowspan="2">A</th><td colspan="2">B</td></tr>'
        "<tr><td>C<td>D</table>"
    )
    assert rows == [["A", "B"], ["C", "D"]]
    assert [(cell["row"], cell["column"]) for cell in cells] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
    ]
    assert cells[0]["row_span"] == 2
    assert cells[1]["column_span"] == 2


def test_bootstrap_keeps_failures_and_reports_undefined_draws():
    rows = [
        {"source_group_id": str(i), "value": value}
        for i, value in enumerate((1.0, None))
    ]

    def aggregate(draw):
        values = [row["value"] for row in draw if row["value"] is not None]
        return {"quality": sum(values) / len(values) if values else None}

    result = bootstrap_sources(
        rows, aggregate, resamples=100, seed=42, confidence=0.95, minimum_sources=1
    )["quality"]
    assert result["estimate"] == 1
    assert result["contributing_source_count"] == 1
    assert result["requested_resamples"] == 100
    assert result["defined_resamples"] + result["undefined_resamples"] == 100
    assert result["undefined_resamples"] > 0
    assert result["resampling_performed"]


def test_granite_executes_and_logs_the_protocol_image_scale(monkeypatch, tmp_path):
    converter_module = pytest.importorskip("docling.document_converter")
    from docling.datamodel.base_models import InputFormat

    from edumind.extraction import ExtractionRequest
    from experiments.benchmarks.extraction.document.adapters import (
        ExperimentalDocumentExtractor,
    )
    from experiments.benchmarks.extraction.document.worker import pipeline_parameters

    def converter(*, format_options):
        return SimpleNamespace(format_to_options=format_options)

    monkeypatch.setattr(converter_module, "DocumentConverter", converter)
    options = {"model_path": str(tmp_path), "image_scale": 4.0, "load_in_8bit": False}
    request = ExtractionRequest(
        source_path=tmp_path / "image.png",
        checksum="fixture",
        source_kind=SourceKind.IMAGE,
        profile=ExtractionProfile(
            "fixture", "docling-vlm-granite-258m", "revision", options=options
        ),
        options=options,
    )
    extractor = ExperimentalDocumentExtractor("docling-vlm-granite-258m", "revision")
    runtime = extractor._docling_converter(request)
    executed = runtime.format_to_options[InputFormat.IMAGE].pipeline_options
    assert executed.images_scale == executed.vlm_options.scale == 4.0
    logged = pipeline_parameters(extractor)[str(InputFormat.IMAGE)]
    assert logged["images_scale"] == logged["vlm_options"]["scale"] == 4.0


def test_asr_bootstrap_preserves_speech_and_control_allocation():
    rows = [
        {"source_group_id": str(i), "sample_type": kind}
        for i, kind in enumerate(("speech", "speech", "nonspeech"))
    ]
    allocations = []

    def aggregate(draw):
        if len(draw) == 3:
            allocations.append(sum(row["sample_type"] == "nonspeech" for row in draw))
        return {"quality": 1.0}

    result = bootstrap_sources(
        rows,
        aggregate,
        resamples=50,
        seed=42,
        confidence=0.95,
        minimum_sources=1,
        stratum_field="sample_type",
    )["quality"]
    assert allocations == [1] * 51  # full estimate and every resample
    assert result["stratum_source_counts"] == {"speech": 2, "nonspeech": 1}


def test_document_data_validation_checks_reference_renderability(monkeypatch, tmp_path):
    import json

    from edumind.common.artifacts import atomic_write_json
    from experiments.benchmarks.common.datasets import manifest_content_checksum
    from experiments.benchmarks.extraction.document import official_metrics, validation

    manifest = json.loads((ROOT / "data/benchmarks/extraction/smoke.json").read_text())
    sample = next(row for row in manifest["samples"] if row["kind"] == "image")
    sample.update(
        reference_capabilities=["text", "tables"],
        has_table=True,
        reference_elements=[
            {
                "id": "table",
                "kind": "table",
                "page_number": 1,
                "html": "<table><tr><td>x</td></tr></table>",
                "text": "x",
            }
        ],
    )
    manifest["samples"] = [sample]
    manifest["checksum"] = manifest_content_checksum(manifest["samples"])
    path = tmp_path / "manifest.json"
    atomic_write_json(path, manifest)
    calls = []

    def invalid_reference(tables, formulas, *, timeout_seconds):
        calls.append((tables, formulas, timeout_seconds))
        return [(0.0, 0.0)], []

    monkeypatch.setattr(official_metrics, "score_official_metrics", invalid_reference)
    with pytest.raises(ValueError, match="pinned official evaluators"):
        validation.validate(path, "smoke", load_protocol())
    assert calls[0][0][0][0] == calls[0][0][0][1]
    assert calls[0][2] == load_protocol().evaluator_timeout_seconds


def test_nested_paddle_cpu_weights_are_not_mistaken_for_verified_cuda():
    from edumind.common.model_placement import inspect_model_placement

    root = SimpleNamespace(parameters=lambda: [SimpleNamespace(place="Place(cpu)")])
    for _ in range(12):
        root = SimpleNamespace(runtime={"nested": root})
    report = inspect_model_placement(root)
    assert report["status"] == "offload_detected"
    assert report["observed_devices"] == ["cpu"]


@pytest.mark.parametrize("accessor", ["buffers", "named_modules", "get_providers"])
def test_partial_placement_inspection_cannot_qualify_a_model(accessor):
    from edumind.common.model_placement import inspect_model_placement

    def broken():
        raise RuntimeError("unavailable model state")

    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device="cuda:0")])
    setattr(model, accessor, broken)
    report = inspect_model_placement(model)
    assert report["status"] == "placement_unverifiable"
    assert not report["verified"]
    assert report["inspection_errors"]


def test_late_telemetry_failure_preserves_partial_resource_evidence():
    from experiments.benchmarks.common.resources import ResourceMonitor

    monitor = ResourceMonitor(require_vram=True)
    monitor._ram_sampled = True
    monitor._peak_ram_bytes = 1024**2
    monitor._samples = [{"ram_mb": 1.0, "vram_mb": None}]
    with pytest.raises(RuntimeError, match="device-memory") as failure:
        monitor.metrics()
    assert failure.value.resource_evidence["peak_process_tree_ram_mb"] == 1
    assert failure.value.resource_evidence["resource_samples"] == monitor.samples()


@pytest.mark.parametrize("candidate", ["video-fixed-5", "video-fixed-05s"])
def test_video_candidate_ids_are_canonical(candidate):
    from experiments.benchmarks.extraction.video.candidates import parse_candidate
    from experiments.benchmarks.extraction.video.protocol import load_protocol

    with pytest.raises(ValueError, match="Unknown video"):
        parse_candidate(candidate, load_protocol())


def document(text="", segments=()):
    return ExtractedDocument(
        "fixture",
        "fixture",
        SourceKind.DOCX,
        "checksum",
        None,
        text,
        tuple(segments),
        ExtractionProfile("fixture", "docling-standard", "revision"),
    )


def score(item, prediction, **options):
    protocol = load_protocol()
    return score_document(
        item,
        prediction,
        element_matching_threshold=protocol.element_matching_threshold,
        duplicate_content_threshold=protocol.duplicate_content_threshold,
        **options,
    )


@pytest.mark.parametrize(
    "attempts,expected",
    [
        (("A", "A", "A"), 1),
        (("A", "A", "B"), 1 / 3),
        (("A", "B", "C"), 0),
        (("A", "A", None), 1 / 3),
        ((None, "A", "A"), 1 / 3),
        (("A", None, None), 0),
        ((None, None, None), 0),
    ],
)
def test_all_scheduled_repeatability_pairs_are_accounted_for(attempts, expected):
    result = attempt_rates(attempts)
    assert result["repeatability_success_rate"] == expected
    assert result["attempt_failure_rate"] == attempts.count(None) / len(attempts)


def test_verified_blank_is_not_a_missing_annotation():
    item = {
        "id": "blank",
        "kind": "docx",
        "reference": "",
        "reference_elements": [],
        "reference_capabilities": [
            "text",
            "reading_order",
            "element_types",
            "tables",
            "formulas",
        ],
        "has_table": False,
        "has_formula": False,
    }
    validate_reference(item, authoritative=False)
    result = score(item, document())
    assert result.metrics["text.content_f1"] == 1
    assert result.metrics["text.reading_order_ned"] == 0
    assert result.metrics["tables.detection_f1"] == 1
    assert result.metrics["tables.teds"] is None
    assert result.statuses["tables.teds"]["status"] == "inapplicable"
    assert result.metrics["reliability.unexpected_empty_output_rate"] == 0
    del item["reference"]
    with pytest.raises(ValueError, match="reference text"):
        validate_reference(item, authoritative=False)


def test_omissions_affect_reference_denominator_attributes_and_order():
    item = {
        "id": "missing",
        "kind": "docx",
        "reference_capabilities": ["reading_order", "element_types", "hierarchy"],
        "reference_elements": [
            {
                "id": "h",
                "kind": "heading",
                "text": "Heading",
                "order": 0,
                "parent_id": None,
            },
            {
                "id": "p",
                "kind": "text",
                "text": "Paragraph",
                "order": 1,
                "parent_id": "h",
            },
        ],
    }
    validate_reference(item, authoritative=False)
    partial = document(
        "Heading",
        [
            ExtractedSegment(
                "Heading",
                0,
                7,
                element_id="new",
                kind=SegmentKind.HEADING,
                order=0,
                metadata={"parent_annotated": True},
            )
        ],
    )
    result = score(item, partial)
    assert result.metrics["layout.element_type_recall"] == 0.5
    assert result.metrics["layout.hierarchy_preservation_rate"] == 0.5
    assert result.metrics["text.reading_order_ned"] == 0.5
    missing = score(item, document())
    assert missing.metrics["text.reading_order_ned"] == 1
    assert missing.metrics["layout.element_type_recall"] == 0


def test_extra_element_is_an_insertion_in_reading_order():
    item = {
        "id": "extra",
        "kind": "docx",
        "reference_elements": [{"text": "A", "order": 0}],
        "reference_capabilities": ["reading_order"],
    }
    result = score(
        item,
        document("A\nB", [ExtractedSegment("A", 0, 1), ExtractedSegment("B", 2, 3)]),
    )
    assert result.metrics["text.reading_order_ned"] == 0.5


def test_detection_macro_retains_verified_negatives():
    negative = {
        "id": "negative",
        "kind": "docx",
        "reference_elements": [],
        "reference_capabilities": ["element_types"],
    }
    positive = {
        "id": "positive",
        "kind": "docx",
        "reference_elements": [{"text": str(index)} for index in range(9)],
        "reference_capabilities": ["element_types"],
    }
    records = [score(negative, document()), score(positive, document())]
    metrics, _ = aggregate_evaluations(records, resamples=0, seed=42, confidence=0.95)
    assert metrics["layout.element_f1"] == 0.5


def test_incidental_ids_do_not_change_repeatability_but_parent_structure_does():
    item = {"id": "repeat", "kind": "docx", "reference": "A B"}
    first = document(
        "A B",
        [
            ExtractedSegment("A", 0, 1, element_id="a"),
            ExtractedSegment("B", 2, 3, element_id="b", parent_id="a"),
        ],
    )
    renamed = document(
        "A B",
        [
            replace(first.segments[0], element_id="x"),
            replace(first.segments[1], element_id="y", parent_id="x"),
        ],
    )
    root = document(
        "A B", [first.segments[0], replace(first.segments[1], parent_id=None)]
    )
    result = score(item, first, repeated_documents=(first, renamed, root))
    assert result.metrics["reliability.repeatability_success_rate"] == 1 / 3


def test_missing_predicted_parent_ids_cannot_verify_hierarchy():
    item = {
        "id": "hierarchy",
        "kind": "docx",
        "reference_capabilities": ["hierarchy"],
        "reference_elements": [
            {"id": "p", "text": "Parent", "parent_id": None},
            {"id": "c", "text": "Child", "parent_id": "p"},
        ],
    }
    result = score(
        item,
        document(
            "Parent Child",
            [ExtractedSegment("Parent", 0, 6), ExtractedSegment("Child", 7, 12)],
        ),
    )
    assert result.metrics["layout.hierarchy_preservation_rate"] == 0


@pytest.mark.parametrize("suite", ["document", "audio", "video"])
@pytest.mark.parametrize("failed_attempt", [1, 3])
def test_workers_preserve_designated_quality_and_all_measured_attempts(
    suite, failed_attempt, monkeypatch
):
    import importlib
    import time

    from experiments.benchmarks.extraction.audio.adapters import Transcript

    module = "visual_worker" if suite == "video" else "worker"
    worker = importlib.import_module(
        f"experiments.benchmarks.extraction.{suite}.{module}"
    )
    protocol_module = importlib.import_module(
        f"experiments.benchmarks.extraction.{suite}.protocol"
    )
    settings = deepcopy(protocol_module.load_protocol().meta.resolved)
    settings["profiles"]["smoke"]["repetitions"] = 3
    protocol = protocol_module.protocol_from_mapping(settings)
    events = []

    class Runtime:
        requests = 0

        def load(self):
            events.append("load")

        def initialize_image_pipeline(self, request):
            self.load()

        def inference(self):
            self.requests += 1
            events.append("inference")
            # Warmup is call 1 and is discarded; measured attempts follow it.
            if self.requests == failed_attempt + 1:
                raise RuntimeError("injected request failure")

        def extract(self, request, kind):
            self.inference()
            return document("alpha", [ExtractedSegment("alpha", 0, 5)])

        def transcribe(self, path):
            self.inference()
            return Transcript("alpha", ({"text": "alpha", "start": 0, "end": 1},))

        def close(self):
            events.append("close")

        def parameters(self):
            return {}

    runtime = Runtime()
    monkeypatch.setattr(worker, "seed_deterministically", lambda _seed: None)
    monkeypatch.setattr(worker, "synchronize", lambda *_args: None)
    item = {
        "id": "one",
        "kind": "docx",
        "source_path": "unused",
        "asset_sha256": "checksum",
        "duration_seconds": 1,
        "canonical_path": "unused.wav",
        "reference": "alpha",
        "reference_segments": [{"text": "alpha", "start": 0, "end": 1}],
        "reference_visual_text": ["alpha"],
        "visual_occurrences": [{"id": "occ", "text": "alpha", "start": 0, "end": 1}],
    }
    payload = {
        "_worker_started_at": time.perf_counter() - 10,
        "profile": "smoke",
        "device": "cpu",
        "protocol": protocol.meta.worker_payload(),
    }
    if suite == "audio":
        monkeypatch.setattr(worker, "build_runtime", lambda *_args: runtime)
        payload.update(candidate="mock", model_lock={}, speech=[item], reliability=[])
    else:
        monkeypatch.setattr(
            worker,
            "build_experiment_registry",
            lambda: SimpleNamespace(create=lambda *_args: runtime),
        )
        payload["items"] = [item]
        if suite == "document":
            monkeypatch.setattr(worker, "initialize", lambda *_args: runtime.load())
            payload.update(
                candidate="docling-standard-native",
                model_lock={"docling-standard": {"revision": "system"}},
            )
        else:
            from experiments.benchmarks.extraction.audio.protocol import (
                load_protocol as asr_protocol,
            )
            from experiments.benchmarks.extraction.document.protocol import (
                load_protocol as doc_protocol,
            )

            doc = doc_protocol()
            monkeypatch.setattr(
                worker,
                "load_frozen_asr_artifact",
                lambda *_args, **_kwargs: ({}, "checksum"),
            )
            monkeypatch.setattr(
                worker,
                "_image_request",
                lambda *_args: SimpleNamespace(source_kind=SourceKind.IMAGE),
            )
            monkeypatch.setattr(
                worker,
                "_extract_frames",
                lambda *_args: ([(Path("frame.png"), 0.0)], ["ffmpeg"]),
            )
            payload.update(
                candidate="video-fixed-5s",
                document_protocol=doc.meta.worker_payload(),
                audio_protocol=asr_protocol().meta.worker_payload(),
                image_candidate="docling-standard",
                image_engine="docling-standard",
                image_revision="revision",
                image_options=doc.parser_options("docling-standard"),
                frozen_asr_path="unused",
                manifest_checksum="manifest",
                frozen_asr_identity={},
                frozen_asr_checksum="checksum",
            )
    result = worker.execute(payload)
    attempts = result["attempts"] if suite == "document" else result["timings"]
    assert len(attempts) == 3
    assert [row["success"] for row in attempts] == [
        index != failed_attempt for index in range(1, 4)
    ]
    assert events[:2] == ["load", "inference"]
    assert events.count("inference") == 4
    if suite == "document":
        assert result["operational"]["first_item_latency_seconds"] >= 10
        assert (attempts[0]["document"] is None) == (failed_attempt == 1)
    else:
        if suite == "audio":
            assert all(not {"text", "segments"} & row.keys() for row in attempts)
            assert (
                attempts[0]["projected_transcript_sha256"] is not None
                or failed_attempt == 1
            )
        metric = "word_error_rate" if suite == "audio" else "visual_content_f1"
        expected = None if failed_attempt == 1 else 0 if suite == "audio" else 1
        assert result["metrics"][metric] == expected
        repeat = (
            "transcript_repeatability_success_rate"
            if suite == "audio"
            else "repeatability_success_rate"
        )
        assert result["metrics"][repeat] == pytest.approx(1 / 3)
        assert result["metrics"]["attempt_failure_rate"] == pytest.approx(1 / 3)
        assert result["samples"][0]["metric_statuses"][metric]["status"] == (
            "unavailable" if failed_attempt == 1 else "scored"
        )


def test_data_verification_reuses_sealed_report_without_running_full_validator(
    tmp_path,
):
    import json

    from edumind.common.artifacts import atomic_write_json, sha256_file
    from experiments.benchmarks.common.datasets import manifest_content_checksum
    from experiments.benchmarks.extraction.document.validation import (
        prepare_report,
        verified_inputs,
    )

    manifest = json.loads(
        (ROOT / "data/benchmarks/extraction/smoke.json").read_text(encoding="utf-8")
    )
    manifest["samples"] = [
        next(row for row in manifest["samples"] if row["kind"] == "image")
    ]
    asset = tmp_path / "image.png"
    asset.write_bytes((ROOT / manifest["samples"][0]["source_path"]).read_bytes())
    manifest["samples"][0]["source_path"] = str(asset)
    manifest["checksum"] = manifest_content_checksum(manifest["samples"])
    path = tmp_path / "manifest.json"
    atomic_write_json(path, manifest)
    report_root = tmp_path / "reports"
    destination, report = prepare_report(
        path, "smoke", load_protocol(), report_root=report_root
    )
    assert report["status"] == "passed"
    assert (
        verified_inputs(path, "smoke", load_protocol(), report_root=report_root)[0]
        == destination
    )
    report["metadata"]["samples"][manifest["samples"][0]["id"]][
        "physical_page_count"
    ] = 99
    atomic_write_json(destination, report)
    with pytest.raises(ValueError, match="report content checksum"):
        verified_inputs(path, "smoke", load_protocol(), report_root=report_root)
    prepare_report(path, "smoke", load_protocol(), report_root=report_root)
    original_checksum = sha256_file(asset)
    asset.write_bytes(asset.read_bytes() + b"changed")
    assert sha256_file(asset) != original_checksum
    with pytest.raises(ValueError, match="Missing or stale"):
        verified_inputs(path, "smoke", load_protocol(), report_root=report_root)
