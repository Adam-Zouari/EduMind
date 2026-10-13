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
from experiments.benchmarks.common.protocol import unique_strings
from experiments.benchmarks.extraction.document.metrics import (
    DocumentEvaluation,
    aggregate_evaluations,
    apply_official_metrics,
    score_document,
    validate_reference,
)
from experiments.benchmarks.extraction.document.protocol import load_protocol
from experiments.benchmarks.extraction.scoring import attempt_rates, bootstrap_sources

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("values", [[], "a", None, [""], [1], ["a", " a "]])
def test_shared_protocol_strings_reject_empty_invalid_and_duplicate_values(values):
    with pytest.raises(ValueError):
        unique_strings(values, "values")


def test_shared_protocol_strings_preserve_order_and_trim_values():
    assert unique_strings([" a ", "b"], "values") == ("a", "b")


def test_official_scores_keep_document_boundaries_and_content_scores(monkeypatch):
    from experiments.benchmarks.extraction.document import metrics

    records = [
        DocumentEvaluation(("pdf",)),
        DocumentEvaluation(
            ("pdf",),
            metrics={"tables.content_f1": 0.5},
            table_pairs=[("one", "one"), ("two", "two")],
            formula_pairs=[("x", "x")],
        ),
        DocumentEvaluation(
            ("pdf",),
            table_pairs=[("three", "three")],
            formula_pairs=[("y", "y"), ("z", "z")],
        ),
    ]
    monkeypatch.setattr(
        metrics,
        "score_official_metrics",
        lambda *_args, **_kwargs: ([(0.2, 0.4), (0.6, 0.8), (1, 1)], [0.5, 0, 1]),
    )
    apply_official_metrics(records, timeout_seconds=1)
    assert records[0].metrics == {}
    assert records[1].metrics == pytest.approx(
        {
            "tables.content_f1": 0.5,
            "tables.teds": 0.4,
            "tables.teds_s": 0.6,
            "formulas.recognition_similarity": 0.5,
            "formulas.exact_match": 0,
        }
    )
    assert records[2].metrics == pytest.approx(
        {
            "tables.teds": 1,
            "tables.teds_s": 1,
            "formulas.recognition_similarity": 0.5,
            "formulas.exact_match": 0.5,
        }
    )


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
    from experiments.benchmarks.extraction.document.worker import (
        initialize,
        pipeline_parameters,
    )

    initialized_formats = []

    class InitializedPipeline:
        def __init__(self, options):
            self.pipeline_options = options

    def converter(*, format_options):
        runtime = SimpleNamespace(
            format_to_options=format_options, initialized_pipelines={}
        )

        def initialize_pipeline(input_format):
            initialized_formats.append(input_format)
            runtime.initialized_pipelines[(InitializedPipeline, input_format.value)] = (
                InitializedPipeline(format_options[input_format].pipeline_options)
            )

        runtime.initialize_pipeline = initialize_pipeline
        return runtime

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
    assert InputFormat.PDF in runtime.format_to_options
    assert pipeline_parameters(extractor) == {}
    initialize(extractor, request)
    assert initialized_formats == [InputFormat.IMAGE]
    parameters = pipeline_parameters(extractor)
    assert set(parameters) == {f"0:InitializedPipeline:{InputFormat.IMAGE.value}"}
    logged = next(iter(parameters.values()))
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


def test_empty_device_map_is_not_model_placement_evidence():
    from edumind.common.model_placement import inspect_model_placement

    report = inspect_model_placement(SimpleNamespace(hf_device_map={}))
    assert report["status"] == "placement_unverifiable"
    assert not report["verified"]


def test_content_cache_reuses_extraction_with_current_source_identity(tmp_path):
    from edumind.extraction import ExtractionRequest
    from edumind.extraction.cache import ExtractionCache

    original, renamed = tmp_path / "original.png", tmp_path / "renamed.png"
    original.write_bytes(b"same source content")
    renamed.write_bytes(original.read_bytes())
    prediction = document("content")
    request = ExtractionRequest.from_path(
        original, profile=prediction.profile, source_kind=SourceKind.IMAGE
    )
    cache = ExtractionCache(tmp_path / "cache")
    cache.put(request, prediction)
    current = replace(request, source_path=renamed)
    assert cache.key(current) == cache.key(request)
    loaded = cache.get(current)
    assert loaded.cache_hit
    assert loaded.source_name == renamed.name
    assert loaded.source_path == str(renamed)
    assert loaded.text == "content"
    assert cache.key(replace(request, source_kind=SourceKind.PDF)) != cache.key(request)
    assert cache.key(replace(request, mime_type="image/png")) != cache.key(request)


@pytest.mark.parametrize("payload", ["[]", "null", '{"schema": 1, "document": []}'])
def test_corrupt_cache_shape_is_a_cache_miss(tmp_path, payload):
    from edumind.extraction import ExtractionRequest
    from edumind.extraction.cache import ExtractionCache

    cache = ExtractionCache(tmp_path)
    request = ExtractionRequest(
        tmp_path / "image", "fixture", profile=document().profile
    )
    (tmp_path / f"{cache.key(request)}.json").write_text(payload, encoding="utf-8")
    assert cache.get(request) is None


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


@pytest.mark.parametrize(
    "capability,presence", [("tables", "has_table"), ("formulas", "has_formula")]
)
@pytest.mark.parametrize("element_field", ["elements", "reference_elements"])
def test_structured_negative_reference_requires_explicit_element_list(
    capability, presence, element_field
):
    from experiments.benchmarks.extraction.document.metrics import (
        _reference_from_mapping,
    )

    payload = {"reference_capabilities": [capability], presence: False}
    reference = _reference_from_mapping(payload)
    item = {"id": "negative", "kind": "docx", "reference_path": "unused.json"}
    with pytest.raises(ValueError, match="without an element annotation list"):
        validate_reference(
            item, authoritative=True, payload=payload, reference=reference
        )
    payload[element_field] = []
    validate_reference(item, authoritative=True, payload=payload, reference=reference)


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


@pytest.mark.parametrize(
    "reference_kind,predicted_kind,text",
    [("code", SegmentKind.TEXT, "X=Y"), ("text", SegmentKind.CODE, "Hello")],
)
def test_matching_uses_reference_representation_before_scoring_type(
    reference_kind, predicted_kind, text
):
    item = {
        "id": "fixed-representation",
        "kind": "docx",
        "reference_capabilities": ["element_types", "reading_order"],
        "reference_elements": [
            {"id": "r", "kind": reference_kind, "text": text, "order": 0}
        ],
    }
    result = score(
        item,
        document(text, [ExtractedSegment(text, 0, len(text), kind=predicted_kind)]),
    )
    assert result.metrics["layout.element_f1"] == 1
    assert result.metrics["layout.element_type_recall"] == 0
    assert result.metrics["text.reading_order_ned"] == 0
    if reference_kind == "code":
        changed = score(
            item, document(text.lower(), [ExtractedSegment(text.lower(), 0, len(text))])
        )
        assert changed.metrics["layout.element_f1"] == 0


def test_type_recall_excludes_unannotated_reference_types():
    item = {
        "id": "partial-types",
        "kind": "docx",
        "reference_capabilities": ["element_types"],
        "reference_elements": [
            {"id": "a", "kind": "text", "text": "A"},
            {"id": "b", "text": "B"},
        ],
    }
    result = score(
        item,
        document(
            "A B",
            [
                ExtractedSegment("A", 0, 1),
                ExtractedSegment("B", 2, 3, kind=SegmentKind.HEADING),
            ],
        ),
    )
    assert result.metrics["layout.element_type_recall"] == 1
    assert score(item, document()).metrics["layout.element_type_recall"] == 0
    del item["reference_elements"][0]["kind"]
    result = score(item, document())
    assert result.metrics["layout.element_type_recall"] is None
    assert result.statuses["layout.element_type_recall"]["status"] == "inapplicable"


def test_document_bootstrap_uses_metric_eligible_sources_and_document_macro():
    name = "tables.teds"
    records = [
        DocumentEvaluation(
            ("pdf",),
            metrics={name: value},
            source_group_id=source,
            statuses={name: {"status": "scored", "reason": None}},
        )
        for source, value in [("a", 0.0), ("a", 1.0), ("b", 1.0)]
    ]
    records.extend(
        DocumentEvaluation(
            ("pdf",),
            metrics={name: None},
            source_group_id=str(i),
            statuses={name: {"status": "inapplicable", "reason": "no_reference_task"}},
        )
        for i in range(8)
    )
    estimates, intervals = aggregate_evaluations(
        records, resamples=200, seed=42, confidence=0.95, minimum_sources=2
    )
    bounds = intervals[name]
    assert estimates[name] == pytest.approx(2 / 3)
    assert bounds["independent_source_count"] == 2
    assert bounds["undefined_resamples"] == 0
    assert (bounds["lower"], bounds["upper"]) == (0.5, 1.0)
    for record in records[:3]:
        record.metrics[name] = None
        record.statuses[name] = {
            "status": "unavailable",
            "reason": "first_attempt_failed",
        }
    records.append(
        DocumentEvaluation(
            ("pdf",),
            metrics={name: 1.0},
            source_group_id="c",
            statuses={name: {"status": "scored", "reason": None}},
        )
    )
    _, intervals = aggregate_evaluations(
        records, resamples=200, seed=42, confidence=0.95, minimum_sources=1
    )
    assert intervals[name]["independent_source_count"] == 3
    assert intervals[name]["contributing_source_count"] == 1
    assert intervals[name]["undefined_resamples"] > 0


def test_bootstrap_batches_identical_row_cohorts_with_per_metric_support(monkeypatch):
    from experiments.benchmarks.extraction.document import metrics

    first, second, table = "text.content_f1", "text.character_error_rate", "tables.teds"
    records = [
        DocumentEvaluation(
            (),
            source_group_id=source,
            metrics={first: a, second: b, table: t},
            statuses={
                first: {"status": "scored"},
                second: {"status": "unavailable" if b is None else "scored"},
                table: {"status": "inapplicable" if t is None else "scored"},
            },
        )
        for source, a, b, t in [
            ("a", 0.0, 0.0, 1.0),
            ("a", 1.0, 1.0, None),
            ("b", 1.0, None, 0.0),
        ]
    ]
    calls = []
    original = metrics.bootstrap_sources

    def recorded(rows, aggregate, **options):
        calls.append((len(rows), set(aggregate(rows))))
        return original(rows, aggregate, **options)

    monkeypatch.setattr(metrics, "bootstrap_sources", recorded)
    _, bounds = aggregate_evaluations(
        records, resamples=200, seed=42, confidence=0.95, minimum_sources=2
    )
    assert calls == [(2, {table}), (3, {first, second})]
    assert bounds[first]["contributing_source_count"] == 2
    assert bounds[first]["undefined_resamples"] == 0
    assert bounds[second]["contributing_source_count"] == 1
    assert bounds[second]["reason"] == "insufficient_independent_sources"
    assert bounds[second]["lower"] is None
    assert bounds[second]["undefined_resamples"] > 0
    for name, indices in [(first, [0, 1, 2]), (table, [0, 2])]:
        rows = [
            {"source_group_id": records[i].source_group_id, "record": records[i]}
            for i in indices
        ]
        separate = original(
            rows,
            lambda draw, name=name: metrics._aggregate_group(
                [row["record"] for row in draw], names=(name,)
            ),
            resamples=200,
            seed=42,
            confidence=0.95,
            minimum_sources=2,
        )
        assert bounds[name] == separate[name]


def test_reference_counts_are_fixed_before_first_attempt_failure():
    item = {
        "id": "counts",
        "kind": "image",
        "has_table": True,
        "has_formula": True,
        "reference_capabilities": [
            "pages",
            "reading_order",
            "element_types",
            "hierarchy",
            "layout_boxes",
            "tables",
            "formulas",
        ],
        "reference_page_texts": ["H P X"],
        "reference_elements": [
            {
                "id": "h",
                "kind": "heading",
                "text": "H",
                "order": 0,
                "page_number": 1,
                "parent_id": None,
                "bounding_box": [0.1, 0.1, 0.2, 0.2],
            },
            {
                "id": "p",
                "text": "P",
                "order": 1,
                "page_number": 1,
                "bounding_box": [0.2, 0.2, 0.3, 0.3],
            },
            {
                "id": "t",
                "kind": "table",
                "order": 2,
                "page_number": 1,
                "html": "<table><tr><td></td></tr></table>",
            },
            {"id": "f", "kind": "formula", "order": 3, "page_number": 1, "latex": "X"},
        ],
    }
    validate_reference(item, authoritative=False)
    expected = {
        "elements": 4,
        "layout_elements": 2,
        "tables": 1,
        "formulas": 1,
        "pages": 1,
        "reading_order_elements": 4,
        "typed_layout_elements": 1,
        "hierarchy_elements": 1,
        "boxed_layout_elements": 2,
        "page_attribution_elements": 3,
    }
    assert score(item, document()).reference_counts == expected
    assert score(item, None, failed=True).reference_counts == expected
    item["reference_capabilities"] = ["pages"]
    counts = score(item, document()).reference_counts
    assert counts["elements"] == 4
    assert (
        counts["layout_elements"]
        == counts["typed_layout_elements"]
        == counts["tables"]
        == counts["formulas"]
        == 0
    )


def test_detection_aggregate_reports_reference_labels_including_failed_positives():
    positive = {
        "id": "positive",
        "kind": "docx",
        "has_table": True,
        "has_formula": True,
        "reference_capabilities": ["element_types", "tables", "formulas"],
        "reference_elements": [
            {"id": "p", "kind": "text", "text": "P"},
            {"id": "t", "kind": "table", "html": "<table><tr><td>T</td></tr></table>"},
            {"id": "f", "kind": "formula", "latex": "X"},
        ],
    }
    negative = {
        **positive,
        "id": "negative",
        "has_table": False,
        "has_formula": False,
        "reference_elements": [],
    }
    unclaimed = {
        "id": "unclaimed",
        "kind": "docx",
        "reference": "Text",
        "reference_capabilities": ["text"],
    }
    records = [
        score(positive, None, failed=True),
        score(negative, document()),
        score(unclaimed, document()),
    ]
    estimates, _ = aggregate_evaluations(records, resamples=0, seed=42, confidence=0.95)
    for name in ("layout.element_f1", "tables.detection_f1", "formulas.detection_f1"):
        assert estimates[name + ".reference_positive_count"] == 1
        assert estimates[name + ".verified_negative_count"] == 1
        assert estimates[name + ".eligible_count"] == 2
        assert estimates[name + ".sample_count"] == 1
        category, metric = name.split(".", 1)
        assert estimates[f"{category}.docx.{metric}.reference_positive_count"] == 1


def test_page_coverage_fallback_combines_emitted_page_segments():
    text = "a b c d e f g h"
    item = {
        "id": "page-text-only",
        "kind": "pdf",
        "reference_capabilities": ["pages"],
        "reference_page_texts": [text],
    }
    segments = [
        ExtractedSegment(text[i : i + 3], i, i + 3, page_number=1)
        for i in range(0, len(text), 4)
    ]
    result = score(item, document(text, segments))
    assert result.metrics["pages.page_content_f1"] == 1
    assert result.metrics["pages.page_coverage"] == 1
    assert score(item, document()).metrics["pages.page_coverage"] == 0


@pytest.mark.parametrize("cell_text", ["", "invented"])
def test_blank_table_matching_uses_structure_without_cell_text(cell_text):
    html = f'<table style="color:red"><tr><td rowspan="1">{cell_text}</td></tr></table>'
    item = {
        "id": "blank-table",
        "kind": "docx",
        "reference_capabilities": ["tables"],
        "has_table": True,
        "reference_elements": [
            {"id": "r", "kind": "table", "html": "<table><tr><td></td></tr></table>"}
        ],
    }
    validate_reference(item, authoritative=False)
    predicted = document(
        cell_text,
        [
            ExtractedSegment(
                cell_text,
                0,
                len(cell_text),
                kind=SegmentKind.TABLE,
                structured_content={"rows": [[cell_text]], "html": html},
            )
        ],
    )
    result = score(item, predicted)
    assert result.metrics["tables.detection_f1"] == 1
    assert result.metrics["tables.content_f1"] == (0 if cell_text else 1)
    assert result.table_pairs[0][1] == html
    assert result.metrics["reliability.unexpected_empty_output_rate"] == 0
    assert score(item, document()).metrics["tables.content_f1"] == 0


@pytest.mark.parametrize(
    "html",
    [
        "<table><thead><tr><td></td></tr></thead></table>",
        '<table><tr><td colspan="2"></td></tr></table>',
        "<table><tr><th></th></tr></table>",
        '<table><tr><td rowspan="bad"></td></tr></table>',
        "<table><tr><td>",
    ],
)
def test_blank_table_structure_fallback_rejects_changed_or_invalid_structure(html):
    from experiments.benchmarks.extraction.document.metrics import (
        ReferenceElement,
        _match_elements,
    )

    reference = ReferenceElement(
        "r",
        SegmentKind.TABLE,
        html="<table><tr><td></td></tr></table>",
        identity="same-id",
    )
    prediction = ExtractedSegment(
        "",
        0,
        0,
        kind=SegmentKind.TABLE,
        structured_content={"rows": [[""]], "html": html, "identity": "same-id"},
    )
    assert _match_elements([reference], [prediction], visual=False, threshold=0.5) == []


@pytest.mark.parametrize(
    "capability", ["element_types", "reading_order", "hierarchy", "pages"]
)
def test_claimed_matching_tasks_reject_unidentifiable_reference_elements(capability):
    item = {
        "id": "unidentifiable",
        "kind": "docx",
        "reference_capabilities": [capability],
        "reference_page_texts": [""],
        "reference_elements": [
            {
                "id": "r",
                "kind": "figure",
                "text": "",
                "order": 0,
                "parent_id": None,
                "page_number": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="frozen non-text identity"):
        validate_reference(item, authoritative=False)
    item["reference_elements"][0]["identity"] = "media-1"
    validate_reference(item, authoritative=False)


def test_prose_only_reference_does_not_require_incidental_element_identity_or_page():
    item = {
        "id": "prose-only",
        "kind": "image",
        "reference": "Text",
        "reference_capabilities": ["text"],
        "reference_elements": [{"id": "r", "kind": "figure"}],
    }
    validate_reference(item, authoritative=False)
    item["reference_capabilities"] = ["layout_boxes"]
    item["reference_elements"][0].update(
        page_number=1, bounding_box=[0.1, 0.1, 0.5, 0.5]
    )
    validate_reference(item, authoritative=False)


def _cdm_renderer(code=0):
    return SimpleNamespace(
        run_cmd=lambda *_args, **_kwargs: code,
        _parse_pdflatex_log=lambda _path: (1, False),
        formular_template=r"\documentclass{article}\usepackage{amsmath}",
        formular_template_zh="",
        tabular_template="",
    )


@pytest.mark.parametrize(
    "command,code",
    [
        ("pdflatex formula.tex", -1),
        ("pdflatex formula.tex", 126),
        ("pdflatex formula.tex", 127),
        ("magick formula.pdf formula.png", 1),
    ],
)
def test_cdm_infrastructure_command_failures_are_unavailable(command, code):
    from experiments.benchmarks.extraction.document.omnidocbench_worker import (
        _score_formula,
    )

    renderer = _cdm_renderer(code)
    original = renderer.run_cmd

    def evaluate(*_args):
        renderer.run_cmd(command)
        return {"F1_score": 0.0, "gt_tokens": 1, "pred_tokens": 0}

    with pytest.raises(RuntimeError, match="infrastructure failed"):
        _score_formula(
            SimpleNamespace(evaluate=evaluate), renderer, "x", "bad", "fixture"
        )
    assert renderer.run_cmd is original


def test_cdm_bad_predicted_latex_is_quality_zero_and_empty_reference_render_is_not():
    from experiments.benchmarks.extraction.document.omnidocbench_worker import (
        _score_formula,
    )

    renderer = _cdm_renderer(1)

    def evaluate(*_args):
        renderer.run_cmd("pdflatex bad.tex")
        return {"F1_score": 0.0, "gt_tokens": 1, "pred_tokens": 0}

    assert (
        _score_formula(
            SimpleNamespace(evaluate=evaluate), renderer, "x", r"\bad", "fixture"
        )
        == 0
    )
    evaluator = SimpleNamespace(
        evaluate=lambda *_args: {"F1_score": 0.0, "gt_tokens": 0, "pred_tokens": 0}
    )
    with pytest.raises(RuntimeError, match="reference: rendered no tokens"):
        _score_formula(evaluator, renderer, "x", "x", "fixture")
    evaluator = SimpleNamespace(
        evaluate=lambda *_args: {
            "F1_score": 0.0,
            "gt_tokens": 0,
            "cdm_eval_error": "broken evaluator",
        }
    )
    with pytest.raises(RuntimeError, match="CDM evaluation failed"):
        _score_formula(evaluator, renderer, "x", "x", "fixture")


@pytest.mark.parametrize(
    "missing,unavailable", [("amsmath.sty", True), ("prediction-input.tex", False)]
)
def test_cdm_missing_template_dependency_is_distinct_from_bad_prediction_input(
    tmp_path, missing, unavailable
):
    from experiments.benchmarks.extraction.document.omnidocbench_worker import (
        _score_formula,
    )

    log = tmp_path / "formula.log"
    log.write_text(f"! LaTeX Error: File `{missing}' not found.", encoding="utf-8")
    renderer = _cdm_renderer(1)

    def evaluate(*_args):
        renderer.run_cmd("pdflatex bad.tex")
        renderer._parse_pdflatex_log(log)
        return {"F1_score": 0.0, "gt_tokens": 1, "pred_tokens": 0}

    evaluator = SimpleNamespace(evaluate=evaluate)
    if unavailable:
        with pytest.raises(
            RuntimeError, match="missing evaluator dependency amsmath.sty"
        ):
            _score_formula(evaluator, renderer, "x", "bad", "fixture")
    else:
        assert _score_formula(evaluator, renderer, "x", "bad", "fixture") == 0


@pytest.mark.parametrize("reference_tokens,expected", [(1, 0.0), (0, None)])
def test_cdm_worker_checks_reference_tokens_before_emitting_quality(
    tmp_path, monkeypatch, reference_tokens, expected
):
    import json
    import sys
    from types import ModuleType

    from experiments.benchmarks.extraction.document import omnidocbench_worker as worker

    modules = {
        name: ModuleType(name)
        for name in (
            "metrics",
            "metrics.cdm",
            "metrics.cdm.modules",
            "metrics.cdm_metric",
        )
    }
    modules["metrics.cdm.modules"].latex2bbox_color = _cdm_renderer()

    class CDM:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, *_args):
            return {"F1_score": 0.0, "gt_tokens": reference_tokens, "pred_tokens": 0}

    modules["metrics.cdm_metric"].CDM = CDM
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CDM_SAVE_VIS", "0")
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value: tmp_path / "cdm" if str(value) == "/work/cdm" else Path(value),
    )
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(
        json.dumps({"formulas": [{"reference": "x", "prediction": r"\bad"}]}),
        encoding="utf-8",
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="reference: rendered no tokens"):
            worker.main(input_path, output_path)
        assert not output_path.exists()
    else:
        worker.main(input_path, output_path)
        assert json.loads(output_path.read_text(encoding="utf-8"))["formulas"] == [
            expected
        ]


def test_official_host_preserves_cdm_infrastructure_failure_as_unavailable(
    tmp_path, monkeypatch
):
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

    def failed(command, **_kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr="CDM renderer infrastructure failed: pdflatex failed with exit code -1",
        )

    monkeypatch.setattr(official.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="CDM renderer infrastructure failed"):
        official.score_official_metrics([], [("x", "x")], timeout_seconds=1)


@pytest.mark.parametrize(
    "suite,latency_support",
    [("document", None), ("audio", None), ("video", None), ("video", 2)],
)
@pytest.mark.parametrize("failed_attempt", [1, 3])
def test_workers_preserve_designated_quality_and_all_measured_attempts(
    suite, latency_support, failed_attempt, monkeypatch
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
    if suite == "video":
        settings["statistics"]["minimum_latency_ci_sources"] = latency_support
    protocol = protocol_module.protocol_from_mapping(settings)
    events = []
    bootstrap_minima = []

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
            return replace(
                document("alpha", [ExtractedSegment("alpha", 0, 5)]), source_kind=kind
            )

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
            original_bootstrap = worker.bootstrap_sources

            def recorded_bootstrap(*args, **kwargs):
                bootstrap_minima.append(kwargs["minimum_sources"])
                return original_bootstrap(*args, **kwargs)

            monkeypatch.setattr(worker, "bootstrap_sources", recorded_bootstrap)
            monkeypatch.setattr(
                worker,
                "load_frozen_asr_artifact",
                lambda *_args, **_kwargs: ({}, "checksum"),
            )
            monkeypatch.setattr(
                worker,
                "_image_request",
                lambda *_args: SimpleNamespace(
                    source_kind=SourceKind.IMAGE, checksum="checksum"
                ),
            )
            monkeypatch.setattr(
                worker,
                "_extract_frames",
                lambda *_args, **_kwargs: ([(Path("frame.png"), 0.0)], ["ffmpeg"]),
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
                canonical_audio_checksums={"one": "checksum"},
                ffmpeg_version="test",
            )
    result = worker.execute(payload)
    if suite == "video":
        assert bootstrap_minima == (
            [None] if latency_support is None else [None, latency_support]
        )
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


@pytest.mark.parametrize("text", [None, "", "alpha"])
def test_claimed_image_pages_require_explicit_page_annotations(text):
    payload = {"kind": "image", "reference_capabilities": ["pages"]}
    if text is not None:
        payload["text"] = text
    with pytest.raises(ValueError, match="verified page text"):
        validate_reference(
            {"id": "image", "kind": "image", "reference_path": "reference.json"},
            authoritative=True,
            payload=payload,
        )


@pytest.mark.parametrize("text", ["", "alpha"])
def test_smoke_image_page_inference_requires_supplied_text(text):
    from experiments.benchmarks.extraction.document.metrics import load_reference

    reference = load_reference({"id": "image", "kind": "image", "text": text})
    assert reference.pages == {1: text}
    assert "pages" in reference.capabilities
    missing = load_reference({"id": "image", "kind": "image"})
    assert missing.pages == {}
    assert "pages" not in missing.capabilities
