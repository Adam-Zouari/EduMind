"""Native Docling outcomes through both image parsers; no models or GPU work."""

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.extraction import ExtractionProfile, ExtractionRequest, SourceKind
from edumind.extraction.errors import ExtractionBackendError
from edumind.extraction.extractors.document import (
    DoclingExtractor,
    docling_conversion_output,
)
from experiments.benchmarks.extraction.document.adapters import (
    ExperimentalDocumentExtractor,
)
from experiments.benchmarks.extraction.document.worker import validate_document_payload
from experiments.benchmarks.extraction.video import visual_worker


class Status(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    PENDING = "pending"
    STARTED = "started"


def _native(text=""):
    items = (
        []
        if not text
        else [
            (
                SimpleNamespace(
                    text=text, label="text", self_ref="#/texts/0", parent=None, prov=[]
                ),
                1,
            )
        ]
    )
    return SimpleNamespace(iterate_items=lambda **_kwargs: iter(items))


def _error(message="OCR region failed", *, page=1, category="inference_failure"):
    return SimpleNamespace(
        component_type=SimpleNamespace(value="model"),
        module_name="RapidOcrModel",
        error_message=message,
        category=SimpleNamespace(value=category),
        page_no=page,
    )


def _result(status=Status.SUCCESS, *, text="", errors=()):
    return SimpleNamespace(status=status, document=_native(text), errors=list(errors))


@pytest.fixture(params=["docling-standard", "docling-vlm-granite-258m"])
def parser(request, monkeypatch):
    engine = request.param
    source = ExtractionRequest(
        Path("frame.png"),
        "frame-checksum",
        source_kind=SourceKind.IMAGE,
        profile=ExtractionProfile("visual", engine, "pinned", device="cpu"),
    )
    state = SimpleNamespace(result=_result(), calls=[], failure=None)

    def convert(path, *, raises_on_error):
        state.calls.append((path, raises_on_error))
        if state.failure is not None:
            raise state.failure
        return state.result

    converter = SimpleNamespace(convert=convert)
    if engine == "docling-standard":
        extractor = DoclingExtractor("pinned")
        monkeypatch.setattr(extractor, "_converter", lambda *_args: (converter, None))
    else:
        extractor = ExperimentalDocumentExtractor(engine, "pinned")
        monkeypatch.setattr(extractor, "_docling_converter", lambda *_args: converter)
    state.extractor, state.request, state.engine = extractor, source, engine
    return state


def test_successful_blank_is_complete_canonical_output_not_a_failed_request(parser):
    document = parser.extractor.extract(parser.request, SourceKind.IMAGE)
    assert document.text == ""
    assert document.segments == document.warnings == ()
    assert document.metadata["docling_conversion_status"] == "success"
    assert document.metadata["docling_conversion_errors"] == []
    assert document.metadata["engine"] == parser.engine
    assert document.metadata["engine_revision"] == "pinned"
    assert parser.calls == [("frame.png", False)]
    assert validate_document_payload(document.to_dict()).text == ""


@pytest.mark.parametrize("text", ["", "Visible text"])
def test_partial_output_keeps_text_native_diagnostics_and_recoverable_warnings(
    parser, text
):
    parser.result = _result(
        Status.PARTIAL_SUCCESS,
        text=text,
        errors=[
            _error("OCR region failed"),
            _error("enrichment unavailable", page=None, category="unknown"),
        ],
    )
    document = parser.extractor.extract(parser.request, SourceKind.IMAGE)
    assert document.text == text
    assert document.metadata["docling_conversion_status"] == "partial_success"
    assert document.metadata["docling_conversion_errors"] == [
        {
            "component_type": "model",
            "module_name": "RapidOcrModel",
            "error_message": "OCR region failed",
            "category": "inference_failure",
            "page_no": 1,
        },
        {
            "component_type": "model",
            "module_name": "RapidOcrModel",
            "error_message": "enrichment unavailable",
            "category": "unknown",
            "page_no": None,
        },
    ]
    assert [warning.code for warning in document.warnings] == [
        "docling_conversion_error"
    ] * 2
    assert all(
        warning.recoverable and warning.segment_index is None
        for warning in document.warnings
    )
    assert "page 1" in document.warnings[0].message
    assert "OCR region failed" in document.warnings[0].message
    assert (
        validate_document_payload(json.loads(json.dumps(document.to_dict()))).text
        == text
    )


def test_partial_status_without_errors_is_not_silently_clean_success(parser):
    parser.result = _result(Status.PARTIAL_SUCCESS)
    document = parser.extractor.extract(parser.request, SourceKind.IMAGE)
    assert document.text == ""
    assert document.metadata["docling_conversion_errors"] == []
    assert document.metadata["docling_conversion_status"] == "partial_success"
    assert document.warnings[0].code == "docling_partial_success"
    assert document.warnings[0].recoverable


@pytest.mark.parametrize(
    "status", [Status.FAILURE, Status.SKIPPED, Status.PENDING, Status.STARTED]
)
def test_non_deliverable_status_is_failure_even_when_native_document_has_text(
    parser, status
):
    parser.result = _result(
        status,
        text="Do not score this partial document",
        errors=[_error("native failure")],
    )
    with pytest.raises(ExtractionBackendError) as caught:
        parser.extractor.extract(parser.request, SourceKind.IMAGE)
    assert status.value in caught.value.detail
    assert "native failure" in caught.value.detail
    assert parser.calls == [("frame.png", False)]


@pytest.mark.parametrize(
    "status", [Status.FAILURE, Status.PARTIAL_SUCCESS, Status.SUCCESS]
)
def test_fatal_oom_details_reach_video_failure_classification(
    parser, monkeypatch, status
):
    parser.result = _result(
        status, errors=[_error("CUDA out of memory", category="capacity")]
    )
    monkeypatch.setattr(
        visual_worker,
        "_extract_frames",
        lambda *_args: ([(Path("frame.png"), 0.0)], ["ffmpeg"]),
    )
    monkeypatch.setattr(visual_worker, "_image_request", lambda *_args: parser.request)
    with pytest.raises(
        visual_worker.VisualAttemptFailure, match="CUDA out of memory"
    ) as caught:
        visual_worker._process_video(
            parser.extractor,
            None,
            {"source_path": "video.mp4", "duration_seconds": 10.0},
            Path("frames"),
            parser.engine,
            "pinned",
            "cpu",
            {},
        )
    assert caught.value.selected_frame_count == 1
    assert caught.value.command == ["ffmpeg"]


@pytest.mark.parametrize("status", [Status.SUCCESS, Status.PARTIAL_SUCCESS])
def test_video_accepts_deliverable_blank_and_warned_partial_without_loading_models(
    parser, monkeypatch, status
):
    parser.result = _result(
        status,
        text="" if status == Status.SUCCESS else "Visible text",
        errors=[] if status == Status.SUCCESS else [_error()],
    )
    monkeypatch.setattr(
        visual_worker,
        "_extract_frames",
        lambda *_args: ([(Path("frame.png"), 0.0)], ["ffmpeg"]),
    )
    monkeypatch.setattr(visual_worker, "_image_request", lambda *_args: parser.request)
    predictions, command, count, fingerprint = visual_worker._process_video(
        parser.extractor,
        None,
        {"source_path": "video.mp4", "duration_seconds": 10.0},
        Path("frames"),
        parser.engine,
        "pinned",
        "cpu",
        {},
    )
    assert count == 1 and command == ["ffmpeg"] and fingerprint
    assert predictions == (
        []
        if status == Status.SUCCESS
        else [{"text": "Visible text", "timestamp": 0.0, "warnings": 1}]
    )


def test_converter_exception_is_still_a_structured_request_failure(parser):
    parser.failure = RuntimeError("decoder failed before native outcome")
    with pytest.raises(ExtractionBackendError) as caught:
        parser.extractor.extract(parser.request, SourceKind.IMAGE)
    assert "decoder failed before native outcome" in caught.value.detail


@pytest.mark.parametrize(
    "status", ["success", Status.SUCCESS, "partial_success", Status.PARTIAL_SUCCESS]
)
def test_helper_is_pure_supports_native_enum_and_string_status_and_does_not_mutate(
    status,
):
    native = _result(status, errors=[_error()])
    original = dict(vars(native.errors[0]))
    document, metadata, warnings = docling_conversion_output(native)
    assert document is native.document
    assert native.errors[0].__dict__ == original
    assert metadata["docling_conversion_status"] == getattr(status, "value", status)
    assert (
        metadata["docling_conversion_errors"][0]["error_message"] == "OCR region failed"
    )
    assert warnings[0].recoverable
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("status", [None, "unknown", "SUCCESS", {}, ["success"]])
def test_missing_unknown_or_malformed_status_is_not_success(status):
    with pytest.raises(RuntimeError, match="did not complete"):
        docling_conversion_output(_result(status))


@pytest.mark.parametrize(
    "errors", [None, "error", {}, [SimpleNamespace(error_message="missing fields")]]
)
def test_malformed_native_error_records_are_not_dropped(errors):
    native = _result(Status.PARTIAL_SUCCESS)
    native.errors = errors
    with pytest.raises(ValueError, match="error records"):
        docling_conversion_output(native)


def test_completed_status_requires_a_document_but_does_not_require_nonempty_text():
    native = _result()
    native.document = None
    with pytest.raises(ValueError, match="lacks its document"):
        docling_conversion_output(native)


def test_installed_docling_datamodel_serializes_through_the_pure_helper():
    models = pytest.importorskip("docling.datamodel.base_models")
    documents = pytest.importorskip("docling.datamodel.document")
    native = documents.ConversionAssets(
        status=models.ConversionStatus.PARTIAL_SUCCESS,
        errors=[
            models.ErrorItem(
                component_type=models.DoclingComponentType.MODEL,
                module_name="fixture",
                error_message="Truncated VLM output",
                category=models.FailureCategory.INFERENCE_FAILURE,
                page_no=1,
            )
        ],
    )
    document, metadata, warnings = docling_conversion_output(native)
    assert document is native.document
    assert metadata["docling_conversion_status"] == "partial_success"
    assert metadata["docling_conversion_errors"] == [
        native.errors[0].model_dump(mode="json")
    ]
    assert len(warnings) == 1 and warnings[0].recoverable
