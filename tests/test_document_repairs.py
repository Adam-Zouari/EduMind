"""Behavioral regressions for the extraction repair handoff (no model inference)."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from edumind.common.artifacts import sha256_file
from edumind.extraction import ExtractedDocument, ExtractionProfile, SourceKind
from edumind.extraction.contracts import ExtractionRequest
from experiments.benchmarks.extraction.document import adapters, metrics, worker
from experiments.benchmarks.extraction.document.profiles import lock_paths
from experiments.benchmarks.extraction.document.protocol import load_protocol


@pytest.mark.parametrize(
    "payload",
    [
        {"reference_capabilities": ["text"], "text": None},
        {
            "reference_capabilities": ["hierarchy"],
            "elements": [{"id": "a", "text": "x", "parent_id": ""}],
        },
        {
            "reference_capabilities": ["tables"],
            "has_table": True,
            "elements": [
                {
                    "id": "a",
                    "kind": "table",
                    "text": "x",
                    "html": "<table><tr><td>x</td></tr></table>",
                    "bounding_box": [0, 0, 1],
                }
            ],
        },
    ],
)
def test_claimed_reference_values_are_not_coerced(payload):
    with pytest.raises(ValueError):
        metrics.load_reference({"kind": "image", **payload})


@pytest.mark.parametrize(
    "text,expected", [("page-only content", 1), ("", 0), ("...", 0)]
)
def test_page_only_reference_controls_unexpected_empty(text, expected):
    item = {
        "id": "page-only",
        "kind": "image",
        "reference_capabilities": ["pages"],
        "pages": [text],
    }
    document = ExtractedDocument(
        "blank",
        "blank.png",
        SourceKind.IMAGE,
        "hash",
        None,
        "",
        (),
        ExtractionProfile("test", "test", "1"),
    )
    scored = metrics.score_document(
        item, document, element_matching_threshold=0.5, duplicate_content_threshold=0.95
    )
    assert scored.metrics["reliability.unexpected_empty_output_rate"] == expected
    failed = metrics.score_document(
        item,
        None,
        failed=True,
        element_matching_threshold=0.5,
        duplicate_content_threshold=0.95,
    )
    assert failed.metrics["reliability.unexpected_empty_output_rate"] == 0


@pytest.mark.parametrize(
    "counts,exact",
    [
        ((1000, 1000, 1001), 0),
        ((3, 3, 3), 1),
        ((0, 2, 3), 0),
        ((None, 2, 2), None),
        ((True, 2, 2), None),
        ((3, 2, 2), None),
        ((1, -2, 2), None),
    ],
)
def test_cdm_exactness_uses_counts_and_keeps_similarity(monkeypatch, counts, exact):
    value = dict(
        zip(("tp", "gt_tokens", "pred_tokens"), counts, strict=True),
        similarity=1.0,
        status="evaluated",
    )
    monkeypatch.setattr(
        metrics, "score_official_metrics", lambda *_args, **_kwargs: ([], [value])
    )
    record = metrics.DocumentEvaluation(("image",), formula_pairs=[("x", "x")])
    metrics.apply_official_metrics([record], timeout_seconds=1)
    assert record.metrics["formulas.recognition_similarity"] == 1
    assert record.metrics["formulas.exact_match"] == exact
    assert record.formula_evidence[0]["tp"] == counts[0]
    assert record.formula_evidence[0]["exact_match_policy"] == "cdm_exact_counts_v1"
    if exact is None:
        assert (
            record.statuses["formulas.exact_match"]["reason"]
            == "cdm_counts_unavailable"
        )


@pytest.mark.parametrize("status", ["missing_prediction", "invalid_prediction"])
def test_cdm_reconstruction_zero_does_not_invent_success_counts(status):
    value = metrics._formula_result({"similarity": 0, "status": status})
    assert value["exact_match"] == 0
    assert "tp" not in value
    assert metrics._formula_result(1.0)["exact_match"] is None


def test_cdm_missing_counts_do_not_erase_completed_similarity():
    from experiments.benchmarks.extraction.document.omnidocbench_worker import (
        _score_formula,
    )

    renderer = SimpleNamespace(
        run_cmd=lambda *_args: 0,
        _parse_pdflatex_log=lambda _: None,
        formular_template="",
        formular_template_zh="",
        tabular_template="",
    )
    value = _score_formula(
        SimpleNamespace(evaluate=lambda *_args: {"F1_score": 0.9}),
        renderer,
        "x",
        "y",
        "fixture",
    )
    scored = metrics._formula_result(value)
    assert scored["similarity"] == 0.9
    assert scored["exact_match"] is None
    assert scored["exact_match_status"] == "unavailable"


@pytest.mark.parametrize("drift", [False, True])
def test_paddle_initialization_freezes_every_exposed_batch(
    monkeypatch, tmp_path, drift
):
    created = []
    config = yaml.safe_load(
        (
            Path(sys.executable).parent.parent
            / "Lib/site-packages/paddlex/configs/pipelines/PaddleOCR-VL-1.6.yaml"
        ).read_text(encoding="utf-8")
    )

    def runtime(**kwargs):
        created.append(kwargs)
        effective = deepcopy(kwargs["paddlex_config"])
        if drift:
            effective["SubModules"]["VLRecognition"]["batch_size"] = -1
        return SimpleNamespace(
            _merged_paddlex_config=effective,
            predict=lambda _: iter(
                [
                    SimpleNamespace(json={"parsing_res_list": []}),
                    SimpleNamespace(json={"parsing_res_list": []}),
                ]
            ),
        )

    for name, value in {
        "torch": SimpleNamespace(__version__="mock"),
        "paddle": SimpleNamespace(
            __version__="3.3.1",
            is_compiled_with_cuda=lambda: True,
            set_device=lambda _: None,
        ),
        "paddleocr": SimpleNamespace(PaddleOCRVL=runtime),
        "paddlex.inference": SimpleNamespace(
            load_pipeline_config=lambda _: deepcopy(config)
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(adapters, "version", lambda _: "3.7.0")
    options = {
        **load_protocol().parser_options("paddleocr-vl-1.6"),
        "model_path": str(tmp_path),
        "paddle_cache_path": str(tmp_path),
    }
    profile = ExtractionProfile(
        "test", "paddleocr-vl-1.6", "revision", device="cuda", options=options
    )
    request = ExtractionRequest(
        tmp_path / "multipage.pdf", "hash", profile=profile, options=options
    )
    extractor = adapters.ExperimentalDocumentExtractor("paddleocr-vl-1.6", "revision")
    if drift:
        with pytest.raises(RuntimeError, match="batches differ"):
            extractor._paddle_runtime(request)
        assert not extractor._runtimes
    else:
        extractor._paddle_runtime(request)
        assert extractor._paddle_vl(request) == ([], ())
        initialized = next(iter(worker.pipeline_parameters(extractor).values()))
        assert set(adapters._paddle_batches(initialized).values()) == {1}
        assert len(adapters._paddle_batches(initialized)) == 5
    assert created[0]["paddlex_config"]["batch_size"] == 1


def test_tesseract_options_bind_and_verify_prepared_files(monkeypatch, tmp_path):
    from edumind.extraction.extractors import document

    executable, english = tmp_path / "tesseract.exe", tmp_path / "eng.traineddata"
    executable.write_bytes(b"verified executable")
    english.write_bytes(b"verified English data")
    prepared = {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "english_data_path": str(english),
        "english_data_sha256": sha256_file(english),
    }
    options = {
        "model_path": str(tmp_path),
        "ocr_engine": "tesseract",
        **lock_paths({"system_components": {"tesseract-cli": prepared}}),
    }
    profile = ExtractionProfile(
        "test", "docling-standard", "revision", device="cpu", options=options
    )
    request = ExtractionRequest(
        tmp_path / "page.png", "hash", profile=profile, options=options
    )

    class Options(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.table_structure_options = SimpleNamespace()

    fake = {
        "docling.datamodel.accelerator_options": SimpleNamespace(
            AcceleratorOptions=Options
        ),
        "docling.datamodel.base_models": SimpleNamespace(
            InputFormat=SimpleNamespace(IMAGE="image", PDF="pdf")
        ),
        "docling.datamodel.pipeline_options": SimpleNamespace(
            EasyOcrOptions=Options,
            RapidOcrOptions=Options,
            TesseractCliOcrOptions=Options,
            PdfPipelineOptions=Options,
            OcrMode=SimpleNamespace(
                FULL_PAGE="full", PDF_AWARE_LAYOUT_REGIONS="regions"
            ),
            TableFormerMode=SimpleNamespace(ACCURATE="accurate", FAST="fast"),
        ),
        "docling.document_converter": SimpleNamespace(
            DocumentConverter=Options,
            ImageFormatOption=Options,
            PdfFormatOption=Options,
        ),
    }
    for name, value in fake.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(document, "version", lambda _: document.DOCLING_VERSION)
    monkeypatch.setenv("PATH", "untrusted alternate installation")
    monkeypatch.setenv("TESSDATA_PREFIX", "untrusted language data")
    extractor = document.DoclingExtractor("revision")
    runtime, _ = extractor._converter(request, SourceKind.IMAGE)
    effective = runtime.format_options["image"].pipeline_options.ocr_options
    assert effective.tesseract_cmd == str(executable.resolve())
    assert effective.path == str(english.parent.resolve())
    english.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="missing or changed"):
        extractor._converter(request, SourceKind.IMAGE)


def test_official_wrapper_normalizes_nonempty_pairs_symmetrically(
    tmp_path, monkeypatch
):
    import importlib.util

    from experiments.benchmarks.extraction.document import (
        omnidocbench_worker as official,
    )

    root = (
        Path(__file__).resolve().parents[1] / "data/benchmarks/evaluators/OmniDocBench"
    )
    if not (root / "src/core/preprocess/data_preprocess.py").is_file():
        pytest.skip("Prepared pinned normalizer unavailable; no download performed")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "src"))
    # Execute the full prepared normalizer, not a project rewrite or a stub.
    spec = importlib.util.spec_from_file_location(
        "prepared_normalizer", root / "src/core/preprocess/data_preprocess.py"
    )
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    monkeypatch.setitem(sys.modules, "src.core.preprocess.data_preprocess", normalizer)
    calls = []

    class TEDS:
        def __init__(self, structure_only):
            self.structure_only = structure_only

        def evaluate(self, prediction, reference):
            calls.append((prediction, reference))
            assert "<th" not in prediction and "<th" not in reference
            assert 'colspan="2"' in prediction and 'colspan="2"' in reference
            return 1.0

    monkeypatch.setitem(sys.modules, "metrics.table_metric", SimpleNamespace(TEDS=TEDS))
    reference = '<table><thead><tr><th colspan="2">Header</th></tr></thead></table>'
    prediction = '<table><tr><td colspan="2"><span>Header</span></td></tr></table>'
    source, dest = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text(
        json.dumps({"tables": [{"reference": reference, "prediction": prediction}]}),
        encoding="utf-8",
    )
    official.main(source, dest)
    result = json.loads(dest.read_text())
    assert result["tables"] == [[1, 1]]
    assert len(calls) == 3
    evidence = result["table_evidence"][0]
    assert evidence["reference"] == reference and evidence["prediction"] == prediction
    assert evidence["normalized_reference"] == evidence["normalized_prediction"]
    assert result["provenance"]["exact_match_policy"] == "cdm_exact_counts_v1"
