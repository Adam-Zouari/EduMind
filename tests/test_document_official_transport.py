"""Mocked Docker acceptance, not live official-scorer qualification."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.extraction import ExtractionProfile, SourceKind
from edumind.extraction.contracts import ExtractionRequest
from edumind.extraction.structured import build_structured_document
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from experiments.benchmarks.common.runner import _write_candidate_artifact
from experiments.benchmarks.extraction.document import (
    metrics,
    runner,
)
from experiments.benchmarks.extraction.document import (
    official_metrics as official,
)
from experiments.benchmarks.extraction.document import (
    omnidocbench_worker as worker,
)
from experiments.benchmarks.extraction.document.protocol import load_protocol
from experiments.benchmarks.preparation import evaluators

HTML = '<table><tr><th colspan="2">HeaderAlpha</th></tr></table>'
NEAR = {
    "similarity": 1.0,
    "status": "evaluated",
    "tp": 1000,
    "gt_tokens": 1000,
    "pred_tokens": 1001,
}
EXACT = {
    "similarity": 1.0,
    "status": "evaluated",
    "tp": 1,
    "gt_tokens": 1,
    "pred_tokens": 1,
}


@pytest.fixture
def docker_transport(tmp_path, monkeypatch):
    # Test-only prepared inventory. No source download, image pull or real Docker.
    source = tmp_path / "data/benchmarks/evaluators/OmniDocBench"
    source.mkdir(parents=True)
    (source / "fixture.py").write_text("# mocked evaluator fixture\n", encoding="utf-8")
    (source / ".edumind-revision").write_text(evaluators.OMNIDOCBENCH_REVISION)
    files = evaluators._source_inventory(source)
    atomic_write_json(
        source / evaluators.OMNIDOCBENCH_SOURCE_MANIFEST,
        {
            "schema_version": 1,
            "source_revision": evaluators.OMNIDOCBENCH_REVISION,
            "files": files,
            "source_manifest_sha256": stable_hash(files),
        },
    )
    atomic_write_json(
        source / evaluators.OMNIDOCBENCH_IMAGE_LOCK,
        {
            "tag": evaluators.OMNIDOCBENCH_IMAGE,
            "source_revision": evaluators.OMNIDOCBENCH_REVISION,
            "digest": "unit-test/image@sha256:" + "a" * 64,
        },
    )
    state = SimpleNamespace(
        source=source,
        commands=[],
        formulas=None,
        tables=None,
        mutate=lambda _: None,
        native=False,
        after=lambda: None,
    )
    monkeypatch.setattr(official, "PROJECT_ROOT", tmp_path)
    original_run = subprocess.run

    def run(command, **kwargs):
        if command[:2] in (
            ["git", "rev-parse"],
            ["git", "status"],
            ["git", "diff"],
            ["git", "ls-files"],
        ):
            return original_run(command, **kwargs)
        state.commands.append(command)
        assert command[:2] == ["docker", "run"]
        assert command[command.index("--network") + 1] == "none"
        assert "unit-test/image@sha256:" + "a" * 64 in command
        mount = next(
            value
            for value in command
            if value.startswith("type=bind,") and value.endswith("target=/work")
        )
        work = Path(mount.split("source=", 1)[1].rsplit(",target=", 1)[0])
        inputs = json.loads((work / "input.json").read_text(encoding="utf-8"))
        if state.native:
            worker.main(work / "input.json", work / "output.json")
            result = json.loads((work / "output.json").read_text(encoding="utf-8"))
        else:
            result = {
                "schema_version": worker.SCHEMA_VERSION,
                "tables": state.tables
                if state.tables is not None
                else [[1.0, 1.0] for _ in inputs["tables"]],
                "formulas": deepcopy(
                    state.formulas
                    if state.formulas is not None
                    else [NEAR for _ in inputs["formulas"]]
                ),
                "table_evidence": [
                    {
                        **pair,
                        "normalized_reference": "<html><body>"
                        + pair["reference"]
                        + "</body></html>",
                        "normalized_prediction": "<html><body>"
                        + pair["prediction"]
                        + "</body></html>",
                    }
                    for pair in inputs["tables"]
                ],
                "provenance": {
                    "normalization": worker.NORMALIZATION,
                    "exact_match_policy": worker.EXACT_MATCH_POLICY,
                    "wrapper_sha256": sha256_file(Path(worker.__file__)),
                },
            }
        state.mutate(result)
        (work / "output.json").write_text(json.dumps(result), encoding="utf-8")
        state.after()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(official.subprocess, "run", run)
    return state


def test_typed_transport_retains_counts_normalization_and_verified_identity(
    docker_transport,
):
    state = docker_transport
    missing = {
        "similarity": 0.0,
        "status": "missing_prediction",
        "tp": None,
        "gt_tokens": None,
        "pred_tokens": None,
    }
    state.formulas = [NEAR, EXACT, missing]
    evidence = {}
    tables, formulas = official.score_official_metrics(
        [(HTML, HTML)],
        [("x", "near"), ("x", "x"), ("x", "")],
        timeout_seconds=1,
        evidence=evidence,
    )
    assert tables == [(1, 1)]
    assert formulas == [NEAR, EXACT, missing]
    identity = evaluators.evaluator_identity(state.source)
    assert {key: evidence["provenance"][key] for key in identity} == identity
    assert evidence["provenance"]["wrapper_sha256"] == sha256_file(
        Path(worker.__file__)
    )
    assert evidence["table_evidence"][0]["reference"] == HTML
    assert len(evidence["table_evidence"][0]["normalized_reference_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("schema_version"),
        lambda p: p.update(schema_version=True),
        lambda p: p.update(schema_version=999),
        lambda p: p.update(unknown=True),
        lambda p: p.update(formulas=[1.0]),
        lambda p: p["formulas"][0].update(status="evaluator_failed"),
        lambda p: p["formulas"][0].update(similarity=True),
        lambda p: p["formulas"][0].update(similarity=float("nan")),
        lambda p: p["formulas"][0].update(similarity="1.0"),
        lambda p: p.update(tables=[]),
        lambda p: p.update(tables=[[True, 1.0]]),
        lambda p: p.pop("table_evidence"),
        lambda p: p["table_evidence"][0].pop("normalized_reference"),
        lambda p: p["table_evidence"][0].update(reference="different input"),
        lambda p: p["provenance"].update(normalization="envelope-only"),
        lambda p: p["provenance"].update(exact_match_policy="rounded-f1"),
        lambda p: p["provenance"].update(wrapper_sha256="0" * 64),
    ],
)
def test_transport_rejects_schema_score_and_provenance_drift(
    docker_transport, mutation
):
    docker_transport.mutate = mutation
    evidence = {}
    with pytest.raises(RuntimeError, match="malformed results"):
        official.score_official_metrics(
            [(HTML, HTML)], [("x", "x")], timeout_seconds=1, evidence=evidence
        )
    assert evidence["provenance"]["source_manifest_sha256"]


def test_transport_rechecks_source_after_scoring(docker_transport):
    state = docker_transport
    state.after = lambda: (state.source / "fixture.py").write_text(
        "# changed during mocked execution\n"
    )
    with pytest.raises(RuntimeError, match="source integrity"):
        official.score_official_metrics([(HTML, HTML)], [], timeout_seconds=1)


def test_nonempty_worker_runs_pinned_normalizer_through_mocked_docker(
    docker_transport, monkeypatch, tmp_path
):
    state = docker_transport
    root = (
        Path(__file__).resolve().parents[1] / "data/benchmarks/evaluators/OmniDocBench"
    )
    if not (root / "src/core/preprocess/data_preprocess.py").is_file():
        pytest.skip("Prepared pinned normalizer unavailable; no download performed")
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "src"))
    spec = importlib.util.spec_from_file_location(
        "prepared_transport_normalizer", root / "src/core/preprocess/data_preprocess.py"
    )
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    monkeypatch.setitem(sys.modules, "src.core.preprocess.data_preprocess", normalizer)

    class TEDS:
        def __init__(self, structure_only):
            self.structure_only = structure_only

        def evaluate(self, prediction, reference):
            assert "<th" not in prediction and "<th" not in reference
            return 1.0

    renderer = SimpleNamespace(
        run_cmd=lambda *_args: 0,
        _parse_pdflatex_log=lambda _: None,
        formular_template="",
        formular_template_zh="",
        tabular_template="",
    )
    for name, value in {
        "metrics.table_metric": SimpleNamespace(TEDS=TEDS),
        "metrics.cdm.modules": SimpleNamespace(latex2bbox_color=renderer),
        "metrics.cdm_metric": SimpleNamespace(
            CDM=lambda **_kwargs: SimpleNamespace(
                evaluate=lambda *_args: {
                    "F1_score": 1.0,
                    "tp": 1000,
                    "gt_tokens": 1000,
                    "pred_tokens": 1001,
                }
            )
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(
        worker,
        "Path",
        lambda value: tmp_path / "cdm" if str(value) == "/work/cdm" else Path(value),
    )
    state.native = True
    reference = (
        '<table><thead><tr><th colspan="2">HeaderAlpha</th></tr></thead></table>'
    )
    prediction = '<table><tr><td colspan="2"><span>HeaderAlpha</span></td></tr></table>'
    record = metrics.DocumentEvaluation(
        ("image",), table_pairs=[(reference, prediction)], formula_pairs=[("x", "x")]
    )
    metrics.apply_official_metrics([record], timeout_seconds=1)
    assert record.metrics["tables.teds"] == 1
    assert record.metrics["formulas.recognition_similarity"] == 1
    assert record.metrics["formulas.exact_match"] == 0
    assert (
        record.table_evidence[0]["normalized_reference"]
        == record.table_evidence[0]["normalized_prediction"]
    )
    assert record.official_provenance["source_manifest_sha256"]


def _candidate(monkeypatch, tmp_path):
    profile = ExtractionProfile("test", "docling-standard", "fixture")
    request = ExtractionRequest(
        tmp_path / "sample.docx", "fixture-asset", profile=profile
    )
    document = build_structured_document(
        request,
        SourceKind.DOCX,
        profile,
        [
            {
                "kind": "table",
                "text": "HeaderAlpha",
                "structured_content": {"rows": [["HeaderAlpha"]], "html": HTML},
            },
            {"kind": "formula", "text": "x", "structured_content": {"latex": "x"}},
        ],
    )
    payload = {
        "attempts": [
            {
                "sample_id": "sample",
                "repetition": 1,
                "executed": True,
                "success": True,
                "error": None,
                "document": document.to_dict(),
                "output_fingerprint": metrics._document_fingerprint(document),
                "latency_seconds": 0.1,
            }
        ],
        "operational": {"peak_process_tree_ram_mb": 1, "peak_vram_mb": None},
        "measured_batch_seconds": 0.1,
        "parameters": {},
        "resource_samples": [],
    }
    monkeypatch.setattr(
        runner, "run_json_worker", lambda *_args, **_kwargs: deepcopy(payload)
    )
    item = {
        "id": "sample",
        "kind": "docx",
        "asset_sha256": "fixture-asset",
        "reference_capabilities": ["tables", "formulas"],
        "has_table": True,
        "has_formula": True,
        "elements": [
            {"id": "table", "kind": "table", "text": "HeaderAlpha", "html": HTML},
            {"id": "formula", "kind": "formula", "text": "x", "latex": "x"},
        ],
    }

    def evaluate():
        return runner.evaluate_candidate(
            "docling-standard-native",
            [item],
            BenchmarkPlan(
                "extraction", "document", "smoke", "fixture", ("parser",), 42, 1, 0, 1
            ),
            {},
            {"device": "cpu"},
            {"sample": metrics.load_reference(item)},
            load_protocol(),
        )

    return evaluate


@pytest.mark.parametrize(
    "counts",
    [
        NEAR,
        EXACT,
        {"similarity": 1.0, "status": "evaluated"},
        {**NEAR, "tp": True},
        {**NEAR, "tp": "1000"},
    ],
)
def test_runner_persists_typed_counts_and_normalized_source_evidence(
    docker_transport, monkeypatch, tmp_path, counts
):
    docker_transport.formulas = [counts]
    evaluate = _candidate(monkeypatch, tmp_path)
    if type(counts.get("tp")) is int:
        result = evaluate()
        assert result[0][0].metrics["formulas.exact_match"] == float(
            counts["tp"] == counts["gt_tokens"] == counts["pred_tokens"] > 0
        )
        artifacts = result[-1]
    else:
        with pytest.raises(
            CandidateExecutionError, match="scoring incomplete"
        ) as failure:
            evaluate()
        assert failure.value.samples[0].metrics["formulas.recognition_similarity"] == 1
        assert failure.value.samples[0].metrics["formulas.exact_match"] is None
        artifacts = failure.value.artifacts
    path = _write_candidate_artifact(
        tmp_path, "parser", "official_evaluations", artifacts["official_evaluations"]
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path.suffix == ".json"
    assert saved["provenance"]["source_manifest_sha256"]
    assert saved["provenance"]["image_digest"].endswith("a" * 64)
    assert saved["documents"][0]["table_evidence"][0]["reference"] == HTML
    assert saved["documents"][0]["formula_evidence"][0].get("tp") == counts.get("tp")


@pytest.mark.parametrize(
    "scores,pass_gate",
    [([(1, 1), (1, 1), (0.6, 1), (0.5, 0.5)], True), ([(1, 1)] * 4, False)],
)
def test_runtime_conformance_checks_header_perturbations(
    docker_transport, scores, pass_gate
):
    docker_transport.tables = scores
    docker_transport.formulas = [EXACT]
    if pass_gate:
        official.validate_official_runtime(
            tables=True, formulas=True, timeout_seconds=1
        )
    else:
        with pytest.raises(RuntimeError, match="header perturbation"):
            official.validate_official_runtime(
                tables=True, formulas=True, timeout_seconds=1
            )


def test_infrastructure_failure_retains_source_identity_and_predictions(
    docker_transport, monkeypatch, tmp_path
):
    def failed():
        raise subprocess.CalledProcessError(
            1, ["docker", "run"], stderr="renderer infrastructure failed"
        )

    docker_transport.after = failed
    evaluate = _candidate(monkeypatch, tmp_path)
    with pytest.raises(CandidateExecutionError, match="scoring incomplete") as caught:
        evaluate()
    assert caught.value.samples[0].metrics["formulas.recognition_similarity"] is None
    evidence = caught.value.artifacts["official_evaluations"]
    assert evidence["provenance"]["source_manifest_sha256"]
    assert caught.value.artifacts["outputs"]["attempts"][0]["document"]["text"]
    assert (
        "renderer infrastructure failed"
        in caught.value.artifacts["evaluator_error"]["error"]
    )


def test_mocked_transport_evidence_is_uploaded_to_mlflow(
    docker_transport, monkeypatch, tmp_path, request
):
    from experiments.benchmarks.common.runner import run_benchmark

    mlflow = pytest.importorskip("mlflow")
    previous = mlflow.get_tracking_uri()
    request.addfinalizer(lambda: mlflow.set_tracking_uri(previous))
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    client.create_experiment(
        "EduMind / official-transport-regression",
        artifact_location=(tmp_path / "mlruns").as_uri(),
    )
    evaluate = _candidate(monkeypatch, tmp_path)
    result = run_benchmark(
        BenchmarkPlan(
            "extraction",
            "official-transport-regression",
            "smoke",
            "fixture",
            ("parser",),
            42,
            1,
            0,
            1,
        ),
        lambda _: evaluate(),
        dataset_checksum="mocked-docker-fixture",
        directions={"tables.teds": "max"},
        primary_metric="tables.teds",
        monitor_resources=False,
        paired_comparisons=False,
        artifact_root=tmp_path / "artifacts",
    )
    candidate = result.candidates[0]
    assert candidate.status == "success"
    uploaded = Path(
        client.download_artifacts(candidate.mlflow_run_id, "official_evaluations.json")
    )
    saved = json.loads(uploaded.read_text(encoding="utf-8"))
    assert (
        saved["provenance"]["source_manifest_sha256"]
        == evaluators.evaluator_identity(docker_transport.source)[
            "source_manifest_sha256"
        ]
    )
    assert saved["provenance"]["wrapper_sha256"] == sha256_file(Path(worker.__file__))
    document = saved["documents"][0]
    assert document["formula_evidence"][0]["tp"] == 1000
    assert document["formula_evidence"][0]["exact_match"] == 0
    assert document["table_evidence"][0]["normalized_reference"]
