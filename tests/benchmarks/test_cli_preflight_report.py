from __future__ import annotations

import json
from pathlib import Path

from edumind.benchmarks import cli, preflight
from edumind.benchmarks.contracts import BenchmarkPlan, BenchmarkResult, CandidateResult
from edumind.benchmarks.report import render_report


def _result(tmp_path: Path, stage: str) -> BenchmarkResult:
    plan = BenchmarkPlan("test", stage, "smoke", "fixture", ("candidate",))
    candidate = CandidateResult(
        "candidate",
        "success",
        "fingerprint",
        {"quality": 1.0},
        {"quality": {"lower": 0.9, "upper": 1.0}},
        (),
        {"p95_latency_seconds": 0.1},
    )
    return BenchmarkResult(
        "run",
        plan,
        {},
        (candidate,),
        ("candidate",),
        False,
        tmp_path,
    )


def test_benchmark_cli_runs_every_smoke_stage(monkeypatch, tmp_path, capsys) -> None:
    stages: list[str] = []

    def extraction(stage: str, profile: str) -> BenchmarkResult:
        assert profile == "smoke"
        stages.append(f"extraction:{stage}")
        return _result(tmp_path, stage)

    def rag(stage: str):
        def run(profile: str) -> BenchmarkResult:
            assert profile == "smoke"
            stages.append(f"rag:{stage}")
            return _result(tmp_path, stage)

        return run

    monkeypatch.setattr(cli, "run_extraction_stage", extraction)
    monkeypatch.setattr(cli, "run_chunking_embedding", rag("chunking-embedding"))
    monkeypatch.setattr(cli, "run_retrieval", rag("retrieval"))
    monkeypatch.setattr(cli, "run_generation", rag("generation"))
    monkeypatch.setattr(cli, "run_final", rag("final"))
    monkeypatch.setattr(
        cli,
        "run_vectordb",
        lambda profile: stages.append("systems:vectordb") or _result(tmp_path, "vectordb"),
    )
    assert cli.main(["all"]) == 0
    assert len(stages) == 12
    assert json.loads(capsys.readouterr().out)[0]["authoritative"] is False


def test_preflight_standard_checks_all_local_capabilities(monkeypatch) -> None:
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"C:/{name}.exe")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": []}

    monkeypatch.setattr(preflight.requests, "get", lambda *args, **kwargs: Response())
    checks = preflight.run_preflight("standard")
    names = {check.name for check in checks}
    assert {"tesseract", "ffmpeg", "ollama", "chromadb", "qdrant_client", "lancedb"} <= names
    assert preflight.preflight_payload(checks)["checks"]


def test_preflight_rejects_unknown_profile_and_handles_ollama_failure(monkeypatch) -> None:
    try:
        preflight.run_preflight("invalid")
    except ValueError as exc:
        assert "Unknown benchmark profile" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid profile was accepted")
    monkeypatch.setattr(
        preflight.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(preflight.requests.ConnectionError()),
    )
    assert preflight._ollama().ready is False


def test_report_and_cli_report_render_candidate_details(tmp_path, capsys) -> None:
    summary = {
        "run_id": "run-1",
        "plan": {"suite": "rag", "stage": "retrieval", "profile": "smoke", "dataset": "x"},
        "authoritative": False,
        "pareto_candidates": ["dense"],
        "candidates": [
            {
                "candidate": "dense",
                "status": "success",
                "metrics": {"ndcg_at_5": 0.75},
                "intervals": {"ndcg_at_5": {"lower": 0.7, "upper": 0.8}},
                "operational": {"p95_latency_seconds": 0.2},
            }
        ],
    }
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    report = render_report(source)
    assert "nDCG" not in report.read_text(encoding="utf-8")
    assert "ndcg_at_5: 0.750000" in report.read_text(encoding="utf-8")
    output = tmp_path / "from-cli.md"
    assert cli.main(["report", str(source), "--output", str(output)]) == 0
    assert str(output) in capsys.readouterr().out
