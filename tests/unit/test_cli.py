from __future__ import annotations

import edumind.cli as cli


def test_cli_forwards_all_benchmark_arguments(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(cli, "run_benchmark", lambda arguments: captured.extend(arguments) or 7)
    assert cli.main(["benchmark", "--profile", "smoke", "rag", "retrieval"]) == 7
    assert captured == ["--profile", "smoke", "rag", "retrieval"]


def test_cli_uses_new_extraction_name_only() -> None:
    choices = cli.build_parser()._subparsers._group_actions[0].choices
    assert "extraction-api" in choices
    assert "ocr-api" not in choices
