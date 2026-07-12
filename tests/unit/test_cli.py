from __future__ import annotations

import sys

import pytest

import edumind.cli as cli


def test_cli_dispatches_supported_ui_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["edumind", "ui"])
    monkeypatch.setattr(cli, "run_ui", lambda: 7)

    assert cli.main() == 7


def test_cli_rejects_removed_legacy_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["edumind", "ui-microservices"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
