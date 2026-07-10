from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest


def test_editable_install_imports_from_outside_repo_root(tmp_path: Path) -> None:
    try:
        importlib.metadata.version("edumind-ai")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("edumind-ai is not installed; run `pip install -e .[dev,ocr,api]` first.")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from edumind.ocr import DataIngestionPipeline; print(DataIngestionPipeline.__name__)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DataIngestionPipeline" in result.stdout
