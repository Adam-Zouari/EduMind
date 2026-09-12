"""Fresh-process helpers shared by benchmark workers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from edumind.common.artifacts import atomic_write_json
from edumind.common.paths import PROJECT_ROOT


def worker_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NVIDIA_VISIBLE_DEVICES"] = "none"
    return environment


def run_json_worker(
    script: Path,
    payload: Mapping[str, object],
    *,
    device: str,
    prefix: str,
    error_label: str,
    temporary_root: Path | None = None,
) -> dict[str, object]:
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=temporary_root) as raw:
        directory = Path(raw)
        input_path = directory / "input.json"
        output_path = directory / "output.json"
        atomic_write_json(input_path, dict(payload))
        completed = subprocess.run(
            [sys.executable, str(script), str(input_path), str(output_path)],
            cwd=PROJECT_ROOT,
            env=worker_environment(device),
            capture_output=True,
            text=True,
        )
        if completed.returncode or not output_path.is_file():
            detail = (
                completed.stderr or completed.stdout or "no worker output"
            ).strip()
            raise RuntimeError(f"{error_label} failed: {detail[-4000:]}")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError(f"{error_label} returned a non-object result")
        return result


def json_worker_main(
    execute: Callable[[dict[str, object]], dict[str, object]],
) -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PAYLOAD_JSON RESULT_JSON")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Worker payload must be a JSON object")
    atomic_write_json(Path(sys.argv[2]), execute(payload))
    return 0
