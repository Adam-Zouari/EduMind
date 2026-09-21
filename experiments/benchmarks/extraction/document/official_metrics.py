"""Run the pinned OmniDocBench table/formula scorers in their Docker runtime."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from edumind.common.artifacts import atomic_write_json
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.preparation.evaluators import (
    OMNIDOCBENCH_IMAGE_LOCK,
    OMNIDOCBENCH_REVISION,
)


def score_official_metrics(
    table_pairs: Sequence[tuple[str, str]],
    formula_pairs: Sequence[tuple[str, str]],
    *,
    timeout_seconds: int,
) -> tuple[list[tuple[float, float]], list[float]]:
    if not table_pairs and not formula_pairs:
        return [], []
    source = PROJECT_ROOT / "data/benchmarks/evaluators/OmniDocBench"
    revision_file = source / ".edumind-revision"
    image_lock = source / OMNIDOCBENCH_IMAGE_LOCK
    if (
        not revision_file.is_file()
        or revision_file.read_text(encoding="utf-8").strip() != OMNIDOCBENCH_REVISION
        or not image_lock.is_file()
    ):
        raise RuntimeError(
            "The pinned OmniDocBench source or Docker image is missing; run "
            "python experiments/benchmarks/prepare.py evaluators"
        )
    image = official_image_digest()
    worker = Path(__file__).with_name("omnidocbench_worker.py").resolve()
    temporary_root = PROJECT_ROOT / "artifacts/benchmarks/.omnidocbench"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        work = Path(directory).resolve()
        atomic_write_json(
            work / "input.json",
            {
                "tables": [
                    {"reference": reference, "prediction": prediction}
                    for reference, prediction in table_pairs
                ],
                "formulas": [
                    {"reference": reference, "prediction": prediction}
                    for reference, prediction in formula_pairs
                ],
            },
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--workdir",
            "/opt/omnidocbench",
            "--mount",
            _mount(source.resolve(), "/opt/omnidocbench", readonly=True),
            "--mount",
            _mount(worker, "/opt/edumind/worker.py", readonly=True),
            "--mount",
            _mount(work, "/work"),
            "--entrypoint",
            "python",
            image,
            "/opt/edumind/worker.py",
            "/work/input.json",
            "/work/output.json",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker is required for official OmniDocBench scoring") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no container output").strip()
            raise RuntimeError(f"OmniDocBench scorer failed: {detail[-3000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"OmniDocBench scorer exceeded its {timeout_seconds:g}-second timeout"
            ) from exc
        output = work / "output.json"
        if not output.is_file():
            raise RuntimeError(
                "OmniDocBench scorer produced no result: "
                + (completed.stderr or completed.stdout or "no container output")[-1000:]
            )
        result = json.loads(output.read_text(encoding="utf-8"))
    tables = [tuple(map(float, values)) for values in result.get("tables", [])]
    formulas = [float(value) for value in result.get("formulas", [])]
    if len(tables) != len(table_pairs) or len(formulas) != len(formula_pairs):
        raise RuntimeError("OmniDocBench scorer returned an incomplete result")
    return tables, formulas


def official_image_digest() -> str:
    source = PROJECT_ROOT / "data/benchmarks/evaluators/OmniDocBench"
    image_lock = source / OMNIDOCBENCH_IMAGE_LOCK
    try:
        payload = json.loads(image_lock.read_text(encoding="utf-8"))
        digest = str(payload["digest"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "The OmniDocBench Docker image lock is missing or invalid; run "
            "python experiments/benchmarks/prepare.py evaluators"
        ) from exc
    if "@sha256:" not in digest:
        raise RuntimeError("The OmniDocBench Docker image lock has no immutable digest")
    if payload.get("source_revision") != OMNIDOCBENCH_REVISION:
        raise RuntimeError("The OmniDocBench image lock targets a different source revision")
    return digest


def validate_official_runtime(
    *, tables: bool, formulas: bool, timeout_seconds: int
) -> None:
    html = "<table><tr><td>x</td></tr></table>"
    table_pairs = [(html, html)] if tables else []
    formula_pairs = [("x", "x")] if formulas else []
    table_scores, formula_scores = score_official_metrics(
        table_pairs, formula_pairs, timeout_seconds=timeout_seconds
    )
    if table_scores and table_scores[0] != (1.0, 1.0):
        raise RuntimeError("Official TEDS evaluator failed its identity preflight")
    if formula_scores and formula_scores[0] != 1.0:
        raise RuntimeError("Official CDM evaluator failed its identity preflight")


def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
    value = f"type=bind,source={source.as_posix()},target={target}"
    return value + (",readonly" if readonly else "")
