"""Run the pinned OmniDocBench table/formula scorers in their Docker runtime."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from edumind.common.artifacts import atomic_write_json, sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.preparation.evaluators import (
    OMNIDOCBENCH_IMAGE_LOCK,
    OMNIDOCBENCH_REVISION,
    evaluator_identity,
)

from .omnidocbench_worker import EXACT_MATCH_POLICY, NORMALIZATION, SCHEMA_VERSION


def score_official_metrics(
    table_pairs: Sequence[tuple[str, str]],
    formula_pairs: Sequence[tuple[str, str]],
    *,
    timeout_seconds: int,
    evidence: dict[str, object] | None = None,
) -> tuple[list[tuple[float, float]], list[dict[str, object]]]:
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
            "python -m experiments.benchmarks.prepare evaluators"
        )
    identity = official_evaluator_identity()
    image = identity["image_digest"]
    worker = Path(__file__).with_name("omnidocbench_worker.py").resolve()
    wrapper_checksum = sha256_file(worker)
    provenance = {
        **identity,
        "normalization": NORMALIZATION,
        "exact_match_policy": EXACT_MATCH_POLICY,
        "wrapper_sha256": wrapper_checksum,
        "transport_schema_version": SCHEMA_VERSION,
    }
    if (
        NORMALIZATION
        != "omnidocbench.normalized_html_table@" + identity["source_revision"]
    ):
        raise RuntimeError("OmniDocBench normalizer/source revision mismatch")
    if evidence is not None:
        evidence.update(schema_version=SCHEMA_VERSION, provenance=provenance)
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
        container_name = f"edumind-scorer-{uuid4().hex}"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
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
            raise RuntimeError(
                "Docker is required for official OmniDocBench scoring"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no container output").strip()
            raise RuntimeError(f"OmniDocBench scorer failed: {detail[-3000:]}") from exc
        except subprocess.TimeoutExpired as exc:
            try:
                subprocess.run(
                    ["docker", "rm", "--force", container_name],
                    check=False,
                    capture_output=True,
                    timeout=timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass  # Preserve the scorer timeout if Docker cleanup also fails.
            raise RuntimeError(
                f"OmniDocBench scorer exceeded its {timeout_seconds:g}-second timeout"
            ) from exc
        output = work / "output.json"
        if not output.is_file():
            raise RuntimeError(
                "OmniDocBench scorer produced no result: "
                + (completed.stderr or completed.stdout or "no container output")[
                    -1000:
                ]
            )
        try:
            result = json.loads(
                output.read_text(encoding="utf-8"), parse_constant=_invalid_constant
            )
            tables, formulas, normalized = _validated_results(
                result, table_pairs, formula_pairs, wrapper_checksum
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "OmniDocBench scorer returned malformed results"
            ) from exc
        if (
            official_evaluator_identity() != identity
            or sha256_file(worker) != wrapper_checksum
        ):
            raise RuntimeError(
                "OmniDocBench source/image/wrapper changed during scoring"
            )
        if evidence is not None:
            evidence["table_evidence"] = normalized
    return tables, formulas


def _score(value):
    if type(value) not in (int, float):
        raise ValueError(
            "Official scores must be numbers, not booleans/coerced strings"
        )
    try:
        valid = math.isfinite(value) and 0 <= value <= 1
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("Official scores must be finite and in [0,1]")
    return float(value)


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant in official worker output: {value}")


def _validated_results(result, table_pairs, formula_pairs, wrapper_checksum):
    if (
        not isinstance(result, dict)
        or set(result)
        != {"schema_version", "tables", "formulas", "table_evidence", "provenance"}
        or type(result["schema_version"]) is not int
        or result["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported official worker result schema")
    if result["provenance"] != {
        "normalization": NORMALIZATION,
        "exact_match_policy": EXACT_MATCH_POLICY,
        "wrapper_sha256": wrapper_checksum,
    }:
        raise ValueError(
            "Official worker normalization/policy/wrapper identity mismatch"
        )
    if any(
        not isinstance(result[name], list) or len(result[name]) != len(pairs)
        for name, pairs in (
            ("tables", table_pairs),
            ("table_evidence", table_pairs),
            ("formulas", formula_pairs),
        )
    ):
        raise ValueError("Incomplete official worker result inventory")
    tables = []
    for pair in result["tables"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Malformed official table score pair")
        tables.append(tuple(_score(value) for value in pair))
    formulas = []
    for value, (_, prediction) in zip(result["formulas"], formula_pairs, strict=True):
        if (
            not isinstance(value, dict)
            or not {"similarity", "status"} <= value.keys()
            or set(value) - {"similarity", "status", "tp", "gt_tokens", "pred_tokens"}
            or value["status"]
            not in {"evaluated", "missing_prediction", "invalid_prediction"}
        ):
            raise ValueError("Malformed official formula result")
        similarity = _score(value["similarity"])
        if value["status"] != "evaluated" and similarity != 0:
            raise ValueError(
                "Missing/invalid prediction cannot have nonzero similarity"
            )
        if value["status"] == "missing_prediction":
            if prediction.strip() or any(
                value.get(name) is not None
                for name in ("tp", "gt_tokens", "pred_tokens")
            ):
                raise ValueError(
                    "Missing prediction status/counts contradict the supplied expression"
                )
        elif not prediction.strip():
            raise ValueError(
                "Blank predicted expressions require missing_prediction status"
            )
        if value["status"] == "invalid_prediction" and (
            value.get("tp") is not None
            or type(value.get("gt_tokens")) is not int
            or value["gt_tokens"] <= 0
            or type(value.get("pred_tokens")) is not int
            or value["pred_tokens"] != 0
        ):
            raise ValueError(
                "Invalid prediction status cannot invent successful evaluation counts"
            )
        # Counts are retained exactly, including missing/malformed evidence. The
        # strict exactness consumer validates them independently of similarity.
        formulas.append({**value, "similarity": similarity})
    normalized = []
    fields = {
        "reference",
        "prediction",
        "normalized_reference",
        "normalized_prediction",
    }
    for value, (reference, prediction) in zip(
        result["table_evidence"], table_pairs, strict=True
    ):
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or any(not isinstance(value[name], str) for name in fields)
            or value["reference"] != reference
            or value["prediction"] != prediction
            or not value["normalized_reference"]
        ):
            raise ValueError(
                "Normalized official table evidence differs from its input pair"
            )
        normalized.append(
            {
                **value,
                **{
                    name + "_sha256": hashlib.sha256(
                        value[name].encode("utf-8")
                    ).hexdigest()
                    for name in fields
                },
            }
        )
    return tables, formulas, normalized


def official_image_digest() -> str:
    return official_evaluator_identity()["image_digest"]


def official_evaluator_identity() -> dict[str, str]:
    """Verified mounted source and immutable image identity for scoring provenance."""
    source = PROJECT_ROOT / "data/benchmarks/evaluators/OmniDocBench"
    return evaluator_identity(source)


def validate_official_runtime(
    *, tables: bool, formulas: bool, timeout_seconds: int
) -> None:
    html = '<table><thead><tr><th colspan="2">HeaderAlpha</th></tr><tr><td>x</td><td>y</td></tr></thead></table>'
    equivalent = '<table><tbody><tr><td colspan="2"><span>HeaderAlpha</span></td></tr><tr><td>x</td><td>y</td></tr></tbody></table>'
    table_pairs = (
        [
            (html, html),
            (html, equivalent),
            (html, html.replace("HeaderAlpha", "ChangedBeta")),
            (html, html.replace('colspan="2"', 'colspan="1"')),
        ]
        if tables
        else []
    )
    formula_pairs = [("x", "x")] if formulas else []
    table_scores, formula_scores = score_official_metrics(
        table_pairs, formula_pairs, timeout_seconds=timeout_seconds
    )
    if table_scores and (
        table_scores[:2] != [(1.0, 1.0), (1.0, 1.0)]
        or not (0 <= table_scores[2][0] < 1 and table_scores[2][1] == 1)
        or not all(0 <= value < 1 for value in table_scores[3])
    ):
        raise RuntimeError(
            "Official TEDS evaluator failed its normalization/header perturbation preflight"
        )
    if formula_scores and (
        formula_scores[0]["similarity"] != 1.0
        or formula_scores[0]["status"] != "evaluated"
        or any(
            type(formula_scores[0].get(name)) is not int
            for name in ("tp", "gt_tokens", "pred_tokens")
        )
        or not formula_scores[0]["tp"]
        == formula_scores[0].get("gt_tokens")
        == formula_scores[0].get("pred_tokens")
        > 0
    ):
        raise RuntimeError("Official CDM evaluator failed its identity preflight")


def _mount(source: Path, target: str, *, readonly: bool = False) -> str:
    value = f"type=bind,source={source.as_posix()},target={target}"
    return value + (",readonly" if readonly else "")
