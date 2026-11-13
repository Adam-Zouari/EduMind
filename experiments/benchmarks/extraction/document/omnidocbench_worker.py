"""Container entry point for the official OmniDocBench scorers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

SCHEMA_VERSION = 2
NORMALIZATION = (
    "omnidocbench.normalized_html_table@193627ae9e97d89188468ed1ee3b7a856ff76044"
)
EXACT_MATCH_POLICY = "cdm_exact_counts_v1"


def main(input_path: Path, output_path: Path) -> None:
    sys.path.insert(0, "/opt/omnidocbench")
    sys.path.insert(0, "/opt/omnidocbench/src")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": SCHEMA_VERSION,
        "tables": [],
        "formulas": [],
        "table_evidence": [],
        "provenance": {
            "normalization": NORMALIZATION,
            "exact_match_policy": EXACT_MATCH_POLICY,
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }

    tables = payload.get("tables", [])
    if tables:
        from lxml.etree import ParserError
        from metrics.table_metric import TEDS
        from src.core.preprocess.data_preprocess import normalized_html_table

        full = TEDS(structure_only=False)
        structure = TEDS(structure_only=True)
        for pair in tables:
            reference = pair["reference"]
            prediction = pair["prediction"]
            if not isinstance(reference, str) or not isinstance(prediction, str):
                raise ValueError("Official table inputs must be strings")
            normalized_reference = normalized_html_table(reference)
            normalized_prediction = normalized_html_table(prediction)
            result["table_evidence"].append(
                {
                    "reference": reference,
                    "prediction": prediction,
                    "normalized_reference": normalized_reference,
                    "normalized_prediction": normalized_prediction,
                }
            )
            # Validate the reference separately so a malformed prediction is a
            # quality zero, not confused with a broken reference or evaluator.
            if (
                not normalized_reference
                or full.evaluate(normalized_reference, normalized_reference) != 1.0
            ):
                raise ValueError("Invalid official table reference")
            if not normalized_prediction:
                result["tables"].append([0.0, 0.0])
                continue
            try:
                scores = [
                    float(full.evaluate(normalized_prediction, normalized_reference)),
                    float(
                        structure.evaluate(normalized_prediction, normalized_reference)
                    ),
                ]
            except (ValueError, ParserError, ZeroDivisionError):
                scores = [0.0, 0.0]
            result["tables"].append(scores)

    formulas = payload.get("formulas", [])
    if formulas:
        from metrics.cdm.modules import latex2bbox_color as renderer
        from metrics.cdm_metric import CDM

        output_root = Path("/work/cdm")
        output_root.mkdir(parents=True, exist_ok=True)
        evaluator = CDM(output_root=str(output_root))
        os.environ["CDM_SAVE_VIS"] = "0"
        for pair in formulas:
            reference = str(pair.get("reference", ""))
            prediction = str(pair.get("prediction", ""))
            if not reference.strip():
                raise ValueError("Invalid official formula reference: empty expression")
            if not prediction.strip():
                result["formulas"].append(
                    {
                        "similarity": 0.0,
                        "status": "missing_prediction",
                        "tp": None,
                        "gt_tokens": None,
                        "pred_tokens": None,
                    }
                )
                continue
            identifier = hashlib.sha256(
                f"{reference}\0{prediction}".encode()
            ).hexdigest()[:16]
            result["formulas"].append(
                _score_formula(evaluator, renderer, reference, prediction, identifier)
            )

    output_path.write_text(json.dumps(result), encoding="utf-8")


def _score_formula(evaluator, renderer, reference, prediction, identifier):
    """Retain the pinned scorer, observing only its swallowed infrastructure failures."""
    diagnostics = []
    run_cmd = renderer.run_cmd
    parse_log = renderer._parse_pdflatex_log
    # Only dependencies from the evaluator's own templates count here. A
    # prediction's missing input file or undefined LaTeX command is quality loss.
    dependencies = set()
    for template_name in (
        "formular_template",
        "formular_template_zh",
        "tabular_template",
    ):
        template = getattr(renderer, template_name)
        for command, names in re.findall(
            r"\\(usepackage|documentclass)(?:\[[^\]]*\])?\{([^}]+)\}", template
        ):
            suffix = ".sty" if command == "usepackage" else ".cls"
            dependencies.update(name.strip() + suffix for name in names.split(","))

    def observed_run_cmd(command, *args, **kwargs):
        code = run_cmd(command, *args, **kwargs)
        executable = Path(shlex.split(command)[0]).name
        if code < 0 or code in {126, 127} or executable == "magick" and code != 0:
            diagnostics.append(f"{executable} failed with exit code {code}")
        return code

    def observed_parse_log(log_path):
        if Path(log_path).is_file():
            log = Path(log_path).read_text(encoding="utf-8", errors="replace")
            missing = re.findall(r"LaTeX Error: File [`']([^`']+)['`] not found", log)
            diagnostics.extend(
                f"missing evaluator dependency {name}"
                for name in missing
                if name in dependencies
            )
        return parse_log(log_path)

    renderer.run_cmd = observed_run_cmd
    renderer._parse_pdflatex_log = observed_parse_log
    try:
        metrics = evaluator.evaluate(reference, prediction, identifier)
    finally:
        renderer.run_cmd = run_cmd
        renderer._parse_pdflatex_log = parse_log
    if diagnostics:
        raise RuntimeError(
            "CDM renderer infrastructure failed: " + "; ".join(diagnostics)
        )
    if metrics.get("cdm_eval_error"):
        raise RuntimeError(f"CDM evaluation failed: {metrics['cdm_eval_error']}")
    if type(metrics.get("gt_tokens")) is int and metrics["gt_tokens"] == 0:
        raise RuntimeError("Invalid official formula reference: rendered no tokens")
    similarity = metrics.get("F1_score")
    if (
        type(similarity) not in (int, float)
        or not math.isfinite(similarity)
        or not 0 <= similarity <= 1
    ):
        raise RuntimeError("CDM evaluator returned invalid similarity")
    counts = {name: metrics.get(name) for name in ("tp", "gt_tokens", "pred_tokens")}
    # Failed prediction rendering is a reconstruction zero, not invented counts.
    if (
        type(counts["pred_tokens"]) is int
        and counts["pred_tokens"] == 0
        and type(counts["gt_tokens"]) is int
        and counts["gt_tokens"] > 0
        and similarity == 0
    ):
        return {
            "similarity": 0.0,
            "status": "invalid_prediction",
            "tp": None,
            "gt_tokens": counts["gt_tokens"],
            "pred_tokens": 0,
        }
    return {"similarity": float(similarity), "status": "evaluated", **counts}


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
