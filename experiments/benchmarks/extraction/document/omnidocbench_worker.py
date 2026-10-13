"""Container entry point for the official OmniDocBench scorers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path


def main(input_path: Path, output_path: Path) -> None:
    sys.path.insert(0, "/opt/omnidocbench")
    sys.path.insert(0, "/opt/omnidocbench/src")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result: dict[str, list[object]] = {"tables": [], "formulas": []}

    tables = payload.get("tables", [])
    if tables:
        from lxml.etree import ParserError
        from metrics.table_metric import TEDS

        full = TEDS(structure_only=False)
        structure = TEDS(structure_only=True)
        for pair in tables:
            reference = str(pair.get("reference", ""))
            prediction = str(pair.get("prediction", ""))
            if not reference or not prediction:
                result["tables"].append([0.0, 0.0])
                continue
            # Validate the reference separately so a malformed prediction is a
            # quality zero, not confused with a broken reference or evaluator.
            if full.evaluate(_html(reference), _html(reference)) != 1.0:
                raise ValueError("Invalid official table reference")
            try:
                scores = [
                    float(full.evaluate(_html(prediction), _html(reference))),
                    float(structure.evaluate(_html(prediction), _html(reference))),
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
                result["formulas"].append(0.0)
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
    if type(metrics.get("gt_tokens")) is not int or metrics["gt_tokens"] <= 0:
        raise RuntimeError("Invalid official formula reference: rendered no tokens")
    return float(metrics["F1_score"])


def _html(value: str) -> str:
    return (
        value if "<body" in value.casefold() else f"<html><body>{value}</body></html>"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
