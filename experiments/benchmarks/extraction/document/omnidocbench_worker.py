"""Container entry point for the official OmniDocBench scorers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def main(input_path: Path, output_path: Path) -> None:
    sys.path.insert(0, "/opt/omnidocbench")
    sys.path.insert(0, "/opt/omnidocbench/src")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result: dict[str, list[object]] = {"tables": [], "formulas": []}

    tables = payload.get("tables", [])
    if tables:
        from metrics.table_metric import TEDS

        full = TEDS(structure_only=False)
        structure = TEDS(structure_only=True)
        for pair in tables:
            reference = str(pair.get("reference", ""))
            prediction = str(pair.get("prediction", ""))
            if not reference or not prediction:
                result["tables"].append([0.0, 0.0])
                continue
            result["tables"].append(
                [
                    float(full.evaluate(_html(prediction), _html(reference))),
                    float(structure.evaluate(_html(prediction), _html(reference))),
                ]
            )

    formulas = payload.get("formulas", [])
    if formulas:
        from metrics.cdm_metric import CDM

        output_root = Path("/work/cdm")
        output_root.mkdir(parents=True, exist_ok=True)
        evaluator = CDM(output_root=str(output_root))
        os.environ["CDM_SAVE_VIS"] = "0"
        for pair in formulas:
            reference = str(pair.get("reference", ""))
            prediction = str(pair.get("prediction", ""))
            if not reference or not prediction:
                result["formulas"].append(0.0)
                continue
            identifier = hashlib.sha256(
                f"{reference}\0{prediction}".encode("utf-8")
            ).hexdigest()[:16]
            metrics = evaluator.evaluate(reference, prediction, identifier)
            if metrics.get("cdm_eval_error"):
                raise RuntimeError(f"CDM evaluation failed: {metrics['cdm_eval_error']}")
            result["formulas"].append(float(metrics["F1_score"]))

    output_path.write_text(json.dumps(result), encoding="utf-8")


def _html(value: str) -> str:
    return value if "<body" in value.casefold() else f"<html><body>{value}</body></html>"


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
