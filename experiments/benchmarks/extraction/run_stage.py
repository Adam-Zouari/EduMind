"""Entry-point helper used by each extraction stage's run.py."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.extraction.common import run

def main(stage: str, directory: Path) -> int:
    argument_parser = parser(f"Benchmark {stage} extraction")
    argument_parser.add_argument(
        "--document-summary",
        type=Path,
        help="summary.json containing exactly one approved document-parser candidate",
    )
    argument_parser.add_argument(
        "--audio-summary",
        type=Path,
        help="summary.json containing exactly one approved audio ASR candidate",
    )
    argument_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    if stage == "document":
        argument_parser.add_argument(
            "--phase", choices=("configuration", "architecture"), default="configuration"
        )
    arguments = argument_parser.parse_args()
    if arguments.profile != "smoke" and stage == "video":
        if arguments.document_summary is None:
            raise ValueError(
                f"video {arguments.profile} requires --document-summary so parsing is frozen"
            )
    if arguments.profile != "smoke" and stage == "video" and arguments.audio_summary is None:
        raise ValueError(
            f"video {arguments.profile} requires --audio-summary so ASR is frozen"
        )
    if stage == "document" and arguments.phase == "architecture":
        if arguments.shortlist is None:
            raise ValueError("Document architecture phase requires the configuration summary")
        finalists = resolved_candidates(
            directory / "candidates.yaml", arguments.profile, arguments.shortlist
        )
        candidates = tuple(
            dict.fromkeys(
                [*finalists, "docling-vlm-granite-258m", "paddleocr-vl-1.6"]
            )
        )
    else:
        candidates = resolved_candidates(
            directory / "candidates.yaml", arguments.profile, arguments.shortlist
        )
    result = run(
        stage,
        arguments.profile,
        candidates,
        manifest_path=arguments.manifest,
        no_mlflow=arguments.no_mlflow,
        component_options=_component_options(
            arguments.document_summary, arguments.audio_summary, arguments.device
        ),
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if all(candidate.status == "success" for candidate in result.candidates) else 2


def _component_options(
    document_summary: Path | None, audio_summary: Path | None, device: str
) -> dict[str, object]:
    options: dict[str, object] = {"device": device}
    if document_summary:
        document = _one_selection(document_summary)
        factors = document.split("|")
        options["image_engine"] = factors[0]
        for factor in factors[1:]:
            key, value = factor.split("=", 1)
            options[
                {
                    "ocr": "image_ocr_engine",
                    "mode": "image_ocr_mode",
                    "table": "image_table_mode",
                    "formula": "image_formula_enrichment",
                }[key]
            ] = value == "on" if key == "formula" else value
    if audio_summary:
        audio_selection = _one_selection(audio_summary)
        options["audio_candidate"] = audio_selection
    return options


def _one_selection(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("pareto_candidates")
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{path} must contain exactly one explicitly approved candidate")
    return str(values[0])
