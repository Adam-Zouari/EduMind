"""Entry-point helper used by each extraction stage's run.py."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.common.arguments import parser, resolved_candidates
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.extraction.common import run

def main(stage: str, directory: Path) -> int:
    argument_parser = parser(f"Benchmark {stage} extraction")
    argument_parser.add_argument(
        "--document-selection",
        type=Path,
        help="engineer decision selecting exactly one document-parser candidate",
    )
    argument_parser.add_argument(
        "--audio-selection",
        type=Path,
        help="engineer decision selecting exactly one audio ASR candidate",
    )
    argument_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    if stage == "document":
        argument_parser.add_argument(
            "--phase", choices=("configuration", "architecture"), default="configuration"
        )
    arguments = argument_parser.parse_args()
    if arguments.profile != "smoke" and stage == "video":
        if arguments.document_selection is None:
            raise ValueError(
                f"video {arguments.profile} requires --document-selection so parsing is frozen"
            )
    if arguments.profile != "smoke" and stage == "video" and arguments.audio_selection is None:
        raise ValueError(
            f"video {arguments.profile} requires --audio-selection so ASR is frozen"
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
            arguments.document_selection, arguments.audio_selection, arguments.device
        ),
        decision_files={
            name: path
            for name, path in {
                "shortlist": arguments.shortlist,
                "document": arguments.document_selection,
                "audio": arguments.audio_selection,
            }.items()
            if path is not None
        },
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if result.complete else 2


def _component_options(
    document_selection: Path | None, audio_selection: Path | None, device: str
) -> dict[str, object]:
    options: dict[str, object] = {"device": device}
    if document_selection:
        document = _one_selection(document_selection)
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
    if audio_selection:
        audio_selection = _one_selection(audio_selection)
        options["audio_candidate"] = audio_selection
    return options


def _one_selection(path: Path) -> str:
    return load_engineer_decision(path, exact=1).selected_candidates[0]
