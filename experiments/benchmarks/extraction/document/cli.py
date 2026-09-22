"""Command-line orchestration for document benchmark phases."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.common.arguments import parser
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.extraction.document.benchmark import run
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main(directory: Path) -> int:
    argument_parser = parser("Benchmark document extraction", shortlist=False)
    argument_parser.add_argument("--device", choices=("cpu", "cuda"))
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    argument_parser.add_argument(
        "--source", choices=("all", "pdf", "image", "docx"), default="all"
    )
    argument_parser.add_argument(
        "--comparison",
        choices=("configuration", "architecture"),
        default="configuration",
        help=(
            "configuration screens Docling settings on development; architecture "
            "compares selected settings on development or finalists on validation"
        ),
    )
    argument_parser.add_argument(
        "--pdf-selection",
        type=Path,
        help=(
            "PDF configuration decision (development) or architecture-finalist "
            "decision (validation)"
        ),
    )
    argument_parser.add_argument(
        "--image-selection",
        type=Path,
        help=(
            "image configuration decision (development) or architecture-finalist "
            "decision (validation)"
        ),
    )
    arguments = argument_parser.parse_args()
    return _document_main(arguments, directory)


def _document_main(arguments, directory: Path) -> int:
    protocol = load_protocol(arguments.protocol)
    device = arguments.device or protocol.profile(arguments.profile).device
    sources = (
        ("pdf", "image", "docx") if arguments.source == "all" else (arguments.source,)
    )
    _validate_document_arguments(arguments, sources)
    comparison = (
        "configuration"
        if arguments.comparison == "configuration"
        else f"architecture-{arguments.profile}"
    )
    results = []
    for source in sources:
        candidates, decisions = _document_candidates(source, arguments, protocol)
        results.append(
            (
                source,
                run(
                    arguments.profile,
                    candidates,
                    manifest_path=arguments.manifest,
                    no_mlflow=arguments.no_mlflow,
                    component_options={"device": device},
                    decision_files=decisions,
                    document_kind=source,
                    document_comparison=comparison,
                    protocol_path=arguments.protocol,
                ),
            )
        )
    print(
        json.dumps(
            [
                {
                    "source": source,
                    "run_id": result.run_id,
                    "complete": result.complete,
                    "artifacts": str(result.artifact_directory),
                }
                for source, result in results
            ],
            indent=2,
        )
    )
    return 0 if all(result.complete for _, result in results) else 2


def _document_candidates(source: str, arguments, protocol):
    comparison = getattr(arguments, "comparison", "configuration")
    if comparison == "configuration":
        if source == "pdf":
            return protocol.configuration_candidates(arguments.profile), {}
        if source == "image":
            return protocol.configuration_candidates(arguments.profile, image=True), {}
        return ("docling-standard-native",), {}

    if source == "docx":
        return ("docling-standard-native",), {}
    decision_path = (
        arguments.pdf_selection if source == "pdf" else arguments.image_selection
    )
    if arguments.profile == "development":
        selected = _document_selection(
            decision_path, f"document-configuration-{source}", exact=1
        )
        return (
            selected[0],
            "docling-vlm-granite-258m",
            "paddleocr-vl-1.6",
        ), {source: decision_path}
    selected = _document_selection(
        decision_path,
        f"document-architecture-development-{source}",
        maximum=protocol.maximum_architecture_finalists,
    )
    return selected, {source: decision_path}


def _validate_document_arguments(arguments, sources: tuple[str, ...]) -> None:
    comparison = arguments.comparison
    if comparison == "configuration":
        if arguments.profile == "validation":
            raise ValueError(
                "Document configuration is selected on development; use "
                "--profile development"
            )
        if arguments.pdf_selection or arguments.image_selection:
            raise ValueError("Selection files apply only to --comparison architecture")
        return
    if arguments.profile == "smoke":
        raise ValueError(
            "Document architecture comparison requires development or validation data"
        )
    for source in sources:
        if source == "pdf" and arguments.pdf_selection is None:
            raise ValueError(
                "Document architecture comparison requires --pdf-selection"
            )
        if source == "image" and arguments.image_selection is None:
            raise ValueError(
                "Document architecture comparison requires --image-selection"
            )


def _document_selection(
    path: Path | None,
    expected_stage: str,
    *,
    exact: int | None = None,
    maximum: int | None = None,
) -> tuple[str, ...]:
    if path is None:
        raise ValueError(
            f"Document comparison requires a decision for {expected_stage}"
        )
    decision = load_engineer_decision(
        path,
        exact=exact,
        maximum=maximum,
        expected_source=("extraction", expected_stage, "development"),
    )
    return decision.selected_candidates
