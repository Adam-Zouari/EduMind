"""Command-line orchestration for document benchmark phases."""

from __future__ import annotations

import json
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import execution_devices, parser
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.extraction.document.benchmark import (
    run,
    run_preflight_profile,
)
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main(directory: Path) -> int:
    argument_parser = parser(
        "Benchmark document extraction",
        shortlist=False,
        profiles=("smoke", "preflight", "development", "validation", "locked"),
    )
    argument_parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    argument_parser.add_argument("--preflight-report", type=Path)
    argument_parser.add_argument("--preflight-run-id")
    argument_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    argument_parser.add_argument(
        "--source", choices=("all", "pdf", "image", "docx"), default="all"
    )
    argument_parser.add_argument(
        "--comparison",
        choices=("configuration", "architecture"),
        default=None,
        help=(
            "comparison override; defaults to configuration for smoke/development "
            "and architecture for validation/locked"
        ),
    )
    argument_parser.add_argument(
        "--pdf-selection",
        type=Path,
        help=(
            "PDF configuration decision (development) or architecture-finalist "
            "decision (validation/locked)"
        ),
    )
    argument_parser.add_argument(
        "--image-selection",
        type=Path,
        help=(
            "image configuration decision (development) or architecture-finalist "
            "decision (validation/locked)"
        ),
    )
    arguments = argument_parser.parse_args()
    return _document_main(arguments, directory)


def _document_main(arguments, directory: Path) -> int:
    del directory
    protocol = load_protocol(arguments.protocol)
    if arguments.comparison is None:
        arguments.comparison = (
            "architecture"
            if arguments.profile in {"validation", "locked"}
            else "configuration"
        )
    execution_name = (
        "development" if arguments.profile == "preflight" else arguments.profile
    )
    execution = protocol.profile(execution_name)
    devices = execution_devices(
        arguments.profile,
        arguments.device,
        smoke_devices=protocol.profile("smoke").devices,
        authoritative_device=execution.device,
    )
    if arguments.profile == "preflight":
        if any(
            value is not None
            for value in (arguments.pdf_selection, arguments.image_selection)
        ):
            raise ValueError("Document preflight does not consume selection decisions")
        result = run_preflight_profile(
            manifest_path=arguments.manifest,
            no_mlflow=arguments.no_mlflow,
            protocol_path=arguments.protocol,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "ready_for_development": result.ready_for_development,
                    "qualified_candidates": result.qualified_candidates,
                    "excluded_candidates": result.excluded_candidates,
                    "blocked_candidates": result.blocked_candidates,
                    "artifacts": str(result.artifact_directory),
                },
                indent=2,
            )
        )
        return 0 if result.ready_for_development else 2
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
    for device in devices:
        for source in sources:
            candidates, decisions = _document_candidates(source, arguments, protocol)
            results.append(
                (
                    device,
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
                        preflight_report=arguments.preflight_report,
                        preflight_run_id=arguments.preflight_run_id,
                        protocol_path=arguments.protocol,
                    ),
                )
            )
    print(
        json.dumps(
            [
                {
                    "device": device,
                    "source": source,
                    "run_id": result.run_id,
                    "complete": result.complete,
                    "artifacts": str(result.artifact_directory),
                }
                for device, source, result in results
            ],
            indent=2,
        )
    )
    return 0 if all(result.complete for _, _, result in results) else 2


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
    if decision_path is None:
        suffix = {
            "development": "configuration",
            "validation": "validation",
            "locked": "locked",
        }[arguments.profile]
        decision_path = (
            PROJECT_ROOT
            / "data/benchmarks/decisions"
            / f"document-{source}-{suffix}.json"
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
    if arguments.profile == "validation":
        selected = _document_selection(
            decision_path,
            f"document-architecture-development-{source}",
            maximum=protocol.maximum_architecture_finalists,
        )
    else:
        selected = _document_selection(
            decision_path,
            f"document-architecture-validation-{source}",
            expected_profile="validation",
            exact=1,
        )
    return selected, {source: decision_path}


def _validate_document_arguments(arguments, sources: tuple[str, ...]) -> None:
    if arguments.profile == "locked" and sources != ("pdf", "image", "docx"):
        raise ValueError(
            "Document locked evaluation must run the complete PDF, image, and DOCX "
            "routing policy; --source overrides are forbidden"
        )
    comparison = arguments.comparison
    if comparison == "configuration":
        if arguments.profile in {"validation", "locked"}:
            raise ValueError(
                "Document configuration is selected on development; validation "
                "and locked runs require --comparison architecture"
            )
        if arguments.pdf_selection or arguments.image_selection:
            raise ValueError("Selection files apply only to --comparison architecture")
        return
    if arguments.profile == "smoke":
        raise ValueError(
            "Document architecture comparison requires development, validation, "
            "or locked-test data"
        )
    for source in sources:
        if source not in {"pdf", "image", "docx"}:
            raise ValueError(f"Unsupported document source: {source}")


def _document_selection(
    path: Path | None,
    expected_stage: str,
    *,
    expected_profile: str = "development",
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
        expected_source=("extraction", expected_stage, expected_profile),
    )
    return decision.selected_candidates
