"""Command-line orchestration for document benchmark phases."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.common.arguments import execution_devices, parser
from experiments.benchmarks.extraction.document.benchmark import (
    run,
    run_preflight_profile,
)
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main() -> int:
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
    return _document_main(arguments)


def _document_main(arguments) -> int:
    protocol = load_protocol(arguments.protocol)
    if arguments.profile == "preflight":
        ignored = []
        if arguments.source != "all":
            ignored.append("--source")
        if arguments.comparison is not None:
            ignored.append("--comparison")
        if arguments.pdf_selection is not None:
            ignored.append("--pdf-selection")
        if arguments.image_selection is not None:
            ignored.append("--image-selection")
        if arguments.preflight_report is not None:
            ignored.append("--preflight-report")
        if arguments.preflight_run_id is not None:
            ignored.append("--preflight-run-id")
        if ignored:
            raise ValueError(
                "Document preflight qualifies every declared route and does not "
                "accept: " + ", ".join(ignored)
            )
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
            if (
                source == "docx"
                and arguments.profile == "development"
                and arguments.comparison == "architecture"
            ):
                if arguments.source == "docx":
                    raise ValueError(
                        "Native DOCX has no parser comparison; reuse development-docx configuration evidence"
                    )
                continue
            results.append(
                (
                    device,
                    source,
                    run(
                        arguments.profile,
                        manifest_path=arguments.manifest,
                        no_mlflow=arguments.no_mlflow,
                        component_options={"device": device},
                        selection_path=(
                            arguments.pdf_selection
                            if source == "pdf"
                            else arguments.image_selection
                            if source == "image"
                            else None
                        ),
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
