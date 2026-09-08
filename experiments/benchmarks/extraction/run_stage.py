"""Entry-point helper used by each extraction stage's run.py."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmarks.common.arguments import load_candidates, parser, resolved_candidates
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.extraction.common import run


def main(stage: str, directory: Path) -> int:
    argument_parser = parser(
        f"Benchmark {stage} extraction", shortlist=stage != "document"
    )
    if stage == "video":
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
            help="PDF configuration decision (standard) or architecture-finalist decision (full)",
        )
        argument_parser.add_argument(
            "--image-selection",
            type=Path,
            help="image configuration decision (standard) or architecture-finalist decision (full)",
        )
    arguments = argument_parser.parse_args()
    if stage == "document":
        return _document_main(arguments, directory)
    if stage == "video" and arguments.profile != "smoke":
        raise RuntimeError(
            "Authoritative video runs are deferred until the downloaded data fixes "
            "the occurrence-matching and ASR window-stitching rules; use --profile smoke "
            "only for the current wiring check."
        )
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
            getattr(arguments, "document_selection", None),
            getattr(arguments, "audio_selection", None),
            arguments.device,
        ),
        decision_files={
            name: path
            for name, path in {
                "shortlist": getattr(arguments, "shortlist", None),
                "document": getattr(arguments, "document_selection", None),
                "audio": getattr(arguments, "audio_selection", None),
            }.items()
            if path is not None
        },
    )
    print(json.dumps({"run_id": result.run_id, "artifacts": str(result.artifact_directory)}, indent=2))
    return 0 if result.complete else 2


def _document_main(arguments, directory: Path) -> int:
    sources = ("pdf", "image", "docx") if arguments.source == "all" else (arguments.source,)
    _validate_document_arguments(arguments, sources)
    comparison = (
        "configuration"
        if arguments.comparison == "configuration"
        else f"architecture-{'development' if arguments.profile == 'standard' else 'validation'}"
    )
    results = []
    for source in sources:
        candidates, decisions = _document_candidates(
            source, arguments, directory / "candidates.yaml"
        )
        results.append(
            (
                source,
                run(
                    "document",
                    arguments.profile,
                    candidates,
                    manifest_path=arguments.manifest,
                    no_mlflow=arguments.no_mlflow,
                    component_options={"device": arguments.device},
                    decision_files=decisions,
                    document_kind=source,
                    document_comparison=comparison,
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


def _document_candidates(source: str, arguments, path: Path):
    comparison = getattr(arguments, "comparison", "configuration")
    if comparison == "configuration":
        configured = load_candidates(path, arguments.profile)
        if source == "pdf":
            return configured, {}
        if source == "image":
            unique = []
            for candidate in configured:
                factors = [
                    factor
                    for factor in candidate.split("|")
                    if not factor.startswith("mode=")
                ]
                factors.insert(2, "mode=full_page")
                value = "|".join(factors)
                if value not in unique:
                    unique.append(value)
            return tuple(unique), {}
        return ("docling-standard-native",), {}

    if source == "docx":
        return ("docling-standard-native",), {}
    decision_path = (
        arguments.pdf_selection if source == "pdf" else arguments.image_selection
    )
    if arguments.profile == "standard":
        selected = _document_selection(
            decision_path, f"document-configuration-{source}", exact=1
        )
        return (
            selected[0],
            "docling-vlm-granite-258m",
            "paddleocr-vl-1.6",
        ), {source: decision_path}
    selected = _document_selection(
        decision_path, f"document-architecture-development-{source}", maximum=3
    )
    return selected, {source: decision_path}


def _validate_document_arguments(arguments, sources: tuple[str, ...]) -> None:
    comparison = arguments.comparison
    if comparison == "configuration":
        if arguments.profile == "full":
            raise ValueError(
                "Document configuration is selected on development; use --profile standard"
            )
        if arguments.pdf_selection or arguments.image_selection:
            raise ValueError("Selection files apply only to --comparison architecture")
        return
    if arguments.profile == "smoke":
        raise ValueError("Document architecture comparison requires standard or full data")
    for source in sources:
        if source == "pdf" and arguments.pdf_selection is None:
            raise ValueError("Document architecture comparison requires --pdf-selection")
        if source == "image" and arguments.image_selection is None:
            raise ValueError("Document architecture comparison requires --image-selection")


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


def _document_selection(
    path: Path | None,
    expected_stage: str,
    *,
    exact: int | None = None,
    maximum: int | None = None,
) -> tuple[str, ...]:
    if path is None:
        raise ValueError(f"Document comparison requires a decision for {expected_stage}")
    decision = load_engineer_decision(path, exact=exact, maximum=maximum)
    summary = json.loads(decision.source_summary.read_text(encoding="utf-8"))
    if summary.get("plan", {}).get("stage") != expected_stage:
        raise ValueError(f"{path} must select from a completed {expected_stage} run")
    return decision.selected_candidates
