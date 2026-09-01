"""Direct CLI for explicit benchmark preparation; never runs work at import time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarks.common.selection import selection_entries
from experiments.benchmarks.preparation.datasets import (
    prepare_public_assets,
    prepare_qasper,
    prepare_rag_selection_manifest,
)
from experiments.benchmarks.preparation.fixtures import prepare_smoke_fixtures
from experiments.benchmarks.preparation.evaluators import prepare_evaluators
from experiments.benchmarks.preparation.models import (
    DOCLING_BENCHMARK_COMPONENTS,
    EXTRACTION_COMPONENTS,
    MODEL_COMPONENTS,
    RAG_COMPONENTS,
    preparation_plan,
    prepare_app_models,
    prepare_selected_models,
    selected_model_names,
)
from experiments.benchmarks.preparation.vectordb import prepare_vectordb

def main() -> int:
    parser = argparse.ArgumentParser(description="Explicit EduMind benchmark preparation")
    parser.add_argument(
        "target",
        nargs="?",
        choices=(
            "app-models",
            "rag-models",
            "extraction-models",
            "all-models",
            "qasper",
            "rag-selection",
            "assets",
            "vectordb",
            "smoke-fixtures",
            "evaluators",
        ),
    )
    parser.add_argument("--list", action="store_true", help="List approved model downloads")
    parser.add_argument("--dry-run", action="store_true", help="Print without downloading")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qasper-manifest", type=Path)
    parser.add_argument("--structured-manifest", type=Path)
    parser.add_argument(
        "--modality",
        choices=("all", "document", "audio", "video"),
        default="all",
        help="Fixture group to regenerate with the smoke-fixtures target",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="Prepare only this candidate; repeat the option for several candidates",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]

    if arguments.list:
        print(
            json.dumps(
                preparation_plan(
                    selected_model_names(MODEL_COMPONENTS),
                    DOCLING_BENCHMARK_COMPONENTS,
                ),
                indent=2,
            )
        )
        return 0
    if arguments.target is None:
        parser.error("target is required unless --list is used")
    if arguments.target != "smoke-fixtures" and arguments.modality != "all":
        parser.error("--modality is only valid with smoke-fixtures")

    if arguments.target == "app-models":
        if arguments.candidate:
            parser.error("app-models has a fixed provisional control set")
        outputs = prepare_app_models(root, dry_run=arguments.dry_run)
    elif arguments.target == "qasper":
        outputs = prepare_qasper(arguments.output or root / "data/benchmarks/rag")
    elif arguments.target == "rag-selection":
        if not arguments.qasper_manifest or not arguments.structured_manifest or not arguments.output:
            parser.error(
                "rag-selection requires --qasper-manifest, --structured-manifest, and --output"
            )
        outputs = [
            prepare_rag_selection_manifest(
                arguments.qasper_manifest,
                arguments.structured_manifest,
                arguments.output,
            )
        ]
    elif arguments.target == "assets":
        if arguments.plan is None:
            parser.error("assets requires --plan")
        outputs = prepare_public_assets(
            arguments.plan, arguments.output or root / "data/benchmarks/raw"
        )
    elif arguments.target in {"rag-models", "extraction-models", "all-models"}:
        components = {
            "rag-models": RAG_COMPONENTS,
            "extraction-models": EXTRACTION_COMPONENTS,
            "all-models": MODEL_COMPONENTS,
        }[arguments.target]
        selected = tuple(arguments.candidate or selected_model_names(components))
        entries = {entry.candidate: entry for entry in selection_entries()}
        for candidate in selected:
            entry = entries.get(candidate)
            if entry is None:
                parser.error(f"{candidate} is not an included model-selection candidate")
            if entry.component not in components:
                parser.error(f"{candidate} is not part of {arguments.target}")
        outputs = [
            prepare_selected_models(
                arguments.output or root / "data/benchmarks/models/selected.json",
                root / "data/benchmarks/downloads/models",
                selected,
                docling_components=(
                    DOCLING_BENCHMARK_COMPONENTS
                    if arguments.target in {"extraction-models", "all-models"}
                    and (
                        not arguments.candidate
                        or any(entries[name].component == "document_extraction" for name in selected)
                    )
                    else ()
                ),
                dry_run=arguments.dry_run,
            )
        ]
    elif arguments.target == "vectordb":
        outputs = [
            prepare_vectordb(
                arguments.output or root / "data/benchmarks/models/vectordb.json"
            )
        ]
    elif arguments.target == "evaluators":
        outputs = prepare_evaluators(root, dry_run=arguments.dry_run)
    else:
        fixture_manifest = root / "data/benchmarks/extraction/smoke.json"
        reliability_manifest = (
            root / "data/benchmarks/extraction/audio-reliability-smoke.json"
        )
        if not arguments.dry_run:
            prepare_smoke_fixtures(root, modality=arguments.modality)
        outputs = [fixture_manifest]
        if arguments.modality in {"all", "audio"}:
            outputs.append(reliability_manifest)

    print(json.dumps([str(path) for path in outputs], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
