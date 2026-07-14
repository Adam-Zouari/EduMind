"""Command tree for all benchmark, review, and reporting workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import BenchmarkResult
from .extraction import run_extraction_stage
from .preflight import preflight_payload, run_preflight
from .prepare import (
    prepare_extraction_models,
    prepare_huggingface_models,
    prepare_ollama_models,
    prepare_public_assets,
    prepare_qasper,
)
from .rag import run_chunking_embedding, run_final, run_generation, run_retrieval
from .report import render_report
from .review import export_review, import_review
from .vectordb import run_vectordb

EXTRACTION_STAGES = ("image", "pdf", "docx", "audio", "video", "normalization", "routing")
RAG_STAGES = ("chunking-embedding", "retrieval", "generation", "final")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edumind benchmark", description="Reproducible EduMind benchmarks"
    )
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="smoke")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    prepare = commands.add_parser("prepare")
    prepare_commands = prepare.add_subparsers(dest="prepare_command", required=True)
    prepare_qasper_parser = prepare_commands.add_parser("qasper")
    prepare_qasper_parser.add_argument("--output", type=Path, default=Path("data/benchmarks/rag"))
    prepare_assets_parser = prepare_commands.add_parser("assets")
    prepare_assets_parser.add_argument("plan", type=Path)
    prepare_assets_parser.add_argument("--output", type=Path, default=Path("data/benchmarks/raw"))
    prepare_huggingface_parser = prepare_commands.add_parser("huggingface-models")
    prepare_huggingface_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/models/huggingface.json"),
    )
    prepare_ollama_parser = prepare_commands.add_parser("ollama-models")
    prepare_ollama_parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/models/ollama.json")
    )
    prepare_extraction_parser = prepare_commands.add_parser("extraction-models")
    prepare_extraction_parser.add_argument(
        "--output", type=Path, default=Path("data/benchmarks/models/extraction.json")
    )
    prepare_extraction_parser.add_argument(
        "--cache", type=Path, default=Path("data/benchmarks/downloads/models")
    )
    extraction = commands.add_parser("extraction")
    extraction.add_argument("stage", choices=[*EXTRACTION_STAGES, "all"])
    rag = commands.add_parser("rag")
    rag.add_argument("stage", choices=[*RAG_STAGES, "all"])
    systems = commands.add_parser("systems")
    systems.add_argument("stage", choices=["vectordb"])
    review = commands.add_parser("review")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    review_export = review_commands.add_parser("export")
    review_export.add_argument("summary", type=Path)
    review_export.add_argument("output", type=Path)
    review_import = review_commands.add_parser("import")
    review_import.add_argument("review", type=Path)
    report = commands.add_parser("report")
    report.add_argument("summary", type=Path)
    report.add_argument("--output", type=Path)
    commands.add_parser("all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        payload = preflight_payload(run_preflight(args.profile))
        print(json.dumps(payload, indent=2))
        return 0 if payload["ready"] else 2
    if args.command == "prepare":
        if args.prepare_command == "qasper":
            outputs = prepare_qasper(args.output)
        elif args.prepare_command == "assets":
            outputs = prepare_public_assets(args.plan, args.output)
        elif args.prepare_command == "huggingface-models":
            outputs = [prepare_huggingface_models(args.output)]
        elif args.prepare_command == "extraction-models":
            outputs = [prepare_extraction_models(args.output, args.cache)]
        else:
            outputs = [prepare_ollama_models(args.output)]
        print(json.dumps([str(path) for path in outputs], indent=2))
        return 0
    if args.command == "review":
        if args.review_command == "export":
            print(export_review(args.summary, args.output))
        else:
            print(json.dumps(import_review(args.review), indent=2))
        return 0
    if args.command == "report":
        print(render_report(args.summary, args.output))
        return 0
    results: list[BenchmarkResult] = []
    if args.command in {"extraction", "all"}:
        extraction_stages = (
            EXTRACTION_STAGES if args.command == "all" or args.stage == "all" else (args.stage,)
        )
        results.extend(run_extraction_stage(stage, args.profile) for stage in extraction_stages)
    if args.command in {"rag", "all"}:
        rag_stages = RAG_STAGES if args.command == "all" or args.stage == "all" else (args.stage,)
        for stage in rag_stages:
            runner = {
                "chunking-embedding": run_chunking_embedding,
                "retrieval": run_retrieval,
                "generation": run_generation,
                "final": run_final,
            }[stage]
            results.append(runner(args.profile))
    if args.command == "systems" or args.command == "all":
        results.append(run_vectordb(args.profile))
    print(
        json.dumps(
            [
                {
                    "run_id": result.run_id,
                    "stage": result.plan.stage,
                    "artifact_directory": str(result.artifact_directory),
                    "pareto_candidates": result.pareto_candidates,
                    "authoritative": result.authoritative,
                    "failures": [
                        item.error for item in result.candidates if item.status != "success"
                    ],
                }
                for result in results
            ],
            indent=2,
        )
    )
    return (
        2
        if any(item.status != "success" for result in results for item in result.candidates)
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
