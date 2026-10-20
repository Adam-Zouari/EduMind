"""Explicit full data validation; runners only verify the resulting reports."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from experiments.benchmarks.common.data_validation import PROFILES, manifest_path

PACKAGES = {
    **{
        name: f"experiments.benchmarks.extraction.{name}"
        for name in ("document", "audio", "video")
    },
    **{
        name: f"experiments.benchmarks.rag.{name.replace('-', '_')}"
        for name in ("chunking-embedding", "retrieval-reranking", "generation")
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=(*PACKAGES, "all"))
    parser.add_argument("--profile", choices=(*PROFILES, "all"), default="all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--chunking-protocol", type=Path)
    parser.add_argument("--inventory", type=Path, action="append")
    parser.add_argument("--reliability-manifest", type=Path)
    args = parser.parse_args()
    benchmarks = tuple(PACKAGES) if args.benchmark == "all" else (args.benchmark,)
    profiles = PROFILES if args.profile == "all" else (args.profile,)
    if (
        args.manifest
        or args.protocol
        or args.inventory
        or args.reliability_manifest
        or args.chunking_protocol
    ) and (len(benchmarks) > 1 or len(profiles) > 1):
        parser.error("Explicit inputs require one benchmark and one profile")
    if args.reliability_manifest and args.benchmark != "audio":
        parser.error("--reliability-manifest applies only to ASR data validation")
    if args.chunking_protocol and args.benchmark != "retrieval-reranking":
        parser.error(
            "--chunking-protocol applies only to retrieval-reranking validation"
        )
    reports = []
    for benchmark in benchmarks:
        package = PACKAGES[benchmark]
        protocol_module = importlib.import_module(package + ".protocol")
        validator = importlib.import_module(package + ".validation")
        protocol = protocol_module.load_protocol(
            args.protocol or protocol_module.DEFAULT_PROTOCOL_PATH
        )
        for profile in profiles:
            options = (
                {"controls_path": args.reliability_manifest}
                if benchmark == "audio"
                else {}
            )
            if benchmark == "retrieval-reranking":
                from experiments.benchmarks.rag.chunking_embedding.protocol import (
                    DEFAULT_PROTOCOL_PATH,
                    load_protocol,
                )

                options["chunking_protocol"] = load_protocol(
                    args.chunking_protocol or DEFAULT_PROTOCOL_PATH
                )
            path, report = validator.prepare_report(
                args.manifest or manifest_path(benchmark, profile),
                profile,
                protocol,
                inventory=args.inventory,
                **options,
            )
            reports.append(
                {
                    "benchmark": benchmark,
                    "profile": profile,
                    "report": str(path),
                    "status": report["status"],
                    "errors": report["errors"],
                }
            )
    print(json.dumps(reports, indent=2))
    return 0 if all(report["status"] == "passed" for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
