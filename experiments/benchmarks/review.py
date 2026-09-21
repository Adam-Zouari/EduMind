"""Direct CSV human-review export/import command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.benchmarks.rag.final.review import export_review, import_review
from experiments.benchmarks.rag.final.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or import blinded RAG review")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("selection", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    review_import = commands.add_parser("import")
    review_import.add_argument("review", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "export":
        protocol = load_protocol(arguments.protocol)
        print(
            export_review(
                arguments.selection,
                arguments.output,
                finalist_count=protocol.human_review_system_count,
                question_count=protocol.human_review_sample_count,
                seed=protocol.meta.seed,
                protocol=protocol.meta.artifact_payload(),
            )
        )
    else:
        print(json.dumps(import_review(arguments.review), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
