"""Direct CSV human-review export/import command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.benchmarks.common.review import export_review, import_review


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or import blinded RAG review")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("summary", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--finalists", type=int, default=3)
    export.add_argument("--questions", type=int, default=20)
    review_import = commands.add_parser("import")
    review_import.add_argument("review", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "export":
        print(
            export_review(
                arguments.summary,
                arguments.output,
                finalist_count=arguments.finalists,
                question_count=arguments.questions,
            )
        )
    else:
        print(json.dumps(import_review(arguments.review), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
