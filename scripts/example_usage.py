"""Minimal programmatic example for the production pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from edumind.pipeline import EduMindPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and index one local study document")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    result = EduMindPipeline(use_llm=False).process_file(args.file)
    print(f"Source kind: {result.extraction.source_kind.value}")
    print(f"Text length: {len(result.extraction.text)}")
    print(f"Chunks created: {result.ingest.chunks_created if result.ingest else 0}")
    print(f"Total seconds: {result.timings['total_seconds']:.3f}")


if __name__ == "__main__":
    main()
