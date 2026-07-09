"""Minimal programmatic example for the packaged orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from edumind.pipeline.orchestrator import OCRRAGOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple OCR + RAG example")
    parser.add_argument("file", nargs="?", help="Path to a file to ingest")
    args = parser.parse_args()

    orchestrator = OCRRAGOrchestrator(use_llm=False)

    if not args.file:
        print("Provide a document path to ingest, for example:")
        print("  python scripts/example_usage.py path/to/document.pdf")
        return

    file_path = Path(args.file)
    result = orchestrator.process_file(file_path=file_path, ingest_to_rag=True, clean_text=True)
    print(f"Format: {result['format_type']}")
    print(f"Text length: {len(result['text'])}")
    print(f"Chunks created: {result['rag_chunks']}")


if __name__ == "__main__":
    main()
