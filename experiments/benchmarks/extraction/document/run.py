from pathlib import Path

from experiments.benchmarks.extraction.document.cli import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).parent))
