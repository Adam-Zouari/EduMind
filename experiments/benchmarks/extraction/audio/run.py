from pathlib import Path

from experiments.benchmarks.extraction.audio.runner import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).parent))
