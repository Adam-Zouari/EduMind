from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.benchmarks.extraction.audio.runner import main

raise SystemExit(main(Path(__file__).parent))
