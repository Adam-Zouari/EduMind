"""Explicit dependency, dataset, model, and local-engine qualification."""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import requests

from edumind.common.paths import PROJECT_ROOT

from .prepare import load_model_lock


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    ready: bool
    required: bool
    guidance: str


def run_preflight(
    profile: str, *, suites: Iterable[str] = ("extraction", "rag", "systems")
) -> list[PreflightCheck]:
    if profile not in {"smoke", "standard", "full"}:
        raise ValueError(f"Unknown benchmark profile: {profile}")
    selected = set(suites)
    checks = [
        PreflightCheck(
            "python-package",
            importlib.util.find_spec("edumind") is not None,
            True,
            "Install with `pip install -e .[dev]`.",
        ),
        PreflightCheck(
            "smoke-rag-manifest",
            (PROJECT_ROOT / "data/benchmarks/rag/smoke.json").is_file(),
            True,
            "Restore committed smoke fixtures.",
        ),
        PreflightCheck(
            "smoke-extraction-manifest",
            (PROJECT_ROOT / "data/benchmarks/extraction/smoke.json").is_file(),
            True,
            "Restore committed smoke fixtures.",
        ),
    ]
    if profile == "smoke":
        return checks
    checks.extend(
        [
            _module("psutil", "Install .[benchmarks] for process resource measurement."),
            _module("pynvml", "Install nvidia-ml-py for NVIDIA VRAM measurement."),
        ]
    )
    if "extraction" in selected:
        checks.extend(
            [
                PreflightCheck(
                    "standard-extraction-manifests",
                    all(
                        (
                            PROJECT_ROOT / f"data/benchmarks/extraction/{stage}-validation.json"
                        ).is_file()
                        for stage in (
                            "image",
                            "pdf",
                            "docx",
                            "audio",
                            "video",
                            "normalization",
                            "routing",
                        )
                    ),
                    True,
                    "Prepare the licensed extraction manifests and raw assets described in each "
                    "experiment document.",
                ),
                PreflightCheck(
                    "extraction-model-lock",
                    (PROJECT_ROOT / "data/benchmarks/models/extraction.json").is_file(),
                    True,
                    "Run `edumind benchmark prepare extraction-models`.",
                ),
                PreflightCheck(
                    "tesseract",
                    shutil.which("tesseract") is not None,
                    True,
                    "Install Tesseract 5 and add it to PATH.",
                ),
                PreflightCheck(
                    "ffmpeg",
                    shutil.which("ffmpeg") is not None,
                    True,
                    "Install FFmpeg and add it to PATH.",
                ),
                _module("paddleocr", "Install the extraction extra and pinned PaddleOCR models."),
                _module("docling", "Install the benchmark document-conversion dependencies."),
                _module(
                    "faster_whisper", "Install the ASR extra and prepare pinned Whisper weights."
                ),
            ]
        )
    if "rag" in selected:
        checks.extend(
            [
                _module(
                    "sentence_transformers",
                    "Install the RAG extra and prepare pinned embedding/reranker weights.",
                ),
                PreflightCheck(
                    "qasper-manifests",
                    all(
                        (PROJECT_ROOT / f"data/benchmarks/rag/qasper-{split}.json").is_file()
                        for split in ("dev", "validation", "locked-test")
                    ),
                    True,
                    "Run `edumind benchmark prepare qasper` before standard benchmarks.",
                ),
                PreflightCheck(
                    "huggingface-model-lock",
                    (PROJECT_ROOT / "data/benchmarks/models/huggingface.json").is_file(),
                    True,
                    "Run `edumind benchmark prepare huggingface-models`.",
                ),
                _ollama(),
                PreflightCheck(
                    "ollama-model-lock",
                    (PROJECT_ROOT / "data/benchmarks/models/ollama.json").is_file(),
                    True,
                    "Run `edumind benchmark prepare ollama-models` after Ollama starts.",
                ),
            ]
        )
    if "systems" in selected:
        checks.extend(
            [
                _module("chromadb", "Install the benchmark extra."),
                _module("qdrant_client", "Install the benchmark extra."),
                _module("lancedb", "Install the benchmark extra."),
            ]
        )
    return checks


def preflight_payload(checks: list[PreflightCheck]) -> dict[str, object]:
    return {
        "ready": all(check.ready for check in checks if check.required),
        "checks": [asdict(check) for check in checks],
    }


def _module(name: str, guidance: str) -> PreflightCheck:
    return PreflightCheck(name, importlib.util.find_spec(name) is not None, True, guidance)


def _ollama() -> PreflightCheck:
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("models", []) if isinstance(payload, dict) else []
        installed = {
            str(row.get("name")): str(row.get("digest")) for row in rows if isinstance(row, dict)
        }
        locked = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/ollama.json")
        ready = all(installed.get(name) == digest for name, digest in locked.items())
    except (requests.RequestException, RuntimeError, ValueError):
        ready = False
    return PreflightCheck(
        "ollama",
        ready,
        True,
        "Start Ollama and pull every pinned generation profile listed in the "
        "generation benchmark document.",
    )
