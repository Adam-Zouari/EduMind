"""Lazy complete-document parsers used by extraction benchmarks.

The adapters deliberately expose one small production contract.  Candidate-specific
installation and model downloads remain explicit benchmark preparation steps.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..contracts import (
    ExtractedDocument,
    ExtractionRequest,
    ExtractionWarning,
    SourceKind,
)
from ..errors import ExtractionBackendError, MissingDependencyError
from ..structured import build_markdown_document


class StructuredDocumentExtractor:
    """Convert an image, PDF, or DOCX to typed Markdown elements."""

    supported_kinds = frozenset({SourceKind.IMAGE, SourceKind.PDF, SourceKind.DOCX})

    def __init__(self, engine: str, revision: str = "unpinned") -> None:
        self.engine = engine
        self.name = engine
        self.revision = revision
        self._runtime: Any | None = None

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        warnings: list[ExtractionWarning] = []
        try:
            if self.engine == "docling":
                pages = self._docling(request, kind, warnings)
            elif self.engine in {"pp-structure-v3", "paddleocr-vl-1.6"}:
                pages = self._paddle_document(request)
            elif self.engine == "glm-ocr":
                pages = self._glm_ocr(request)
            elif self.engine == "mineru-2.5-pro":
                pages = self._mineru(request)
            elif self.engine == "olmocr-2-7b":
                pages = self._olmocr(request)
            else:
                raise ValueError(f"Unknown complete-document engine: {self.engine}")
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Complete-document extraction failed with {self.engine}", detail=str(exc)
            ) from exc
        return build_markdown_document(
            request,
            kind,
            request.profile,
            pages,
            metadata={
                "engine": self.engine,
                "engine_revision": request.profile.engine_revision,
                "serialization": "markdown",
            },
            warnings=warnings,
            seconds=time.perf_counter() - started,
        )

    def _docling(
        self,
        request: ExtractionRequest,
        kind: SourceKind,
        warnings: list[ExtractionWarning],
    ) -> list[str]:
        try:
            from docling.document_converter import DocumentConverter
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Docling is required for this candidate") from exc
        artifacts = Path(str(request.options.get("docling_artifacts_dir", "")))
        if not artifacts.is_dir():
            raise FileNotFoundError(
                "Docling models are not prepared; run `python "
                "experiments/benchmarks/prepare.py extraction-models`"
            )
        if self._runtime is None:
            os.environ["DOCLING_ARTIFACTS_PATH"] = str(artifacts)
            self._runtime = DocumentConverter()
        document = self._runtime.convert(str(request.source_path)).document
        markdown = str(document.export_to_markdown())
        if kind is SourceKind.PDF:
            warnings.append(
                ExtractionWarning(
                    "page_boundaries_unavailable",
                    "Docling Markdown was returned as one document; page-attribution metrics "
                    "must use Docling's native document export when that annotation is required.",
                )
            )
        return [markdown]

    def _paddle_document(self, request: ExtractionRequest) -> list[str]:
        try:
            if self.engine == "pp-structure-v3":
                from paddleocr import PPStructureV3 as Pipeline
            else:
                from paddlex import create_pipeline
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError(
                f"The installed PaddleOCR package does not provide {self.engine}"
            ) from exc
        cache_directory = Path(str(request.options.get("paddle_cache_dir", "")))
        if not cache_directory.is_dir():
            raise FileNotFoundError(
                "Paddle document models are not prepared; run `python "
                "experiments/benchmarks/prepare.py extraction-models`"
            )
        if self._runtime is None:
            device = "gpu" if request.profile and request.profile.device == "cuda" else "cpu"
            self._runtime = (
                Pipeline(device=device)
                if self.engine == "pp-structure-v3"
                else create_pipeline(pipeline="PaddleOCR-VL-1.6", device=device)
            )
        pages: list[str] = []
        for result in self._runtime.predict(str(request.source_path)):
            markdown = getattr(result, "markdown", {})
            if callable(markdown):
                markdown = markdown()
            pages.append(_markdown_text(markdown))
        if not pages:
            raise RuntimeError(f"{self.engine} produced no pages")
        return pages

    def _glm_ocr(self, request: ExtractionRequest) -> list[str]:
        executable = _required_command("glmocr")
        model_path = Path(str(request.options.get("model_path", "")))
        if not model_path.is_dir():
            raise FileNotFoundError(
                "Pinned GLM-OCR weights are missing; run `python "
                "experiments/benchmarks/prepare.py extraction-models --candidate glm-ocr`"
            )
        config = Path(str(request.options.get("glm_config_path", "")))
        if not config.is_file():
            raise FileNotFoundError(
                "GLM-OCR requires a pinned local/self-hosted config in glm_config_path; "
                "hosted API evaluation is not allowed"
            )
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
        pipeline = payload.get("pipeline", {}) if isinstance(payload, Mapping) else {}
        maas = pipeline.get("maas", {}) if isinstance(pipeline, Mapping) else {}
        ocr_api = pipeline.get("ocr_api", {}) if isinstance(pipeline, Mapping) else {}
        if isinstance(maas, Mapping) and bool(maas.get("enabled")):
            raise ValueError("GLM-OCR benchmark configuration must disable hosted MaaS")
        if not isinstance(ocr_api, Mapping) or not ocr_api.get("api_host"):
            raise ValueError(
                "GLM-OCR benchmark configuration must identify the self-hosted ocr_api"
            )
        with tempfile.TemporaryDirectory(prefix="edumind-glmocr-") as temporary:
            output = Path(temporary) / "output"
            command = [
                executable,
                "parse",
                str(request.source_path),
                "--output",
                str(output),
                "--config",
                str(config),
            ]
            _run(command, request)
            return _read_markdown_outputs(output)

    def _mineru(self, request: ExtractionRequest) -> list[str]:
        executable = _required_command("mineru")
        model_path = Path(str(request.options.get("model_path", "")))
        config_path = Path(str(request.options.get("mineru_config_path", "")))
        if not model_path.is_dir() or not config_path.is_file():
            raise FileNotFoundError(
                "Pinned MinerU weights/config are missing; run `python "
                "experiments/benchmarks/prepare.py extraction-models "
                "--candidate mineru-2.5-pro`"
            )
        with tempfile.TemporaryDirectory(prefix="edumind-mineru-") as temporary:
            output = Path(temporary) / "output"
            command = [executable, "-p", str(request.source_path), "-o", str(output)]
            backend = str(request.options.get("mineru_backend", "vlm-transformers"))
            if backend:
                command.extend(["-b", backend])
            _run(
                command,
                request,
                environment={
                    "MINERU_TOOLS_CONFIG_JSON": str(config_path),
                    "MINERU_MODEL_SOURCE": "local",
                    "HF_HUB_OFFLINE": "1",
                },
            )
            return _read_markdown_outputs(output)

    def _olmocr(self, request: ExtractionRequest) -> list[str]:
        executable = _required_command("olmocr")
        model_path = Path(str(request.options.get("model_path", "")))
        if not model_path.exists():
            raise FileNotFoundError(
                "olmOCR weights are not prepared; run `python "
                "experiments/benchmarks/prepare.py extraction-models`"
            )
        with tempfile.TemporaryDirectory(prefix="edumind-olmocr-") as temporary:
            workspace = Path(temporary) / "workspace"
            command = [
                executable,
                str(workspace),
                "--markdown",
                "--model",
                str(model_path),
                "--workers",
                "1",
                "--pdfs",
                str(request.source_path),
            ]
            _run(command, request)
            return _read_markdown_outputs(workspace / "markdown")


def _markdown_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("markdown_texts", "markdown_text", "markdown"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    raise RuntimeError("Parser result did not expose Markdown text")


def _required_command(name: str) -> str:
    command = shutil.which(name)
    if command is None:
        raise MissingDependencyError(
            f"{name} is not installed in this environment; follow guide.md before running "
            "this candidate"
        )
    return command


def _run(
    command: list[str],
    request: ExtractionRequest,
    environment: Mapping[str, str] | None = None,
) -> None:
    timeout = int(request.options.get("timeout_seconds", 1800))
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **dict(environment or {})},
    )
    if process.returncode:
        details = (process.stderr or process.stdout).strip()[-4000:]
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {details}")


def _read_markdown_outputs(directory: Path) -> list[str]:
    files = sorted(directory.rglob("*.md")) if directory.exists() else []
    if not files:
        # Some parsers emit JSON whose useful field is Markdown.
        json_files = sorted(directory.rglob("*.json")) if directory.exists() else []
        pages: list[str] = []
        for path in json_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                pages.append(_markdown_text(payload))
            except RuntimeError:
                continue
        if pages:
            return pages
        raise RuntimeError(f"Parser created no Markdown output under {directory}")
    return [path.read_text(encoding="utf-8") for path in files]
