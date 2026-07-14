"""Lazy local speech-to-text implementations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document


class AudioExtractor:
    supported_kinds = frozenset({SourceKind.AUDIO})

    def __init__(self, engine: str, model: str, compute_type: str = "int8") -> None:
        self.engine = engine
        self.model = model
        self.compute_type = compute_type
        self.name = f"{engine}-{model}-{compute_type}"
        self.revision = "unpinned"
        self._runtime: Any | None = None

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            texts, timestamps = self._transcribe(request)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Audio extraction failed with {self.name}", detail=str(exc)
            ) from exc
        return build_document(
            request,
            kind,
            request.profile,
            texts,
            timestamps=timestamps,
            metadata={
                "engine": self.engine,
                "model": self.model,
                "compute_type": self.compute_type,
                "engine_revision": request.profile.engine_revision,
            },
            seconds=time.perf_counter() - started,
        )

    def _transcribe(
        self, request: ExtractionRequest
    ) -> tuple[list[str], list[tuple[float, float]]]:
        profile = request.profile
        if profile is None:
            raise ValueError("Resolved extraction profile is required")
        if self.engine == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as exc:
                raise MissingDependencyError("faster-whisper is required; install .[asr]") from exc
            configured_path = request.options.get("model_path")
            model_path = Path(str(configured_path)).expanduser() if configured_path else None
            if model_path is None or not model_path.is_dir():
                raise FileNotFoundError(
                    "faster-whisper weights are not prepared locally; run `edumind benchmark "
                    "prepare extraction-models` and use its model lock"
                )
            if self._runtime is None:
                self._runtime = WhisperModel(
                    str(model_path),
                    device=profile.device,
                    compute_type=self.compute_type,
                    local_files_only=True,
                )
            segments, _ = self._runtime.transcribe(
                str(request.source_path),
                language="en",
                vad_filter=bool(request.options.get("vad", False)),
            )
            resolved = list(segments)
            return [segment.text.strip() for segment in resolved], [
                (float(segment.start), float(segment.end)) for segment in resolved
            ]
        if self.engine == "openai-whisper":
            try:
                import whisper
            except ModuleNotFoundError as exc:
                raise MissingDependencyError("openai-whisper is required; install .[asr]") from exc
            configured_path = request.options.get("model_path")
            model_path = Path(str(configured_path)).expanduser() if configured_path else None
            if model_path is None or not model_path.is_file():
                raise FileNotFoundError(
                    "OpenAI Whisper weights are not prepared locally; set model_path from the "
                    "model lock created by the preparation command"
                )
            if self._runtime is None:
                self._runtime = whisper.load_model(str(model_path), device=profile.device)
            result = self._runtime.transcribe(str(request.source_path), language="en")
            raw_segments = result.get("segments", [])
            return [str(item.get("text", "")).strip() for item in raw_segments], [
                (float(item.get("start", 0.0)), float(item.get("end", 0.0)))
                for item in raw_segments
            ]
        raise ValueError(f"Unknown ASR engine: {self.engine}")
