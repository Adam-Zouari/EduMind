"""Provisional production Whisper speech-to-text extractor."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document


class WhisperExtractor:
    supported_kinds = frozenset({SourceKind.AUDIO})
    engine = "whisper-small-en-control"
    name = engine
    model = "openai/whisper-small.en"

    def __init__(self, revision: str) -> None:
        self.revision = revision
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
                "candidate": self.name,
                "model": self.model,
                "engine_revision": request.profile.engine_revision,
                "alignment_execution": "native",
            },
            seconds=time.perf_counter() - started,
        )

    def _transcribe(
        self, request: ExtractionRequest
    ) -> tuple[list[str], list[tuple[float, float]]]:
        try:
            import torch
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Transformers ASR dependencies are required") from exc
        profile = request.profile
        assert profile is not None
        model_path = model_directory(request)
        if self._runtime is None:
            self._runtime = pipeline(
                "automatic-speech-recognition",
                model=str(model_path),
                device=0 if profile.device == "cuda" else -1,
                dtype=torch.float16 if profile.device == "cuda" else torch.float32,
                model_kwargs={"local_files_only": True},
            )
        result = self._runtime(str(request.source_path), return_timestamps="word")
        return transformers_chunks(result)


def model_directory(request: ExtractionRequest) -> Path:
    path = Path(str(request.options.get("model_path", ""))).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            "Pinned ASR weights are missing; run `python "
            "experiments/benchmarks/prepare.py app-models`."
        )
    return path


def transformers_chunks(result: object) -> tuple[list[str], list[tuple[float, float]]]:
    payload = result if isinstance(result, Mapping) else {}
    chunks = payload.get("chunks", [])
    texts: list[str] = []
    timestamps: list[tuple[float, float]] = []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            stamp = chunk.get("timestamp")
            if not isinstance(stamp, Sequence) or len(stamp) != 2:
                continue
            start = float(stamp[0] or 0.0)
            end = float(stamp[1] if stamp[1] is not None else start)
            texts.append(str(chunk.get("text", "")).strip())
            timestamps.append((start, end))
    if texts:
        return texts, timestamps
    return [str(payload.get("text", "")).strip()], [(0.0, 0.0)]
