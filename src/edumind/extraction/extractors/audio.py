"""Provisional production Whisper speech-to-text extractor."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, ExtractionWarning, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document


@dataclass(frozen=True)
class WhisperTranscript:
    text: str
    segments: tuple[Mapping[str, object], ...]


class WhisperExtractor:
    supported_kinds = frozenset({SourceKind.AUDIO})
    engine = "whisper-small-en-control"
    name = engine
    model = "openai/whisper-small.en"

    def __init__(self, _revision: str) -> None:
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
            separators=" ",
            metadata={
                "candidate": self.name,
                "model": self.model,
                "engine_revision": request.profile.engine_revision,
                "alignment_execution": "native",
            },
            warnings=(
                [
                    ExtractionWarning(
                        "incomplete_timestamp",
                        f"{sum(start is None or end is None for start, end in timestamps)} "
                        "Whisper chunk(s) lacked complete timestamp boundaries",
                    )
                ]
                if any(start is None or end is None for start, end in timestamps)
                else []
            ),
            seconds=time.perf_counter() - started,
        )

    def _transcribe(
        self, request: ExtractionRequest
    ) -> tuple[list[str], list[tuple[float | None, float | None]]]:
        profile = request.profile
        assert profile is not None
        model_path = model_directory(request)
        if self._runtime is None:
            self._runtime, _ = load_whisper_runtime(model_path, profile.device)
        transcript = transcribe_whisper(self._runtime, request.source_path)
        if transcript.segments:
            return (
                [str(segment["text"]) for segment in transcript.segments],
                [
                    (_optional_float(segment["start"]), _optional_float(segment["end"]))
                    for segment in transcript.segments
                ],
            )
        return [transcript.text], []


def model_directory(request: ExtractionRequest) -> Path:
    path = Path(str(request.options.get("model_path", ""))).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            "Pinned ASR weights are missing; run `python "
            "experiments/benchmarks/prepare.py app-models`."
        )
    return path


def load_whisper_runtime(model_path: Path, device: str) -> tuple[Any, str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Whisper device must be cpu or cuda")
    try:
        import torch
        from transformers import pipeline
    except ModuleNotFoundError as exc:
        raise MissingDependencyError("Transformers ASR dependencies are required") from exc
    dtype = torch.float16 if device == "cuda" else torch.float32
    runtime = pipeline(
        "automatic-speech-recognition",
        model=str(model_path),
        device=0 if device == "cuda" else -1,
        dtype=dtype,
        model_kwargs={"local_files_only": True},
    )
    _assert_whisper_device(runtime.model, device)
    return runtime, str(dtype).removeprefix("torch.")


def transcribe_whisper(runtime: Any, source: Path) -> WhisperTranscript:
    result = runtime(
        str(source),
        return_timestamps="word",
        generate_kwargs={"do_sample": False},
    )
    payload = result if isinstance(result, Mapping) else {}
    chunks = payload.get("chunks", [])
    segments: list[Mapping[str, object]] = []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            stamp = chunk.get("timestamp")
            start: float | None = None
            end: float | None = None
            if isinstance(stamp, Sequence) and not isinstance(stamp, (str, bytes)):
                if len(stamp) == 2:
                    start = _optional_float(stamp[0])
                    end = _optional_float(stamp[1])
            segments.append(
                {
                    "text": str(chunk.get("text", "")).strip(),
                    "start": start,
                    "end": end,
                }
            )
    return WhisperTranscript(str(payload.get("text", "")).strip(), tuple(segments))


def _assert_whisper_device(model: object, expected: str) -> None:
    try:
        observed = str(next(model.parameters()).device).split(":", 1)[0]  # type: ignore[attr-defined]
    except (AttributeError, StopIteration, TypeError) as exc:
        raise RuntimeError("Whisper model does not expose its device") from exc
    if observed != expected:
        raise RuntimeError(f"Whisper used {observed} instead of the requested {expected} device")


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
