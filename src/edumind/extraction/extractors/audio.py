"""Provisional production Whisper speech-to-text extractor."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from edumind.common.gpu import release_runtime_memory

from ..contracts import (
    ExtractedDocument,
    ExtractionRequest,
    ExtractionWarning,
    SegmentKind,
    SourceKind,
)
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document


@dataclass(frozen=True)
class WhisperTranscript:
    text: str
    segments: tuple[Mapping[str, object], ...]
    warnings: tuple[ExtractionWarning, ...] = ()
    native_segments: tuple[Mapping[str, object], ...] = ()


class WhisperExtractor:
    supported_kinds = frozenset({SourceKind.AUDIO})
    engine = "whisper-small-en-control"
    name = engine
    model = "openai/whisper-small.en"

    def __init__(self, revision: str) -> None:
        self.revision = revision
        self._runtime: Any | None = None
        self._runtime_key: tuple[Path, str, object] | None = None

    def extract(
        self, request: ExtractionRequest, kind: SourceKind
    ) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            transcript = self._transcribe(request)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Audio extraction failed with {self.name}", detail=str(exc)
            ) from exc
        texts = [str(segment["text"]) for segment in transcript.segments]
        timestamps = [
            (_optional_float(segment["start"]), _optional_float(segment["end"]))
            for segment in transcript.segments
        ]
        document = build_document(
            request,
            kind,
            request.profile,
            texts or ([transcript.text] if transcript.text else []),
            timestamps=timestamps,
            separators=" ",
            metadata={
                "candidate": self.name,
                "model": self.model,
                "engine_revision": request.profile.engine_revision,
                "alignment_execution": "native",
                "native_transcript": transcript.text,
                "native_segments": list(transcript.native_segments),
            },
            warnings=list(transcript.warnings)
            + (
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
        return replace(
            document,
            segments=tuple(
                replace(segment, kind=SegmentKind.AUDIO)
                for segment in document.segments
            ),
        )

    def _transcribe(self, request: ExtractionRequest) -> WhisperTranscript:
        profile = request.profile
        assert profile is not None
        model_path = model_directory(request)
        key = (
            model_path.resolve(),
            profile.device,
            request.options.get("model_bundle_identity"),
        )
        if self._runtime_key != key or self._runtime is None:
            if self._runtime is not None:
                self.unload()
            self._runtime, _ = load_whisper_runtime(model_path, profile.device)
            self._runtime_key = key
        return transcribe_whisper(self._runtime, request.source_path)

    def unload(self) -> None:
        """Release ASR before the independent video visual stage begins."""
        self._runtime = None
        self._runtime_key = None
        release_runtime_memory()


def model_directory(request: ExtractionRequest) -> Path:
    raw_path = request.options.get("model_path")
    path = Path(str(raw_path)).expanduser() if raw_path else None
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            "Pinned ASR weights are missing; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    return path


def load_whisper_runtime(
    model_path: Path, device: str, *, dtype: str | None = None
) -> tuple[Any, str]:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Whisper device must be cpu or cuda")
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Transformers ASR dependencies are required"
        ) from exc
    torch_dtype = (
        getattr(torch, dtype)
        if dtype is not None
        else (torch.float16 if device == "cuda" else torch.float32)
    )
    model = (
        AutoModelForSpeechSeq2Seq.from_pretrained(
            str(model_path), dtype=torch_dtype, local_files_only=True
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    runtime = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=torch.device(device),
    )
    _assert_whisper_device(runtime.model, device)
    return runtime, str(torch_dtype).removeprefix("torch.")


def transcribe_whisper(
    runtime: Any,
    source: Path,
    *,
    return_timestamps: str = "word",
    do_sample: bool = False,
) -> WhisperTranscript:
    result = runtime(
        str(source),
        return_timestamps=return_timestamps,
        generate_kwargs={"do_sample": do_sample},
    )
    if not isinstance(result, Mapping) or not isinstance(result.get("text"), str):
        raise ValueError("Whisper response must be an object with string text")
    payload = result
    text = payload["text"].strip()
    chunks = payload.get("chunks", [])
    segments: list[Mapping[str, object]] = []
    warnings: list[ExtractionWarning] = []
    if isinstance(chunks, Sequence) and not isinstance(chunks, (str, bytes)):
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping) or not isinstance(chunk.get("text"), str):
                warnings.append(
                    ExtractionWarning(
                        "malformed_chunk", f"Whisper chunk {index} lacked string text"
                    )
                )
                continue
            stamp = chunk.get("timestamp")
            start: float | None = None
            end: float | None = None
            if (
                isinstance(stamp, Sequence)
                and not isinstance(stamp, (str, bytes))
                and len(stamp) == 2
            ):
                try:
                    start = _optional_float(stamp[0])
                    end = _optional_float(stamp[1])
                    if any(
                        value is not None and (not math.isfinite(value) or value < 0)
                        for value in (start, end)
                    ) or (start is not None and end is not None and end < start):
                        raise ValueError("Invalid timestamp range")
                except (ValueError, TypeError, OverflowError):
                    warnings.append(
                        ExtractionWarning(
                            "malformed_timestamp",
                            f"Whisper chunk {index} had invalid boundaries; retained as untimed text",
                        )
                    )
                    start, end = None, None
            elif stamp is not None:
                warnings.append(
                    ExtractionWarning(
                        "malformed_timestamp",
                        f"Whisper chunk {index} had malformed boundaries; retained as untimed text",
                    )
                )
            segments.append(
                {
                    "text": chunk["text"].strip(),
                    "start": start,
                    "end": end,
                }
            )
    else:
        warnings.append(
            ExtractionWarning(
                "malformed_chunks",
                "Whisper chunks were not a sequence; retained native transcript without timestamps",
            )
        )
    native_segments = tuple(segments)
    lexical = lambda value: re.findall(r"\w+", value.casefold())
    chunk_text = " ".join(str(segment["text"]) for segment in segments)
    if lexical(text) != lexical(chunk_text):
        warnings.append(
            ExtractionWarning(
                "transcript_chunk_mismatch",
                "Whisper chunk text did not cover the native transcript; retained the complete native transcript without inferred timing",
            )
        )
        # Keep both conflicting representations; never silently replace either one.
        segments = [{"text": text, "start": None, "end": None}] if text else []
    elif text and not segments:
        segments = [{"text": text, "start": None, "end": None}]
    return WhisperTranscript(text, tuple(segments), tuple(warnings), native_segments)


def _assert_whisper_device(model: object, expected: str) -> None:
    try:
        observed = str(next(model.parameters()).device).split(":", 1)[0]  # type: ignore[attr-defined]
    except (AttributeError, StopIteration, TypeError) as exc:
        raise RuntimeError("Whisper model does not expose its device") from exc
    if observed != expected:
        raise RuntimeError(
            f"Whisper used {observed} instead of the requested {expected} device"
        )


def _optional_float(value: object) -> float | None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError("Whisper boundaries must be numbers or null")
    return None if value is None else float(value)
