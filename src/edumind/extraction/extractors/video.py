"""Video extraction using deterministic FFmpeg audio and keyframe policies."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..contracts import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionRequest,
    Extractor,
    SourceKind,
)
from ..errors import ExtractionBackendError
from .audio import WhisperExtractor
from .base import build_document
from .document import DoclingExtractor


class VideoExtractor:
    supported_kinds = frozenset({SourceKind.VIDEO})

    def __init__(
        self,
        keyframes: str = "hybrid",
        *,
        audio_factory: Callable[[str, str], Extractor] | None = None,
        image_factory: Callable[[str, str], Extractor] | None = None,
    ) -> None:
        self.keyframes = keyframes
        self.name = f"video-{keyframes}"
        self.revision = "ffmpeg-system"
        self._audio_factory = audio_factory or _production_audio_factory
        self._image_factory = image_factory or _production_image_factory
        self._audio_extractors: dict[str, Extractor] = {}
        self._image_extractors: dict[str, Extractor] = {}

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="edumind-video-") as temp:
                directory = Path(temp)
                audio_path = directory / "audio.wav"
                self._run_ffmpeg(
                    [
                        "-i",
                        str(request.source_path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(audio_path),
                    ]
                )
                audio_engine = str(
                    request.options.get("audio_candidate", "whisper-small-en-control")
                )
                audio_revision = str(request.options.get("audio_revision", "from-lock"))
                audio_profile = replace(
                    request.profile,
                    engine=audio_engine,
                    engine_revision=audio_revision,
                    routing="video-audio",
                )
                audio_options = {
                    **request.options,
                    **{
                        key.removeprefix("audio_"): value
                        for key, value in request.options.items()
                        if key.startswith("audio_")
                    },
                }
                audio_request = ExtractionRequest.from_path(
                    audio_path,
                    source_kind=SourceKind.AUDIO,
                    profile=audio_profile,
                    options=audio_options,
                )
                if audio_engine not in self._audio_extractors:
                    self._audio_extractors[audio_engine] = self._audio_factory(
                        audio_engine, audio_revision
                    )
                audio_extractor = self._audio_extractors[audio_engine]
                audio = audio_extractor.extract(audio_request, SourceKind.AUDIO)
                frames = self._extract_frames(request.source_path, directory)
                visual_segments: list[tuple[str, float | None]] = []
                image_engine = str(request.options.get("image_engine", "docling-standard"))
                image_revision = str(request.options.get("image_revision", "from-lock"))
                image_profile = replace(
                    request.profile,
                    engine=image_engine,
                    engine_revision=image_revision,
                    preprocessing=str(
                        request.options.get("image_preprocessing", request.profile.preprocessing)
                    ),
                    routing="video-keyframe",
                )
                if image_engine not in self._image_extractors:
                    self._image_extractors[image_engine] = self._image_factory(
                        image_engine, image_revision
                    )
                image_extractor = self._image_extractors[image_engine]
                image_options = {
                    **request.options,
                    **{
                        key.removeprefix("image_"): value
                        for key, value in request.options.items()
                        if key.startswith("image_")
                    },
                }
                for frame, timestamp in frames:
                    image_request = ExtractionRequest.from_path(
                        frame,
                        source_kind=SourceKind.IMAGE,
                        profile=image_profile,
                        options=image_options,
                    )
                    text = image_extractor.extract(image_request, SourceKind.IMAGE).text.strip()
                    if text:
                        visual_segments.append((text, timestamp))
        except Exception as exc:
            raise ExtractionBackendError("Video extraction failed", detail=str(exc)) from exc

        texts = [segment.text for segment in audio.segments] + [item[0] for item in visual_segments]
        timestamps = [
            (segment.timestamp_start, segment.timestamp_end) for segment in audio.segments
        ] + [(timestamp, timestamp) for _, timestamp in visual_segments]
        separators = [
            " " if index < len(audio.segments) else "\n"
            for index in range(1, len(texts))
        ]
        result = build_document(
            request,
            kind,
            request.profile,
            texts,
            timestamps=timestamps,
            separators=separators,
            metadata={
                "keyframe_policy": self.keyframes,
                "audio_engine": audio_engine,
                "image_engine": image_engine,
                "audio_segment_count": len(audio.segments),
                "visual_segment_count": len(visual_segments),
            },
            warnings=list(audio.warnings),
            seconds=time.perf_counter() - started,
        )
        # The final document owns new offsets; local audio offsets are intentionally rebuilt.
        assert all(isinstance(segment, ExtractedSegment) for segment in result.segments)
        return result

    def _extract_frames(self, source: Path, directory: Path) -> list[tuple[Path, float]]:
        pattern = directory / "frame-%05d.png"
        if self.keyframes == "fixed":
            video_filter = "fps=1/10"
        elif self.keyframes == "scene":
            video_filter = "select='gt(scene,0.35)'"
        else:
            video_filter = (
                "select='gt(scene,0.35)+isnan(prev_selected_t)+gte(t-prev_selected_t,10)'"
            )
        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "info",
                "-y",
                "-i",
                str(source),
                "-vf",
                f"{video_filter},showinfo",
                "-vsync",
                "vfr",
                str(pattern),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        timestamps = [
            float(value)
            for value in re.findall(r"showinfo.*?pts_time:([0-9]+(?:\.[0-9]+)?)", process.stderr)
        ]
        frames = sorted(directory.glob("frame-*.png"))
        if len(timestamps) != len(frames):
            raise RuntimeError("FFmpeg keyframe timestamps did not match extracted frames")
        return list(zip(frames, timestamps, strict=True))

    @staticmethod
    def _run_ffmpeg(arguments: list[str]) -> None:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments], check=True
        )


def _production_audio_factory(engine: str, revision: str) -> Extractor:
    if engine != "whisper-small-en-control":
        raise ValueError(f"Audio candidate is not promoted to production: {engine}")
    return WhisperExtractor(revision)


def _production_image_factory(engine: str, revision: str) -> Extractor:
    if engine != "docling-standard":
        raise ValueError(f"Document parser is not promoted to production: {engine}")
    return DoclingExtractor(revision)
