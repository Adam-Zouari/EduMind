"""Video extraction using deterministic FFmpeg audio and keyframe policies."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from ..contracts import ExtractedDocument, ExtractedSegment, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError
from .audio import AudioExtractor
from .base import build_document
from .image import ImageExtractor


class VideoExtractor:
    supported_kinds = frozenset({SourceKind.VIDEO})

    def __init__(self, keyframes: str = "hybrid") -> None:
        self.keyframes = keyframes
        self.name = f"video-{keyframes}"
        self.revision = "ffmpeg-system"
        self._audio_extractors: dict[tuple[str, str, str], AudioExtractor] = {}
        self._image_extractors: dict[tuple[str, str], ImageExtractor] = {}

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
                audio_engine = str(request.options.get("audio_engine", "faster-whisper"))
                audio_model = str(request.options.get("audio_model", "base.en"))
                audio_compute_type = str(request.options.get("audio_compute_type", "int8"))
                audio_profile = replace(
                    request.profile,
                    engine=f"{audio_engine}-{audio_model}-{audio_compute_type}",
                    routing="video-audio",
                )
                audio_request = ExtractionRequest.from_path(
                    audio_path,
                    source_kind=SourceKind.AUDIO,
                    profile=audio_profile,
                    options=request.options,
                )
                audio_key = (audio_engine, audio_model, audio_compute_type)
                if audio_key not in self._audio_extractors:
                    self._audio_extractors[audio_key] = AudioExtractor(*audio_key)
                audio_extractor = self._audio_extractors[audio_key]
                audio = audio_extractor.extract(audio_request, SourceKind.AUDIO)
                frames = self._extract_frames(request.source_path, directory)
                visual_segments: list[tuple[str, float | None]] = []
                image_engine = str(request.options.get("image_engine", "tesseract-5"))
                image_revision = str(request.options.get("image_revision", "5"))
                image_profile = replace(
                    request.profile, engine=image_engine, routing="video-keyframe"
                )
                image_key = (image_engine, image_revision)
                if image_key not in self._image_extractors:
                    self._image_extractors[image_key] = ImageExtractor(*image_key)
                image_extractor = self._image_extractors[image_key]
                for frame, timestamp in frames:
                    image_request = ExtractionRequest.from_path(
                        frame,
                        source_kind=SourceKind.IMAGE,
                        profile=image_profile,
                        options=request.options,
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
        result = build_document(
            request,
            kind,
            request.profile,
            texts,
            timestamps=timestamps,
            metadata={
                "keyframe_policy": self.keyframes,
                "audio_engine": f"{audio_engine}-{audio_model}-{audio_compute_type}",
                "image_engine": image_engine,
            },
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
