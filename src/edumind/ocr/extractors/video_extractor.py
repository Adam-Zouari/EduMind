"""Video extraction using FFmpeg and Whisper."""

from __future__ import annotations

import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import ffmpeg

from ..config import TEMP_DIR, WHISPER_LANGUAGE, WHISPER_MODEL
from ..core.base_extractor import BaseExtractor, ExtractionResult
from ._media_runtime import get_whisper_device, load_whisper_model


class VideoExtractor(BaseExtractor):
    """Extract text from video files."""

    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.model_name = model_name
        self.device = get_whisper_device()
        self.model, self.runtime_error = load_whisper_model(model_name)
        if self.model is None and self.runtime_error:
            self.logger.error(self.runtime_error)

    def extract(self, file_path: Path, **kwargs: object) -> ExtractionResult:
        start_time = time.time()
        self.logger.info(f"Extracting from video: {file_path}")

        if self.model is None:
            return self._create_error_result(
                file_path,
                (
                    self.runtime_error
                    or "Whisper model not available. Install optional video dependencies."
                ),
            )

        audio_path: Path | None = None
        try:
            audio_path = self._extract_audio(file_path)
            transcribe_kwargs: dict[str, object] = {"verbose": False}
            language = kwargs.get("language")
            raw_languages = kwargs.get("languages")
            if language is None and isinstance(raw_languages, list) and raw_languages:
                language = raw_languages[0]
            if language is None:
                language = WHISPER_LANGUAGE
            if isinstance(language, str) and language:
                transcribe_kwargs["language"] = language

            result = cast(Any, self.model).transcribe(str(audio_path), **transcribe_kwargs)
            segments = [
                {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
                for segment in result.get("segments", [])
            ]

            return ExtractionResult(
                text=result["text"],
                metadata={
                    "language": result.get("language", "unknown"),
                    "duration": result.get("duration", 0),
                    "num_segments": len(segments),
                    "segments": segments[:10],
                    "video_info": self._get_video_info(file_path),
                    "model": self.model_name,
                    "device": self.device,
                    "extractor": "video",
                },
                format_type="video",
                file_path=str(file_path),
                extraction_time=time.time() - start_time,
                success=True,
            )
        except Exception as exc:
            self.logger.error(f"Video extraction failed: {exc}")
            return self._create_error_result(file_path, str(exc))
        finally:
            if audio_path is not None:
                audio_path.unlink(missing_ok=True)

    def _extract_audio(self, video_path: Path) -> Path:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            dir=str(TEMP_DIR),
        ) as temp_file:
            audio_path = Path(temp_file.name)

        try:
            (
                ffmpeg.input(str(video_path))
                .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            audio_path.unlink(missing_ok=True)
            stderr = exc.stderr.decode(errors="ignore") if exc.stderr else str(exc)
            self.logger.error(f"FFmpeg error: {stderr}")
            raise RuntimeError(stderr) from exc

        return audio_path

    def _get_video_info(self, video_path: Path) -> dict[str, object]:
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
                None,
            )
            if video_stream:
                return {
                    "width": video_stream.get("width"),
                    "height": video_stream.get("height"),
                    "codec": video_stream.get("codec_name"),
                    "fps": self._parse_frame_rate(video_stream.get("r_frame_rate", "0/1")),
                    "duration": float(probe["format"].get("duration", 0)),
                }
        except Exception as exc:
            self.logger.warning(f"Could not extract video info: {exc}")
        return {}

    @staticmethod
    def _parse_frame_rate(value: str) -> float:
        """Parse FFmpeg frame-rate strings without using eval()."""
        try:
            return float(Fraction(value))
        except (ArithmeticError, ValueError, ZeroDivisionError):
            return 0.0
