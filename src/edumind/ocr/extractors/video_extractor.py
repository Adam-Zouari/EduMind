"""Video extraction using FFmpeg and Whisper."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import ffmpeg

from ..config import WHISPER_MODEL
from ..core.base_extractor import BaseExtractor, ExtractionResult

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WHISPER_AVAILABLE = False
try:
    import torch
    import whisper

    torch.set_num_threads(1)
    WHISPER_AVAILABLE = True
except Exception:
    whisper = None


class VideoExtractor(BaseExtractor):
    """Extract text from video files."""

    _whisper_model: object | None = None

    def __init__(self, model_name: str = WHISPER_MODEL):
        super().__init__()
        self.model = None

        if not WHISPER_AVAILABLE or whisper is None:
            self.logger.error("Whisper is not available. Install optional video dependencies.")
            return

        if VideoExtractor._whisper_model is None:
            self.logger.info(f"Loading Whisper model for video: {model_name}")
            try:
                VideoExtractor._whisper_model = whisper.load_model(model_name, device="cpu")
            except Exception as exc:
                self.logger.error(f"Failed to load Whisper model: {exc}")
                return

        self.model = VideoExtractor._whisper_model

    def extract(self, file_path: Path, **kwargs: Any) -> ExtractionResult:
        start_time = time.time()
        self.logger.info(f"Extracting from video: {file_path}")

        if self.model is None:
            return self._create_error_result(
                file_path,
                "Whisper model not available. Install optional video dependencies.",
            )

        try:
            audio_path = self._extract_audio(file_path)
            result = self.model.transcribe(str(audio_path), verbose=False)
            audio_path.unlink(missing_ok=True)

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
                    "model": WHISPER_MODEL,
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

    def _extract_audio(self, video_path: Path) -> Path:
        audio_path = Path(tempfile.mktemp(suffix=".wav"))
        try:
            (
                ffmpeg.input(str(video_path))
                .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as exc:
            self.logger.error(f"FFmpeg error: {exc.stderr.decode()}")
            raise
        return audio_path

    def _get_video_info(self, video_path: Path) -> dict[str, Any]:
        try:
            probe = ffmpeg.probe(str(video_path))
            video_stream = next((stream for stream in probe["streams"] if stream["codec_type"] == "video"), None)
            if video_stream:
                return {
                    "width": video_stream.get("width"),
                    "height": video_stream.get("height"),
                    "codec": video_stream.get("codec_name"),
                    "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                    "duration": float(probe["format"].get("duration", 0)),
                }
        except Exception as exc:
            self.logger.warning(f"Could not extract video info: {exc}")
        return {}
