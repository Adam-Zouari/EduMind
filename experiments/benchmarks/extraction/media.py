"""Shared FFmpeg and canonical-audio helpers for extraction benchmarks."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path


def ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"], check=True, capture_output=True, text=True
    )
    return completed.stdout.splitlines()[0].strip()


def media_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def decode_canonical_audio(source: Path, destination: Path) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return command


def canonical_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if (
            audio.getnchannels() != 1
            or audio.getframerate() != 16_000
            or audio.getsampwidth() != 2
        ):
            raise ValueError(
                f"FFmpeg did not produce canonical mono 16 kHz PCM audio: {path}"
            )
        return audio.getnframes() / audio.getframerate()
