"""Decode and transcribe video audio once in a fresh phase-level worker."""

from __future__ import annotations

import tempfile
import time
import wave
from pathlib import Path

from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import build_runtime
from experiments.benchmarks.extraction.audio.evaluate import align_sequences, normalize_transcript
from experiments.benchmarks.extraction.audio.protocol import (
    protocol_from_worker as audio_protocol_from_worker,
)
from experiments.benchmarks.extraction.media import decode_canonical_audio
from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.extraction.video.metrics import stitch_text
from experiments.benchmarks.extraction.video.protocol import protocol_from_worker


def execute(payload: dict[str, object]) -> dict[str, object]:
    protocol = protocol_from_worker(payload["protocol"])
    audio_protocol = audio_protocol_from_worker(payload["audio_protocol"])
    if float(payload["window_length_seconds"]) != protocol.window_length_seconds:
        raise ValueError("Frozen ASR window length differs from the video protocol")
    if float(payload["overlap_seconds"]) != protocol.overlap_seconds:
        raise ValueError("Frozen ASR overlap differs from the video protocol")
    device = str(payload["device"])
    runtime = build_runtime(
        str(payload["candidate"]),
        payload["model_lock"],  # type: ignore[arg-type]
        device,
        audio_protocol,
    )
    items = list(payload["items"])  # type: ignore[arg-type]
    length = float(payload["window_length_seconds"])
    overlap = float(payload["overlap_seconds"])
    maximum_overlap = int(payload["maximum_overlap_tokens"])
    monitor = ResourceMonitor(require_vram=device == "cuda", report_zero_vram=device == "cpu")
    videos: list[dict[str, object]] = []
    commands: list[dict[str, object]] = []
    total_latency = 0.0
    total_duration = 0.0
    substitutions = deletions = insertions = reference_words = 0
    with tempfile.TemporaryDirectory(prefix="edumind-video-asr-") as raw_temp:
        temporary = Path(raw_temp)
        try:
            with monitor:
                started = time.perf_counter()
                runtime.load()
                cold_load_seconds = time.perf_counter() - started
                decoded_audio: dict[str, Path] = {}
                first_id = str(items[0]["id"])
                first_audio = temporary / f"{first_id}-full.wav"
                first_command = decode_canonical_audio(
                    Path(str(items[0]["source_path"])),
                    first_audio,
                    sample_rate_hz=int(audio_protocol.audio["sample_rate_hz"]),
                    channels=int(audio_protocol.audio["channels"]),
                )
                decoded_audio[first_id] = first_audio
                commands.append(
                    {"phase": "decode", "sample_id": first_id, "command": first_command}
                )
                warmup_path = _slice_wav(
                    first_audio,
                    temporary / "warmup.wav",
                    start=0.0,
                    duration=min(length, float(items[0]["duration_seconds"])),
                    sample_rate_hz=int(audio_protocol.audio["sample_rate_hz"]),
                    channels=int(audio_protocol.audio["channels"]),
                    sample_width_bytes=int(audio_protocol.audio["sample_width_bytes"]),
                )
                for _ in range(int(payload["warmups"])):
                    runtime.transcribe(warmup_path)

                for item in items:
                    duration = float(item["duration_seconds"])
                    total_duration += duration
                    video_latency = 0.0
                    transcript = ""
                    segments: list[dict[str, object]] = []
                    window_rows = []
                    sample_id = str(item["id"])
                    if sample_id not in decoded_audio:
                        decoded = temporary / f"{sample_id}-full.wav"
                        command = decode_canonical_audio(
                            Path(str(item["source_path"])),
                            decoded,
                            sample_rate_hz=int(audio_protocol.audio["sample_rate_hz"]),
                            channels=int(audio_protocol.audio["channels"]),
                        )
                        decoded_audio[sample_id] = decoded
                        commands.append(
                            {
                                "phase": "decode",
                                "sample_id": sample_id,
                                "command": command,
                            }
                        )
                    for window_index, start in enumerate(_window_starts(duration, length, overlap)):
                        window_duration = min(length, duration - start)
                        window_path = _slice_wav(
                            decoded_audio[sample_id],
                            temporary / f"{item['id']}-{window_index}.wav",
                            start=start,
                            duration=window_duration,
                            sample_rate_hz=int(audio_protocol.audio["sample_rate_hz"]),
                            channels=int(audio_protocol.audio["channels"]),
                            sample_width_bytes=int(audio_protocol.audio["sample_width_bytes"]),
                        )
                        started = time.perf_counter()
                        output = runtime.transcribe(window_path)
                        latency = time.perf_counter() - started
                        if normalize_transcript(output.text) and not output.segments:
                            raise RuntimeError(
                                "Frozen video ASR produced lexical text without timestamps"
                            )
                        if not normalize_transcript(output.text) and output.segments:
                            raise RuntimeError(
                                "Frozen video ASR produced timestamps with an empty transcript"
                            )
                        total_latency += latency
                        video_latency += latency
                        shifted = [
                            {
                                "text": str(segment["text"]),
                                "start": float(segment["start"]) + start,
                                "end": min(
                                    duration,
                                    start + window_duration,
                                    float(segment["end"]) + start,
                                ),
                            }
                            for segment in output.segments
                        ]
                        transcript = stitch_text(
                            transcript,
                            output.text,
                            maximum_overlap_tokens=maximum_overlap,
                        )
                        segments = _stitch_segments(segments, shifted, maximum_overlap)
                        window_rows.append(
                            {
                                "index": window_index,
                                "start": start,
                                "end": start + window_duration,
                                "latency_seconds": latency,
                                "transcript": output.text,
                                "warnings": list(output.warnings),
                            }
                        )
                    expected = normalize_transcript(str(item.get("reference_transcript", "")))
                    observed = normalize_transcript(transcript)
                    alignment = align_sequences(expected.split(), observed.split())
                    reference_words += len(expected.split())
                    substitutions += alignment.substitutions
                    deletions += alignment.deletions
                    insertions += alignment.insertions
                    videos.append(
                        {
                            "sample_id": str(item["id"]),
                            "duration_seconds": duration,
                            "latency_seconds": video_latency,
                            "real_time_factor": video_latency / duration,
                            "transcript": transcript,
                            "segments": segments,
                            "windows": window_rows,
                            "reference_word_count": len(expected.split()),
                            "word_substitutions": alignment.substitutions,
                            "word_deletions": alignment.deletions,
                            "word_insertions": alignment.insertions,
                        }
                    )
        finally:
            runtime.close()
            resources = monitor.metrics()
    return {
        "videos": videos,
        "ffmpeg_commands": commands,
        "metrics": {
            "word_error_rate": (
                (substitutions + deletions + insertions) / reference_words
                if reference_words
                else None
            ),
            "real_time_factor": total_latency / total_duration,
            "total_latency_seconds": total_latency,
            "cold_model_load_seconds": cold_load_seconds,
            "peak_process_tree_ram_mb": float(resources["peak_process_tree_ram_mb"]),
            "peak_vram_mb": 0.0 if device == "cpu" else float(resources["peak_vram_mb"]),
        },
        "parameters": {
            **runtime.parameters(),
            "vram_measurement_method": monitor.vram_measurement_method,
        },
    }


def _window_starts(duration: float, length: float, overlap: float):
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    step = length - overlap
    starts = []
    current = 0.0
    while current < duration:
        starts.append(current)
        if current + length >= duration:
            break
        current += step
    return starts


def _slice_wav(
    source: Path,
    destination: Path,
    *,
    start: float,
    duration: float,
    sample_rate_hz: int,
    channels: int,
    sample_width_bytes: int,
):
    """Create deterministic PCM windows from the one decoded per-video WAV."""

    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != channels
            or reader.getframerate() != sample_rate_hz
            or reader.getsampwidth() != sample_width_bytes
        ):
            raise ValueError("Decoded video audio is not mono 16 kHz PCM16")
        start_frame = min(reader.getnframes(), round(start * reader.getframerate()))
        frame_count = round(duration * reader.getframerate())
        reader.setpos(start_frame)
        frames = reader.readframes(frame_count)
        parameters = reader.getparams()
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(parameters)
        writer.writeframes(frames)
    return destination


def _stitch_segments(existing, current, maximum_overlap):
    left_units = _timestamp_units(existing)
    right_units = _timestamp_units(current)
    left = [str(item["text"]) for item in left_units]
    right = [str(item["text"]) for item in right_units]
    limit = min(len(left), len(right), maximum_overlap)
    overlap = next(
        (width for width in range(limit, 0, -1) if left[-width:] == right[:width]),
        0,
    )
    return [*left_units, *right_units[overlap:]]


def _timestamp_units(segments):
    """Project timestamp segments to normalized tokens for token-counted stitching."""

    return [
        {"text": token, "start": float(item["start"]), "end": float(item["end"])}
        for item in segments
        for token in normalize_transcript(str(item["text"])).split()
    ]


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
