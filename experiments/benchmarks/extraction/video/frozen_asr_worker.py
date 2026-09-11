"""Decode and transcribe video audio once in a fresh phase-level worker."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from edumind.common.artifacts import atomic_write_json
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import build_runtime
from experiments.benchmarks.extraction.audio.evaluate import align_sequences, normalize_transcript
from experiments.benchmarks.extraction.video.metrics import stitch_text


def execute(payload: dict[str, object]) -> dict[str, object]:
    device = str(payload["device"])
    runtime = build_runtime(
        str(payload["candidate"]),
        payload["model_lock"],  # type: ignore[arg-type]
        device,
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
                first_audio, first_command = _decode_audio(
                    Path(str(items[0]["source_path"])), temporary / f"{first_id}-full.wav"
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
                        decoded, command = _decode_audio(
                            Path(str(item["source_path"])),
                            temporary / f"{sample_id}-full.wav",
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


def _decode_audio(source: Path, destination: Path):
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
    return destination, command


def _slice_wav(source: Path, destination: Path, *, start: float, duration: float):
    """Create deterministic PCM windows from the one decoded per-video WAV."""

    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16_000
            or reader.getsampwidth() != 2
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


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: frozen_asr_worker.py PAYLOAD_JSON RESULT_JSON")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    atomic_write_json(Path(sys.argv[2]), execute(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
