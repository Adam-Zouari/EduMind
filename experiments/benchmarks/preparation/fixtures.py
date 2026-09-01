"""Generate deterministic multimodal smoke fixtures."""

from __future__ import annotations

import json
import random
import subprocess
import wave
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from edumind.common.artifacts import atomic_write_json, sha256_file

from experiments.benchmarks.common.datasets import manifest_content_checksum


def prepare_smoke_fixtures(root: Path, *, modality: str = "all") -> Path:
    """Generate tiny valid files; no fake extractor is used by smoke runs."""
    kinds = {
        "all": {"image", "pdf", "docx", "audio", "video"},
        "document": {"image", "pdf", "docx"},
        "audio": {"audio"},
        "video": {"video"},
    }
    if modality not in kinds:
        raise ValueError(f"Unknown smoke-fixture modality: {modality}")
    if modality in {"all", "audio", "video"}:
        _require_ffmpeg_flite()
    manifest_path = root / "data/benchmarks/extraction/smoke.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        kind = sample["kind"]
        if kind not in kinds[modality]:
            continue
        destination = root / sample["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = sample["reference"]
        if kind == "image":
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (1400, 220), "white")
            ImageDraw.Draw(image).text((40, 80), text, fill="black")
            image.save(destination)
        elif kind == "pdf":
            import fitz

            document = fitz.open()
            page = document.new_page(width=595, height=842)
            page.insert_text((72, 100), text, fontsize=16)
            document.save(destination, no_new_id=True)
            document.close()
        elif kind == "docx":
            _write_minimal_docx(destination, text)
        elif kind == "audio":
            _synthesize_speech(text, destination)
            duration = _wav_duration(destination)
            sample["duration_seconds"] = duration
            sample["reference_segments"] = [
                {"text": text, "start": 0.0, "end": duration}
            ]
        elif kind == "video":
            from PIL import Image, ImageDraw

            temporary_image = destination.with_suffix(".png")
            temporary_audio = destination.with_suffix(".wav")
            image = Image.new("RGB", (1280, 720), "white")
            ImageDraw.Draw(image).text((80, 320), text, fill="black")
            image.save(temporary_image)
            _synthesize_speech(text, temporary_audio)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loop",
                        "1",
                        "-i",
                        str(temporary_image),
                        "-i",
                        str(temporary_audio),
                        "-c:v",
                        "libx264",
                        "-tune",
                        "stillimage",
                        "-c:a",
                        "aac",
                        "-pix_fmt",
                        "yuv420p",
                        "-shortest",
                        str(destination),
                    ],
                    check=True,
                    capture_output=True,
                )
            finally:
                temporary_image.unlink(missing_ok=True)
                temporary_audio.unlink(missing_ok=True)
        sample["asset_sha256"] = sha256_file(destination)
    payload["checksum"] = manifest_content_checksum(payload["samples"])
    atomic_write_json(manifest_path, payload)
    if modality in {"all", "audio"}:
        _prepare_audio_reliability(root)
    return manifest_path


def _write_minimal_docx(destination: Path, text: str) -> None:
    from xml.sax.saxutils import escape

    paragraphs = "".join(
        (
            '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>'
            if index == 0 and line == "Heading"
            else f'<w:p><w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>'
        )
        for index, line in enumerate(text.splitlines())
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}<w:sectPr/></w:body></w:document>"
    )
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        _write_deterministic_zip_member(archive, "[Content_Types].xml", content_types)
        _write_deterministic_zip_member(archive, "_rels/.rels", relationships)
        _write_deterministic_zip_member(archive, "word/document.xml", document)


def _write_deterministic_zip_member(archive: ZipFile, name: str, content: str) -> None:
    member = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED
    member.create_system = 0
    archive.writestr(member, content.encode("utf-8"))


def _require_ffmpeg_flite() -> None:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("FFmpeg is required to regenerate audio/video smoke fixtures") from exc
    if " flite " not in completed.stdout:
        raise RuntimeError(
            "This FFmpeg build lacks the optional flite filter required to regenerate "
            "speech fixtures. The committed WAV/MP4 fixtures can still be used directly."
        )


def _synthesize_speech(text: str, destination: Path) -> None:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"flite=text='{escaped}':voice=slt",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(destination),
        ],
        check=True,
    )


def _prepare_audio_reliability(root: Path) -> None:
    manifest_path = root / "data/benchmarks/extraction/audio-reliability-smoke.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        destination = root / sample["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_pcm(
            destination,
            noise=sample["nonspeech_kind"] == "background_noise",
        )
        sample["duration_seconds"] = _wav_duration(destination)
        sample["asset_sha256"] = sha256_file(destination)
    payload["checksum"] = manifest_content_checksum(payload["samples"])
    atomic_write_json(manifest_path, payload)


def _write_pcm(destination: Path, *, noise: bool) -> None:
    randomizer = random.Random(42)
    frames = bytearray()
    for _ in range(16_000 * 2):
        sample = randomizer.randint(-1800, 1800) if noise else 0
        frames.extend(int(sample).to_bytes(2, "little", signed=True))
    with wave.open(str(destination), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(frames)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()
