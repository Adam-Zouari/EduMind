"""Generate deterministic multimodal smoke fixtures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file

from experiments.benchmarks.common.datasets import manifest_content_checksum


def prepare_smoke_fixtures(root: Path) -> Path:
    """Generate tiny valid files; no fake extractor is used by smoke runs."""
    import fitz
    from PIL import Image, ImageDraw

    manifest_path = root / "data/benchmarks/extraction/smoke.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        kind = sample["kind"]
        destination = root / sample["source_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = sample["reference"]
        if kind == "image":
            image = Image.new("RGB", (1400, 220), "white")
            ImageDraw.Draw(image).text((40, 80), text, fill="black")
            image.save(destination)
        elif kind == "pdf":
            document = fitz.open()
            page = document.new_page(width=595, height=842)
            page.insert_text((72, 100), text, fontsize=16)
            document.save(destination)
            document.close()
        elif kind == "docx":
            _write_minimal_docx(destination, text)
        elif kind == "audio":
            _windows_speech(text, destination)
        elif kind == "video":
            temporary_image = destination.with_suffix(".png")
            temporary_audio = destination.with_suffix(".wav")
            image = Image.new("RGB", (1280, 720), "white")
            ImageDraw.Draw(image).text((80, 320), text, fill="black")
            image.save(temporary_image)
            _windows_speech(text, temporary_audio)
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
    return manifest_path


def _write_minimal_docx(destination: Path, text: str) -> None:
    from xml.sax.saxutils import escape
    from zipfile import ZIP_DEFLATED, ZipFile

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
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)


def _windows_speech(text: str, destination: Path) -> None:
    escaped_text = text.replace("'", "''")
    escaped_path = str(destination.resolve()).replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{escaped_path}'); $s.Speak('{escaped_text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)
