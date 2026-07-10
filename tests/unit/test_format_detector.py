from __future__ import annotations

from pathlib import Path

from edumind.ocr.core.format_detector import FormatDetector


def test_detect_uses_extension_when_available(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.pdf"
    file_path.write_bytes(b"pdf")
    detector = FormatDetector()

    monkeypatch.setattr(detector, "_detect_with_magic", lambda _: None)
    monkeypatch.setattr(detector, "_detect_with_tika", lambda _: None)

    result = detector.detect(file_path)

    assert result["format_type"] == "pdf"
    assert result["extension"] == ".pdf"
    assert result["mime_type"] is None


def test_detect_short_circuits_mime_probes_for_known_extensions(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.pdf"
    file_path.write_bytes(b"pdf")
    detector = FormatDetector()
    calls = {"magic": 0, "tika": 0}

    monkeypatch.setattr(
        detector,
        "_detect_with_magic",
        lambda _: calls.__setitem__("magic", calls["magic"] + 1),
    )
    monkeypatch.setattr(
        detector,
        "_detect_with_tika",
        lambda _: calls.__setitem__("tika", calls["tika"] + 1),
    )

    result = detector.detect(file_path)

    assert result["format_type"] == "pdf"
    assert calls == {"magic": 0, "tika": 0}


def test_detect_uses_mime_probes_when_strict_is_enabled(tmp_path: Path, monkeypatch) -> None:
    file_path = tmp_path / "notes.pdf"
    file_path.write_bytes(b"pdf")
    detector = FormatDetector()
    calls = {"magic": 0, "tika": 0}

    monkeypatch.setattr(
        detector,
        "_detect_with_magic",
        lambda _: calls.__setitem__("magic", calls["magic"] + 1) or "application/pdf",
    )
    monkeypatch.setattr(
        detector,
        "_detect_with_tika",
        lambda _: calls.__setitem__("tika", calls["tika"] + 1) or None,
    )

    result = detector.detect(file_path, strict=True)

    assert result["format_type"] == "pdf"
    assert calls == {"magic": 1, "tika": 1}


def test_detect_falls_back_to_mime_when_extension_is_unknown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    file_path = tmp_path / "page.bin"
    file_path.write_bytes(b"html")
    detector = FormatDetector()

    monkeypatch.setattr(detector, "_detect_with_magic", lambda _: "text/html")
    monkeypatch.setattr(detector, "_detect_with_tika", lambda _: None)

    result = detector.detect(file_path)

    assert result["format_type"] == "web"
    assert result["mime_type"] == "text/html"


def test_detect_prefers_tika_mime_when_both_detectors_return_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    file_path = tmp_path / "recording.data"
    file_path.write_bytes(b"audio")
    detector = FormatDetector()

    monkeypatch.setattr(detector, "_detect_with_magic", lambda _: "application/octet-stream")
    monkeypatch.setattr(detector, "_detect_with_tika", lambda _: "audio/mpeg")

    result = detector.detect(file_path)

    assert result["format_type"] == "audio"
    assert result["mime_type"] == "audio/mpeg"
