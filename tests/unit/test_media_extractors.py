from __future__ import annotations

from pathlib import Path

from edumind.ocr.extractors import audio_extractor, video_extractor


def test_audio_extractor_reports_missing_whisper_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(audio_extractor, "get_whisper_device", lambda: "cpu")
    monkeypatch.setattr(audio_extractor, "load_whisper_model", lambda _: (None, "missing whisper"))

    extractor = audio_extractor.AudioExtractor(model_name="tiny")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    result = extractor.extract(audio_path)

    assert result.success is False
    assert result.error == "missing whisper"


def test_video_extractor_reports_missing_whisper_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(video_extractor, "get_whisper_device", lambda: "cpu")
    monkeypatch.setattr(video_extractor, "load_whisper_model", lambda _: (None, "missing whisper"))

    extractor = video_extractor.VideoExtractor(model_name="tiny")
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    result = extractor.extract(video_path)

    assert result.success is False
    assert result.error == "missing whisper"


def test_video_extractor_cleans_up_temp_audio_file_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class DummyModel:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("transcription failed")

    monkeypatch.setattr(video_extractor, "get_whisper_device", lambda: "cpu")
    monkeypatch.setattr(video_extractor, "load_whisper_model", lambda _: (DummyModel(), None))

    extractor = video_extractor.VideoExtractor(model_name="tiny")
    temp_audio = tmp_path / "temp.wav"
    temp_audio.write_bytes(b"audio")
    monkeypatch.setattr(extractor, "_extract_audio", lambda _: temp_audio)

    result = extractor.extract(tmp_path / "video.mp4")

    assert result.success is False
    assert not temp_audio.exists()


def test_video_frame_rate_parser_avoids_eval() -> None:
    assert video_extractor.VideoExtractor._parse_frame_rate("30000/1001") == 29.97002997002997
    assert video_extractor.VideoExtractor._parse_frame_rate("invalid") == 0.0
