from __future__ import annotations

from pathlib import Path

from PIL import Image

from edumind.ocr.core.base_extractor import ExtractionResult
from edumind.ocr.core.pipeline import DataIngestionPipeline
from edumind.ocr.utils.file_handler import FileHandler


def _create_image(path: Path) -> None:
    Image.new("RGB", (80, 50), color="white").save(path)


class _StubExtractor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        self.calls.append(kwargs)
        return ExtractionResult(
            text="Name: Alice Example",
            metadata={
                "extractor": "stub",
                "ocr_data": {
                    "text": ["Name", "Alice", "Example"],
                    "conf": [95.0, 94.0, 94.0],
                    "left": [0, 10, 40],
                    "top": [0, 0, 0],
                    "width": [10, 20, 20],
                    "height": [10, 10, 10],
                },
                "image_shape": [30, 60, 3],
            },
            format_type="image",
            file_path=str(file_path),
            success=True,
        )


def test_process_file_can_skip_hashing(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "sample.png"
    _create_image(image_path)
    pipeline = DataIngestionPipeline()
    stub = _StubExtractor()
    monkeypatch.setattr(pipeline, "_get_extractor", lambda *_args, **_kwargs: stub)
    monkeypatch.setattr(
        FileHandler,
        "get_file_hash",
        staticmethod(lambda _: (_ for _ in ()).throw(AssertionError("hash should be skipped"))),
    )

    result = pipeline.process_file(image_path, include_file_hash=False)

    assert result.success is True
    assert "file_hash" not in result.metadata


def test_process_file_adds_performance_metadata_when_profiled(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "profiled.png"
    _create_image(image_path)
    pipeline = DataIngestionPipeline()
    stub = _StubExtractor()
    monkeypatch.setattr(pipeline, "_get_extractor", lambda *_args, **_kwargs: stub)

    profiled = pipeline.process_file(image_path, profile=True)
    unprofiled = pipeline.process_file(image_path, profile=False)

    assert "performance" in profiled.metadata
    assert "format_detection" in profiled.metadata["performance"]
    assert "performance" not in unprofiled.metadata


def test_process_file_attaches_layout_and_form_metadata_only_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "structured.png"
    _create_image(image_path)
    pipeline = DataIngestionPipeline()
    stub = _StubExtractor()
    monkeypatch.setattr(pipeline, "_get_extractor", lambda *_args, **_kwargs: stub)

    base = pipeline.process_file(image_path)
    enriched = pipeline.process_file(image_path, include_layout=True, include_form_fields=True)

    assert "structured_fields" not in base.metadata
    assert "layout_blocks" not in base.metadata
    assert enriched.metadata["structured_fields"]["name"]["value"] == "Alice Example"
    assert enriched.metadata["layout_blocks"]
    assert "ocr_data" not in enriched.metadata
    assert "image_shape" not in enriched.metadata


def test_process_file_propagates_languages_to_extractors(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "lang.png"
    _create_image(image_path)
    pipeline = DataIngestionPipeline()
    stub = _StubExtractor()
    monkeypatch.setattr(pipeline, "_get_extractor", lambda *_args, **_kwargs: stub)

    pipeline.process_file(image_path, languages=["eng", "fra"])

    assert stub.calls[-1]["languages"] == ["eng", "fra"]


def test_batch_strategy_auto_routes_media_sequentially_and_keeps_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "one.png"
    audio_path = tmp_path / "two.mp3"
    _create_image(image_path)
    audio_path.write_bytes(b"audio")
    pipeline = DataIngestionPipeline()

    formats = {image_path: "image", audio_path: "audio"}
    monkeypatch.setattr(
        pipeline.format_detector,
        "detect",
        lambda file_path, strict=False: {
            "format_type": formats[Path(file_path)],
            "mime_type": None,
            "extension": Path(file_path).suffix.lower(),
        },
    )

    threaded_calls: list[str] = []
    direct_calls: list[str] = []

    def _thread_wrapper(detected_file, options, image_limiter):
        threaded_calls.append(detected_file.format_info.format_type)
        return ExtractionResult(
            text=detected_file.file_path.name,
            file_path=str(detected_file.file_path),
            success=True,
        )

    def _direct_wrapper(file_path: Path, format_info, format_detection_time, total_start, batch_options):
        direct_calls.append(format_info.format_type)
        return ExtractionResult(text=file_path.name, file_path=str(file_path), success=True)

    monkeypatch.setattr(pipeline.batch_coordinator, "_process_with_optional_image_limit", _thread_wrapper)
    monkeypatch.setattr(pipeline.batch_coordinator, "process_detected_file", _direct_wrapper)

    results = pipeline.process_batch([image_path, audio_path], batch_strategy="auto")

    assert threaded_calls == ["image"]
    assert direct_calls == ["audio"]
    assert [result.text for result in results] == [image_path.name, audio_path.name]
