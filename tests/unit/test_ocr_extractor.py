from __future__ import annotations

from pathlib import Path

import numpy as np

from edumind.ocr.extractors import ocr_extractor
from edumind.ocr.extractors.ocr_extractor import OCRExtractor


def test_ocr_extractor_cache_hit_round_trips_full_result(tmp_path: Path, monkeypatch) -> None:
    extractor = OCRExtractor(use_paddle=False, enable_caching=True)
    extractor.cache_dir = tmp_path / "cache"
    extractor.cache_dir.mkdir()

    monkeypatch.setattr(extractor, "_assess_image_quality", lambda _: 80.0)
    monkeypatch.setattr(
        extractor,
        "_preprocess_image_advanced",
        lambda image, quality: (
            np.zeros((10, 10), dtype=np.uint8),
            {"steps": ["mock"], "quality_score": quality},
        ),
    )
    monkeypatch.setattr(
        extractor,
        "_extract_with_retry",
        lambda image, **kwargs: (
            "hello world content",
            95.0,
            ["standard (conf: 95.00)"],
            None,
        ),
    )
    monkeypatch.setattr(extractor, "_validate_extraction", lambda *_: (True, "ok"))
    monkeypatch.setattr(extractor, "_log_optional_mlflow_metrics", lambda **_: None)

    first = extractor.extract_image(
        np.zeros((10, 10, 3), dtype=np.uint8),
        source_name="synthetic-image",
        cache_key="page-cache-key",
    )
    assert first.success is True
    assert (extractor.cache_dir / "page-cache-key.json").exists()
    assert first.metadata["cache"]["hit"] is False

    monkeypatch.setattr(
        extractor,
        "_assess_image_quality",
        lambda _: (_ for _ in ()).throw(AssertionError("cache was not used")),
    )

    second = extractor.extract_image(
        np.zeros((10, 10, 3), dtype=np.uint8),
        source_name="synthetic-image",
        cache_key="page-cache-key",
    )
    assert second.text == first.text
    assert second.metadata["cache"]["hit"] is True


def test_ocr_extractor_can_return_layout_ready_ocr_data(monkeypatch) -> None:
    extractor = OCRExtractor(use_paddle=False, enable_caching=False)
    fake_ocr_data = {
        "text": ["Name", "Alice"],
        "conf": [90.0, 91.0],
        "left": [0, 10],
        "top": [0, 10],
        "width": [20, 30],
        "height": [10, 10],
    }

    monkeypatch.setattr(extractor, "_assess_image_quality", lambda _: 80.0)
    monkeypatch.setattr(
        extractor,
        "_preprocess_image_advanced",
        lambda image, quality: (image[:, :, 0], {"steps": ["mock"], "quality_score": quality}),
    )
    monkeypatch.setattr(
        extractor,
        "_extract_with_retry",
        lambda image, **kwargs: ("Name Alice", 95.0, ["standard"], fake_ocr_data),
    )
    monkeypatch.setattr(extractor, "_validate_extraction", lambda *_: (True, "ok"))
    monkeypatch.setattr(extractor, "_log_optional_mlflow_metrics", lambda **_: None)

    result = extractor.extract_image(
        np.zeros((20, 20, 3), dtype=np.uint8),
        return_ocr_data=True,
        use_cache=False,
    )

    assert result.metadata["ocr_data"] == fake_ocr_data
    assert result.metadata["image_shape"] == [20, 20, 3]


def test_optional_mlflow_logging_is_a_noop_when_dependency_is_missing(monkeypatch) -> None:
    extractor = OCRExtractor(use_paddle=False, enable_caching=False)
    monkeypatch.setattr(ocr_extractor, "mlflow", None)

    extractor._log_optional_mlflow_metrics(
        extraction_time=0.5,
        confidence=99.0,
        quality_score=88.0,
        ocr_engine="tesseract",
    )


def test_parse_confidence_handles_invalid_values() -> None:
    assert OCRExtractor._parse_confidence("91.5") == 91.5
    assert OCRExtractor._parse_confidence("-1") == -1.0
    assert OCRExtractor._parse_confidence("not-a-number") == 0.0
