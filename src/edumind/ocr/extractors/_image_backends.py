"""OCR backend adapters for image extraction."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, cast

import cv2
import numpy as np
import pytesseract

from ..config import OCR_USE_ANGLE_CLS, OCR_USE_GPU, TESSERACT_CMD
from ..core.errors import OCRBackendError
from ..core.types import OCRTokenPayload
from ..utils.logger import get_logger
from ._image_validation import ImageValidationRules

logger = get_logger(__name__)

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLE_AVAILABLE = False


@dataclass(frozen=True)
class OCRRunResult:
    """Result of a backend OCR execution."""

    text: str
    confidence: float
    attempts: list[str]
    ocr_data: OCRTokenPayload | None


class ImageOCRBackends:
    """Run OCR through PaddleOCR or Tesseract."""

    _paddle_instance: object | None = None
    _paddle_lock = Lock()

    def __init__(
        self,
        *,
        use_paddle: bool,
        confidence_threshold: float,
        validation_rules: ImageValidationRules,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.validation_rules = validation_rules
        self.use_paddle = use_paddle
        self.paddle_ocr: object | None = None

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

        if self.use_paddle and not PADDLE_AVAILABLE:
            logger.warning("PaddleOCR requested but not available. Falling back to Tesseract.")
            self.use_paddle = False

        if self.use_paddle:
            self.paddle_ocr = self._get_or_create_paddle_instance()
            if self.paddle_ocr is None:
                logger.warning("PaddleOCR initialization failed. Falling back to Tesseract.")
                self.use_paddle = False

        if not self.use_paddle:
            logger.info(f"Using Tesseract OCR: {TESSERACT_CMD}")

    @property
    def engine_name(self) -> str:
        """Return the backend name used for metadata and cache keys."""
        return "paddleocr" if self.use_paddle else "tesseract"

    def extract(
        self,
        image: np.ndarray,
        *,
        languages: Sequence[str],
        return_ocr_data: bool,
    ) -> OCRRunResult:
        """Run OCR and retry once with an inverted image on low confidence."""
        attempts: list[str] = []
        text, confidence, ocr_data = self._run_once(
            image,
            languages=languages,
            return_ocr_data=return_ocr_data,
        )
        attempts.append(f"standard (conf: {confidence:.2f})")

        if confidence < self.confidence_threshold:
            logger.info(f"Low confidence ({confidence:.2f}), trying alternative preprocessing...")
            inverted = cv2.bitwise_not(image)
            retry_text, retry_confidence, retry_ocr_data = self._run_once(
                inverted,
                languages=languages,
                return_ocr_data=return_ocr_data,
            )
            attempts.append(f"inverted (conf: {retry_confidence:.2f})")
            if retry_confidence > confidence:
                text = retry_text
                confidence = retry_confidence
                ocr_data = retry_ocr_data

        return OCRRunResult(
            text=text,
            confidence=confidence,
            attempts=attempts,
            ocr_data=ocr_data,
        )

    def _run_once(
        self,
        image: np.ndarray,
        *,
        languages: Sequence[str],
        return_ocr_data: bool,
    ) -> tuple[str, float, OCRTokenPayload | None]:
        """Run the configured OCR backend once."""
        if self.use_paddle and self.paddle_ocr is not None:
            try:
                text, confidence = self._extract_with_paddle(image)
                ocr_data = None
                if return_ocr_data:
                    ocr_data = self._extract_layout_data_with_tesseract(image, languages)
                return text, confidence, ocr_data
            except Exception as exc:
                logger.warning(
                    "PaddleOCR failed for this attempt, falling back to Tesseract: {}",
                    exc,
                )

        return self._extract_with_tesseract(
            image,
            languages=languages,
            return_ocr_data=return_ocr_data,
        )

    def _get_or_create_paddle_instance(self) -> object | None:
        """Initialize the shared PaddleOCR instance in a thread-safe way."""
        if not PADDLE_AVAILABLE or PaddleOCR is None:
            return None

        if ImageOCRBackends._paddle_instance is not None:
            return ImageOCRBackends._paddle_instance

        with ImageOCRBackends._paddle_lock:
            if ImageOCRBackends._paddle_instance is not None:
                return ImageOCRBackends._paddle_instance

            try:
                import paddle

                os.environ["FLAGS_allocator_strategy"] = "auto_growth"
                if (
                    OCR_USE_GPU
                    and paddle.is_compiled_with_cuda()
                    and paddle.device.cuda.device_count() > 0
                ):
                    paddle.device.set_device("gpu:0")
                else:
                    paddle.device.set_device("cpu")

                logger.info("Initializing shared PaddleOCR instance...")
                ImageOCRBackends._paddle_instance = PaddleOCR(
                    use_angle_cls=OCR_USE_ANGLE_CLS,
                    lang="en",
                )
                logger.info(f"PaddleOCR initialized (angle_cls={OCR_USE_ANGLE_CLS})")
            except Exception as exc:
                logger.error(f"Failed to initialize PaddleOCR: {exc}")
                return None

        return ImageOCRBackends._paddle_instance

    def _extract_with_tesseract(
        self,
        image: np.ndarray,
        *,
        languages: Sequence[str],
        return_ocr_data: bool,
    ) -> tuple[str, float, OCRTokenPayload | None]:
        """Extract text using Tesseract and filter low-confidence tokens."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang="+".join(languages),
                output_type=pytesseract.Output.DICT,
                config="--psm 3",
            )
        except Exception as exc:
            raise OCRBackendError(f"Tesseract OCR failed: {exc}") from exc

        text_parts: list[str] = []
        confidences: list[float] = []
        for index, raw_confidence in enumerate(data["conf"]):
            confidence = self.validation_rules.parse_confidence(raw_confidence)
            if confidence <= self.confidence_threshold:
                continue

            token = str(data["text"][index]).strip()
            if token:
                text_parts.append(token)
                confidences.append(confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        ocr_data = self._build_ocr_data_payload(data) if return_ocr_data else None
        return " ".join(text_parts), avg_confidence, ocr_data

    def _extract_with_paddle(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text using PaddleOCR while handling multiple result shapes."""
        if self.paddle_ocr is None:
            raise OCRBackendError("PaddleOCR is not initialized")

        processed_image = image
        if image.dtype != np.uint8:
            processed_image = (
                (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            )

        if len(processed_image.shape) == 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        elif processed_image.shape[2] == 4:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGBA2BGR)

        processed_image = np.ascontiguousarray(processed_image)
        paddle_runtime = cast(Any, self.paddle_ocr)
        result = paddle_runtime.ocr(processed_image)

        text_parts: list[str] = []
        confidences: list[float] = []

        if result and isinstance(result, list):
            page_result = result[0]

            if (
                page_result
                and hasattr(page_result, "__class__")
                and "OCRResult" in page_result.__class__.__name__
            ):
                self._collect_paddle_ocr_result(page_result, text_parts, confidences)
            elif page_result and isinstance(page_result, list):
                for line in page_result:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    content = line[1]
                    if not isinstance(content, (list, tuple)) or len(content) < 2:
                        continue

                    text = str(content[0])
                    confidence = float(content[1]) * 100
                    if confidence > self.confidence_threshold:
                        text_parts.append(text)
                        confidences.append(confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return " ".join(text_parts), avg_confidence

    def _collect_paddle_ocr_result(
        self,
        page_result: Any,
        text_parts: list[str],
        confidences: list[float],
    ) -> None:
        """Collect text from Paddle's newer OCRResult container shape."""
        if hasattr(page_result, "json") and callable(page_result.json):
            result_data = page_result.json()
            if isinstance(result_data, dict):
                for key in ("rec_text", "text", "texts", "results", "ocr_results"):
                    if key not in result_data:
                        continue

                    data = result_data[key]
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                text_parts.append(item)
                                confidences.append(100.0)
                            elif isinstance(item, dict) and "text" in item:
                                text_parts.append(str(item["text"]))
                                raw_confidence = item.get("score", item.get("confidence", 1.0))
                                if isinstance(raw_confidence, (int, float, str)):
                                    score = float(raw_confidence)
                                else:
                                    score = 1.0
                                confidences.append(score * 100)
                    elif isinstance(data, str):
                        text_parts.append(data)
                        confidences.append(100.0)
                    break
        elif hasattr(page_result, "items"):
            for _, value in page_result.items():
                if isinstance(value, str) and value:
                    text_parts.append(value)
                    confidences.append(100.0)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            text_parts.append(item)
                            confidences.append(100.0)
        elif hasattr(page_result, "str") and callable(page_result.str):
            result_str = page_result.str()
            if result_str and len(result_str) > 10:
                text_parts.append(result_str)
                confidences.append(100.0)

    def _extract_layout_data_with_tesseract(
        self,
        image: np.ndarray,
        languages: Sequence[str],
    ) -> OCRTokenPayload:
        """Collect token-level OCR data for optional layout analysis."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang="+".join(languages),
                output_type=pytesseract.Output.DICT,
                config="--psm 3",
            )
        except Exception as exc:
            raise OCRBackendError(f"Tesseract layout extraction failed: {exc}") from exc
        return self._build_ocr_data_payload(data)

    def _build_ocr_data_payload(self, data: dict[str, Any]) -> OCRTokenPayload:
        """Normalize OCR token payloads into plain JSON-safe lists."""
        return {
            "text": [str(value) for value in data.get("text", [])],
            "conf": [
                self.validation_rules.parse_confidence(value)
                for value in data.get("conf", [])
            ],
            "left": [int(value) for value in data.get("left", [])],
            "top": [int(value) for value in data.get("top", [])],
            "width": [int(value) for value in data.get("width", [])],
            "height": [int(value) for value in data.get("height", [])],
        }
