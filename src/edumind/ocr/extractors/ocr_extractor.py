"""OCR extraction using Tesseract and PaddleOCR with adaptive preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import pytesseract

from ..config import (
    OCR_ADAPTIVE_PREPROCESSING,
    OCR_CACHE_DIR,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_ENABLE_CACHING,
    OCR_LANGUAGES,
    OCR_PERSPECTIVE_CORRECTION,
    OCR_QUALITY_THRESHOLD,
    OCR_ROTATION_CORRECTION,
    OCR_USE_ANGLE_CLS,
    OCR_USE_GPU,
    OCR_USE_PADDLE,
    TESSERACT_CMD,
)
from ..core.base_extractor import BaseExtractor, ExtractionResult

try:
    from paddleocr import PaddleOCR

    PADDLE_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLE_AVAILABLE = False

try:
    import mlflow
except ImportError:
    mlflow = None


class OCRExtractor(BaseExtractor):
    """Extract text from images with preprocessing, validation, and caching."""

    _paddle_instance: object | None = None
    _paddle_lock = Lock()

    def __init__(
        self,
        use_paddle: bool | None = None,
        confidence_threshold: float | None = None,
        languages: list[str] | None = None,
        enable_caching: bool = OCR_ENABLE_CACHING,
    ) -> None:
        super().__init__()
        self.use_paddle = OCR_USE_PADDLE if use_paddle is None else use_paddle
        self.confidence_threshold = (
            OCR_CONFIDENCE_THRESHOLD
            if confidence_threshold is None
            else confidence_threshold
        )
        self.languages = list(languages or OCR_LANGUAGES)
        self.enable_caching = enable_caching
        self.cache_dir = OCR_CACHE_DIR if enable_caching else None
        self.paddle_ocr: object | None = None

        if self.enable_caching and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.use_paddle and not PADDLE_AVAILABLE:
            self.logger.warning("PaddleOCR requested but not available. Falling back to Tesseract.")
            self.use_paddle = False

        if self.use_paddle:
            self.paddle_ocr = self._get_or_create_paddle_instance()
            if self.paddle_ocr is None:
                self.logger.warning("PaddleOCR initialization failed. Falling back to Tesseract.")
                self.use_paddle = False

        if not self.use_paddle:
            self.logger.info(f"Using Tesseract OCR: {TESSERACT_CMD}")
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    def extract(self, file_path: Path, **kwargs: Any) -> ExtractionResult:
        """Extract text from an image file."""
        image = cv2.imread(str(file_path))
        if image is None:
            return self._create_error_result(file_path, f"Could not read image: {file_path}")

        result = self.extract_image(
            image,
            source_name=str(file_path),
            cache_key=self._get_cache_key(file_path),
            languages=kwargs.get("languages"),
            return_ocr_data=bool(kwargs.get("return_ocr_data", False)),
            use_cache=bool(kwargs.get("use_cache", True)),
            format_type=kwargs.get("format_type", "image"),
        )
        result.file_path = str(file_path)
        return result

    def extract_image(
        self,
        image: np.ndarray,
        *,
        source_name: str = "",
        cache_key: str | None = None,
        languages: list[str] | None = None,
        return_ocr_data: bool = False,
        use_cache: bool = True,
        format_type: str = "image",
    ) -> ExtractionResult:
        """Extract text from an in-memory image array."""
        start_time = time.time()
        effective_languages = list(languages or self.languages)

        if self.enable_caching and use_cache and cache_key:
            cached_result = self._get_cached_result_from_key(cache_key)
            if cached_result is not None:
                cached_result.metadata["cache"] = {
                    "hit": True,
                    "kind": format_type,
                    "key": cache_key,
                }
                return cached_result

        try:
            quality_score = self._assess_image_quality(image)
            self.logger.info(f"Image quality score: {quality_score:.2f}")

            preprocessed, preprocessing_info = self._preprocess_image_advanced(image, quality_score)
            text, confidence, attempt_info, ocr_data = self._extract_with_retry(
                preprocessed,
                languages=effective_languages,
                return_ocr_data=return_ocr_data,
            )
            is_valid, validation_msg = self._validate_extraction(text, confidence)
            ocr_engine = "paddleocr" if self.use_paddle else "tesseract"
            extraction_time = time.time() - start_time

            metadata: dict[str, Any] = {
                "ocr_engine": ocr_engine,
                "confidence": confidence,
                "languages": effective_languages,
                "quality_score": quality_score,
                "preprocessing": preprocessing_info,
                "extraction_attempts": attempt_info,
                "validation": {"is_valid": is_valid, "message": validation_msg},
                "extractor": "ocr",
                "cache": {
                    "hit": False,
                    "kind": format_type,
                    "key": cache_key,
                },
            }
            if return_ocr_data and ocr_data is not None:
                metadata["ocr_data"] = ocr_data
                metadata["image_shape"] = list(image.shape)

            self._log_optional_mlflow_metrics(
                extraction_time=extraction_time,
                confidence=confidence,
                quality_score=quality_score,
                ocr_engine=ocr_engine,
            )

            result = ExtractionResult(
                text=text,
                metadata=metadata,
                format_type=format_type,
                file_path=source_name,
                extraction_time=extraction_time,
                success=is_valid,
            )

            if self.enable_caching and use_cache and cache_key and is_valid:
                self._cache_result_with_key(cache_key, result)

            return result
        except Exception as exc:
            self.logger.error(f"OCR extraction failed: {exc}")
            return ExtractionResult(
                text="",
                metadata={
                    "cache": {
                        "hit": False,
                        "kind": format_type,
                        "key": cache_key,
                    }
                },
                format_type=format_type,
                file_path=source_name,
                success=False,
                error=str(exc),
            )

    def _get_or_create_paddle_instance(self) -> object | None:
        """Initialize the shared PaddleOCR instance in a thread-safe way."""
        if not PADDLE_AVAILABLE or PaddleOCR is None:
            return None

        if OCRExtractor._paddle_instance is not None:
            return OCRExtractor._paddle_instance

        with OCRExtractor._paddle_lock:
            if OCRExtractor._paddle_instance is not None:
                return OCRExtractor._paddle_instance

            try:
                import paddle

                os.environ["FLAGS_allocator_strategy"] = "auto_growth"
                if OCR_USE_GPU and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                    paddle.device.set_device("gpu:0")
                else:
                    paddle.device.set_device("cpu")

                self.logger.info("Initializing shared PaddleOCR instance...")
                OCRExtractor._paddle_instance = PaddleOCR(
                    use_angle_cls=OCR_USE_ANGLE_CLS,
                    lang="en",
                )
                self.logger.info(f"PaddleOCR initialized (angle_cls={OCR_USE_ANGLE_CLS})")
            except Exception as exc:
                self.logger.error(f"Failed to initialize PaddleOCR: {exc}")
                return None

        return OCRExtractor._paddle_instance

    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(100.0, (laplacian_var / 10.0) * 100.0)

    def _preprocess_image_advanced(
        self,
        image: np.ndarray,
        quality_score: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply adaptive preprocessing based on the measured image quality."""
        preprocessing_steps: list[str] = []
        working_image = image.copy()

        low_quality_threshold = OCR_QUALITY_THRESHOLD
        medium_quality_threshold = min(OCR_QUALITY_THRESHOLD + 20, 100)
        adaptive_threshold_limit = min(OCR_QUALITY_THRESHOLD + 10, 100)
        morphology_threshold = max(OCR_QUALITY_THRESHOLD - 10, 0)

        if OCR_ROTATION_CORRECTION:
            working_image, rotation_angle = self._correct_rotation(working_image)
            if abs(rotation_angle) > 0.5:
                preprocessing_steps.append(f"rotation_corrected_{rotation_angle:.1f}deg")

        if OCR_PERSPECTIVE_CORRECTION:
            working_image, perspective_corrected = self._correct_perspective(working_image)
            if perspective_corrected:
                preprocessing_steps.append("perspective_corrected")

        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        preprocessing_steps.append("grayscale")

        if OCR_ADAPTIVE_PREPROCESSING:
            if quality_score < low_quality_threshold:
                gray = cv2.fastNlMeansDenoising(gray, h=10)
                preprocessing_steps.append("aggressive_denoise")
            elif quality_score < medium_quality_threshold:
                gray = cv2.medianBlur(gray, 3)
                preprocessing_steps.append("moderate_denoise")
            else:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                preprocessing_steps.append("light_denoise")
        else:
            gray = cv2.medianBlur(gray, 3)
            preprocessing_steps.append("default_denoise")

        if quality_score < adaptive_threshold_limit:
            threshold = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
            preprocessing_steps.append("adaptive_threshold")
        else:
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessing_steps.append("otsu_threshold")

        if quality_score < morphology_threshold:
            kernel = np.ones((2, 2), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            preprocessing_steps.append("morphological_closing")

        return threshold, {
            "steps": preprocessing_steps,
            "quality_score": quality_score,
        }

    def _correct_rotation(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect and correct image rotation using OCR orientation heuristics."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                osd = pytesseract.image_to_osd(gray)
                rotation_line = next(line for line in osd.splitlines() if "Rotate" in line)
                rotation_angle = int(rotation_line.split(":")[1].strip())
                if rotation_angle != 0:
                    return self._rotate_image(image, rotation_angle), float(rotation_angle)
            except Exception:
                rotation_angle = self._detect_rotation_contours(gray)
                if abs(rotation_angle) > 0.5:
                    return self._rotate_image(image, rotation_angle), rotation_angle

            return image, 0.0
        except Exception as exc:
            self.logger.warning(f"Rotation correction failed: {exc}")
            return image, 0.0

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate an image while keeping the full frame."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _detect_rotation_contours(self, gray: np.ndarray) -> float:
        """Estimate rotation angle using Hough line detection."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            return 0.0

        angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]]
        return float(np.median(angles)) if angles else 0.0

    def _correct_perspective(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """Detect and correct perspective distortion."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image, False

            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            if len(approx) != 4:
                return image, False

            rect = self._order_points(approx.reshape(4, 2))
            top_left, top_right, bottom_right, bottom_left = rect
            width_a = np.linalg.norm(bottom_right - bottom_left)
            width_b = np.linalg.norm(top_right - top_left)
            max_width = max(int(width_a), int(width_b))
            height_a = np.linalg.norm(top_right - bottom_right)
            height_b = np.linalg.norm(top_left - bottom_left)
            max_height = max(int(height_a), int(height_b))

            destination = np.array(
                [
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1],
                ],
                dtype="float32",
            )

            matrix = cv2.getPerspectiveTransform(rect, destination)
            warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
            return warped, True
        except Exception as exc:
            self.logger.warning(f"Perspective correction failed: {exc}")
            return image, False

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order contour points clockwise starting from the top-left."""
        rect = np.zeros((4, 2), dtype="float32")
        sums = points.sum(axis=1)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        diffs = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    def _extract_with_retry(
        self,
        image: np.ndarray,
        *,
        languages: list[str],
        return_ocr_data: bool,
    ) -> tuple[str, float, list[str], dict[str, Any] | None]:
        """Run OCR and retry once with an inverted image on low confidence."""
        attempts = []
        text, confidence, ocr_data = self._run_ocr_engine(
            image,
            languages=languages,
            return_ocr_data=return_ocr_data,
        )
        attempts.append(f"standard (conf: {confidence:.2f})")

        if confidence < self.confidence_threshold:
            self.logger.info(
                f"Low confidence ({confidence:.2f}), trying alternative preprocessing..."
            )
            inverted = cv2.bitwise_not(image)
            retry_text, retry_confidence, retry_ocr_data = self._run_ocr_engine(
                inverted,
                languages=languages,
                return_ocr_data=return_ocr_data,
            )
            attempts.append(f"inverted (conf: {retry_confidence:.2f})")
            if retry_confidence > confidence:
                text, confidence, ocr_data = retry_text, retry_confidence, retry_ocr_data

        return text, confidence, attempts, ocr_data

    def _run_ocr_engine(
        self,
        image: np.ndarray,
        *,
        languages: list[str],
        return_ocr_data: bool,
    ) -> tuple[str, float, dict[str, Any] | None]:
        """Run the configured OCR engine, falling back to Tesseract when needed."""
        if self.use_paddle and self.paddle_ocr is not None:
            try:
                text, confidence = self._extract_with_paddle(image, languages=languages)
                ocr_data = None
                if return_ocr_data:
                    ocr_data = self._extract_layout_data_with_tesseract(image, languages)
                return text, confidence, ocr_data
            except Exception as exc:
                self.logger.warning(
                    f"PaddleOCR failed for this attempt, falling back to Tesseract: {exc}"
                )

        return self._extract_with_tesseract(
            image,
            languages=languages,
            return_ocr_data=return_ocr_data,
        )

    def _extract_with_tesseract(
        self,
        image: np.ndarray,
        *,
        languages: list[str],
        return_ocr_data: bool,
    ) -> tuple[str, float, dict[str, Any] | None]:
        """Extract text using Tesseract and filter low-confidence tokens."""
        data = pytesseract.image_to_data(
            image,
            lang="+".join(languages),
            output_type=pytesseract.Output.DICT,
            config="--psm 3",
        )

        text_parts = []
        confidences = []
        for index, raw_confidence in enumerate(data["conf"]):
            confidence = self._parse_confidence(raw_confidence)
            if confidence <= self.confidence_threshold:
                continue

            token = data["text"][index].strip()
            if token:
                text_parts.append(token)
                confidences.append(confidence)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        ocr_data = self._build_ocr_data_payload(data) if return_ocr_data else None
        return " ".join(text_parts), avg_confidence, ocr_data

    def _extract_with_paddle(
        self,
        image: np.ndarray,
        *,
        languages: list[str],
    ) -> tuple[str, float]:
        """Extract text using PaddleOCR while handling multiple result shapes."""
        if self.paddle_ocr is None:
            raise RuntimeError("PaddleOCR is not initialized")

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
        result = self.paddle_ocr.ocr(processed_image)

        text_parts: list[str] = []
        confidences: list[float] = []

        if result and isinstance(result, list):
            page_result = result[0]

            if page_result and hasattr(page_result, "__class__") and "OCRResult" in page_result.__class__.__name__:
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
                                confidence = item.get("score", item.get("confidence", 1.0))
                                confidences.append(float(confidence) * 100)
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
        languages: list[str],
    ) -> dict[str, Any]:
        """Collect token-level OCR data for optional layout analysis."""
        data = pytesseract.image_to_data(
            image,
            lang="+".join(languages),
            output_type=pytesseract.Output.DICT,
            config="--psm 3",
        )
        return self._build_ocr_data_payload(data)

    def _build_ocr_data_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize OCR token payloads into plain JSON-safe lists."""
        return {
            "text": [str(value) for value in data.get("text", [])],
            "conf": [self._parse_confidence(value) for value in data.get("conf", [])],
            "left": [int(value) for value in data.get("left", [])],
            "top": [int(value) for value in data.get("top", [])],
            "width": [int(value) for value in data.get("width", [])],
            "height": [int(value) for value in data.get("height", [])],
        }

    def _validate_extraction(self, text: str, confidence: float) -> tuple[bool, str]:
        """Validate OCR output quality."""
        if not text or len(text.strip()) == 0:
            return False, "No text extracted"

        if confidence < self.confidence_threshold:
            return False, f"Low confidence: {confidence:.2f} < {self.confidence_threshold}"

        words = text.split()
        if len(words) < 3:
            return False, f"Too few words extracted: {len(words)}"

        special_char_ratio = sum(
            1 for character in text if not character.isalnum() and not character.isspace()
        ) / len(text)
        if special_char_ratio > 0.5:
            return False, f"Too many special characters: {special_char_ratio:.2%}"

        return True, "Extraction validated successfully"

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate a cache key from stable file identity fields."""
        stat = file_path.stat()
        key_str = f"{file_path.resolve()}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, file_path: Path) -> ExtractionResult | None:
        """Retrieve a cached OCR result if available."""
        return self._get_cached_result_from_key(self._get_cache_key(file_path))

    def _get_cached_result_from_key(self, cache_key: str) -> ExtractionResult | None:
        """Retrieve a cached OCR result by a precomputed cache key."""
        if not self.cache_dir:
            return None

        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                return None

            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return ExtractionResult.from_cache_dict(data)
        except Exception as exc:
            self.logger.warning(f"Failed to load cache: {exc}")
            return None

    def _cache_result(self, file_path: Path, result: ExtractionResult) -> None:
        """Persist OCR results for future cache hits."""
        self._cache_result_with_key(self._get_cache_key(file_path), result)

    def _cache_result_with_key(self, cache_key: str, result: ExtractionResult) -> None:
        """Persist OCR results for future cache hits using a precomputed key."""
        if not self.cache_dir:
            return

        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.write_text(
                json.dumps(result.to_cache_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.logger.debug(f"Cached OCR result with key {cache_key}")
        except Exception as exc:
            self.logger.warning(f"Failed to cache result: {exc}")

    def _log_optional_mlflow_metrics(
        self,
        *,
        extraction_time: float,
        confidence: float,
        quality_score: float,
        ocr_engine: str,
    ) -> None:
        """Log OCR metrics to MLflow when the dependency and an active run exist."""
        if mlflow is None:
            return

        try:
            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "ocr_processing_time": extraction_time,
                        "ocr_confidence": float(confidence),
                        "image_quality_score": quality_score,
                    }
                )
                mlflow.log_param("ocr_engine", ocr_engine)
        except Exception as exc:
            self.logger.warning(f"Failed to log to MLflow: {exc}")

    @staticmethod
    def _parse_confidence(value: Any) -> float:
        """Parse OCR confidence values that may arrive as strings or numbers."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
