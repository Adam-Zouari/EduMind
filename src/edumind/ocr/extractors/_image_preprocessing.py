"""Preprocessing helpers for image OCR."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from ..config import (
    OCR_ADAPTIVE_PREPROCESSING,
    OCR_PERSPECTIVE_CORRECTION,
    OCR_QUALITY_THRESHOLD,
    OCR_ROTATION_CORRECTION,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreprocessedImage:
    """Preprocessed image payload plus transformation metadata."""

    image: np.ndarray
    metadata: dict[str, object]


class ImagePreprocessor:
    """Apply adaptive preprocessing to improve OCR robustness."""

    def assess_quality(self, image: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(100.0, (laplacian_var / 10.0) * 100.0)

    def preprocess(self, image: np.ndarray, quality_score: float) -> PreprocessedImage:
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

        return PreprocessedImage(
            image=threshold,
            metadata={
                "steps": preprocessing_steps,
                "quality_score": quality_score,
            },
        )

    def _correct_rotation(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Detect and correct image rotation using OCR orientation heuristics."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            try:
                osd = pytesseract.image_to_osd(gray)
                rotation_line = next(line for line in osd.splitlines() if "Rotate" in line)
                rotation_angle = float(int(rotation_line.split(":")[1].strip()))
                if rotation_angle != 0:
                    return self._rotate_image(image, rotation_angle), rotation_angle
            except Exception:
                rotation_angle = self._detect_rotation_contours(gray)
                if abs(rotation_angle) > 0.5:
                    return self._rotate_image(image, rotation_angle), rotation_angle

            return image, 0.0
        except Exception as exc:
            logger.warning(f"Rotation correction failed: {exc}")
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
            logger.warning(f"Perspective correction failed: {exc}")
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
