"""Lazy image-to-text implementations used by product and benchmarks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document


class ImageExtractor:
    supported_kinds = frozenset({SourceKind.IMAGE})

    def __init__(self, engine: str, revision: str) -> None:
        self.engine = engine
        self.name = engine
        self.revision = revision
        self._runtime: Any | None = None

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            image = self._load_and_preprocess(request)
            if self.engine == "tesseract-5":
                text = self._tesseract(image)
            elif self.engine.startswith("paddleocr"):
                text = self._paddle(image, request)
            elif self.engine == "doctr-fast-parseq":
                text = self._doctr(image, request)
            else:
                raise ValueError(f"Unknown image engine: {self.engine}")
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Image extraction failed with {self.engine}", detail=str(exc)
            ) from exc
        return build_document(
            request,
            kind,
            request.profile,
            [text.strip()],
            pages=[1],
            metadata={
                "engine": self.engine,
                "engine_revision": request.profile.engine_revision,
            },
            seconds=time.perf_counter() - started,
        )

    def _load_and_preprocess(self, request: ExtractionRequest):
        try:
            from PIL import Image, ImageOps
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "Pillow is required for image extraction; install .[extraction]"
            ) from exc
        image = Image.open(request.source_path).convert("RGB")
        profile = request.profile.preprocessing if request.profile else "raw"
        image = ImageOps.exif_transpose(image)
        if profile in {"document", "photo"}:
            image = self._opencv_preprocess(image, profile)
        return image

    @staticmethod
    def _opencv_preprocess(image, profile: str):
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise MissingDependencyError(
                "OpenCV, NumPy, and Pillow are required for document/photo preprocessing"
            ) from exc
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        if profile == "photo":
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                perimeter = cv2.arcLength(contour, True)
                corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(corners) == 4:
                    gray = _perspective_warp(gray, corners.reshape(4, 2).astype("float32"))
                    break
            gray = cv2.medianBlur(gray, 3)
        inverted = cv2.bitwise_not(gray)
        coordinates = np.column_stack(np.where(inverted > 0))
        if len(coordinates):
            angle = cv2.minAreaRect(coordinates)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) > 0.05:
                height, width = gray.shape
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                gray = cv2.warpAffine(
                    gray,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return Image.fromarray(contrast)

    @staticmethod
    def _tesseract(image) -> str:
        try:
            import pytesseract
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("pytesseract is required; install .[extraction]") from exc
        return str(pytesseract.image_to_string(image, lang="eng"))

    def _paddle(self, image, request: ExtractionRequest) -> str:
        try:
            import numpy as np
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("PaddleOCR is required; install .[extraction]") from exc
        use_server = self.engine.endswith("server")
        model_name = "PP-OCRv5_server_rec" if use_server else "en_PP-OCRv5_mobile_rec"
        detection_directory = Path(str(request.options.get("text_detection_model_dir", "")))
        recognition_directory = Path(str(request.options.get("text_recognition_model_dir", "")))
        if not detection_directory.is_dir() or not recognition_directory.is_dir():
            raise FileNotFoundError(
                "PaddleOCR weights are not prepared locally; run `edumind benchmark prepare "
                "extraction-models` and use its model lock"
            )
        if self._runtime is None:
            self._runtime = PaddleOCR(
                lang="en",
                text_detection_model_dir=str(detection_directory),
                text_recognition_model_dir=str(recognition_directory),
                text_recognition_model_name=model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        result = self._runtime.predict(np.asarray(image))
        texts: list[str] = []
        for page in result:
            payload = getattr(page, "json", page)
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict):
                value = payload.get("res", payload).get("rec_texts", [])
                if isinstance(value, list):
                    texts.extend(str(item) for item in value)
        return "\n".join(texts)

    def _doctr(self, image, request: ExtractionRequest) -> str:
        try:
            import numpy as np
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("docTR is required for this candidate") from exc
        cache_directory = Path(str(request.options.get("doctr_cache_dir", "")))
        if not cache_directory.is_dir() or not any(cache_directory.rglob("*")):
            raise FileNotFoundError(
                "docTR weights are not prepared locally; run `edumind benchmark prepare "
                "extraction-models` and use its model lock"
            )
        os.environ["DOCTR_CACHE_DIR"] = str(cache_directory)
        if self._runtime is None:
            self._runtime = ocr_predictor(det_arch="fast_base", reco_arch="parseq", pretrained=True)
        result = self._runtime(DocumentFile.from_images([np.asarray(image)]))
        return str(result.render())


def _perspective_warp(gray, points):
    import cv2
    import numpy as np

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered = np.asarray(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype="float32",
    )
    top_left, top_right, bottom_right, bottom_left = ordered
    width = int(
        max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    )
    height = int(
        max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    )
    if width < 2 or height < 2:
        return gray
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(gray, transform, (width, height))
