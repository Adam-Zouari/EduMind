"""Configuration for the OCR package."""

from __future__ import annotations

import os

from edumind.common.paths import artifact_path

OCR_ARTIFACTS_DIR = artifact_path("ocr", create=True)
TEMP_DIR = artifact_path("ocr", "temp")
OUTPUT_DIR = artifact_path("ocr", "output")
LOGS_DIR = artifact_path("ocr", "logs")
CACHE_DIR = artifact_path("ocr", "cache")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OCR_USE_PADDLE = os.getenv("OCR_USE_PADDLE", "true").lower() == "true"
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
OCR_LANGUAGES = ["eng", "fra", "spa", "deu"]
OCR_CONFIDENCE_THRESHOLD = 60
OCR_ENABLE_CACHING = True
OCR_CACHE_DIR = CACHE_DIR / "ocr"
OCR_ADAPTIVE_PREPROCESSING = True
OCR_ROTATION_CORRECTION = True
OCR_PERSPECTIVE_CORRECTION = True
OCR_QUALITY_THRESHOLD = 50
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "false").lower() == "true"
OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip().lower()
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE")

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

WEB_TIMEOUT = int(os.getenv("WEB_TIMEOUT", "30"))
USER_AGENT = "Mozilla/5.0 (compatible; EduMindAIBot/1.0)"

REMOVE_HEADERS_FOOTERS = True
NORMALIZE_WHITESPACE = True
PRESERVE_LATEX = True
MIN_TEXT_LENGTH = 10

LOG_LEVEL = os.getenv("EDUMIND_LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

SUPPORTED_FORMATS = {
    "pdf": [".pdf"],
    "docx": [".docx", ".doc"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
    "video": [".mp4", ".avi", ".mov", ".mkv", ".flv"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    "web": [".html", ".htm", ".xml"],
}
