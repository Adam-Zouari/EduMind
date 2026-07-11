"""Format detection using Apache Tika and python-magic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import SUPPORTED_FORMATS
from ..utils.logger import get_logger
from .types import FormatInfo

logger = get_logger(__name__)

try:
    import magic

    magic_module: Any | None = magic
    MAGIC_AVAILABLE = True
except (ImportError, OSError) as exc:
    magic_module = None
    MAGIC_AVAILABLE = False
    logger.warning("python-magic not available ({}). Using extension-based detection.", exc)

try:
    from tika import detector

    tika_detector: Any | None = detector
    TIKA_AVAILABLE = True
except ImportError:
    tika_detector = None
    TIKA_AVAILABLE = False
    logger.warning("tika not available. Using extension-based detection.")


class FormatDetector:
    """Detect file format using optional MIME libraries plus extension fallback."""

    def __init__(self) -> None:
        self.magic: Any | None
        if MAGIC_AVAILABLE and magic_module is not None:
            try:
                self.magic = magic_module.Magic(mime=True)
            except Exception as exc:
                logger.warning("Failed to initialize python-magic: {}", exc)
                self.magic = None
        else:
            self.magic = None

    def detect(self, file_path: Path, strict: bool = False) -> FormatInfo:
        """Detect file format and return OCR pipeline format metadata."""
        logger.info(f"Detecting format for: {file_path}")
        extension = file_path.suffix.lower()
        mapped_extension_type = self._map_extension_to_format_type(extension)

        if mapped_extension_type and not strict:
            result = FormatInfo(
                format_type=mapped_extension_type,
                mime_type=None,
                extension=extension,
            )
            logger.info(f"Detected format from extension: {result.to_metadata_dict()}")
            return result

        mime_type = self._detect_with_magic(file_path)
        tika_mime = self._detect_with_tika(file_path)
        final_mime = tika_mime or mime_type
        format_type = mapped_extension_type or self._map_mime_to_format_type(final_mime)

        result = FormatInfo(
            format_type=format_type,
            mime_type=final_mime,
            extension=extension,
        )
        logger.info(f"Detected format: {result.to_metadata_dict()}")
        return result

    def _detect_with_magic(self, file_path: Path) -> str | None:
        """Detect MIME type using python-magic."""
        if not self.magic:
            return None
        try:
            mime_type = self.magic.from_file(str(file_path))
            logger.debug(f"Magic detected: {mime_type}")
            return mime_type
        except Exception as exc:
            logger.warning("Magic detection failed: {}", exc)
            return None

    def _detect_with_tika(self, file_path: Path) -> str | None:
        """Detect MIME type using Apache Tika."""
        if not TIKA_AVAILABLE or tika_detector is None:
            return None
        try:
            mime_type = tika_detector.from_file(str(file_path))
            logger.debug(f"Tika detected: {mime_type}")
            return mime_type
        except Exception as exc:
            logger.warning("Tika detection failed: {}", exc)
            return None

    def _map_extension_to_format_type(self, extension: str) -> str | None:
        """Map a file extension to an internal format category."""
        for format_type, extensions in SUPPORTED_FORMATS.items():
            if extension in extensions:
                return format_type
        return None

    def _map_mime_to_format_type(self, mime_type: str | None) -> str:
        """Map MIME types to internal format categories."""
        if mime_type:
            if "pdf" in mime_type:
                return "pdf"
            if "word" in mime_type or "officedocument" in mime_type:
                return "docx"
            if "image" in mime_type:
                return "image"
            if "video" in mime_type:
                return "video"
            if "audio" in mime_type:
                return "audio"
            if "html" in mime_type or "xml" in mime_type:
                return "web"

        logger.warning(f"Unknown format from MIME type: {mime_type}")
        return "unknown"
