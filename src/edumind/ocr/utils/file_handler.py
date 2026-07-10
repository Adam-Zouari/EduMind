"""File handling utilities."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Utility class for filesystem-related OCR helpers."""

    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """Generate the MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Return the size of a file in bytes."""
        return file_path.stat().st_size

    @staticmethod
    def get_file_identity_string(file_path: Path) -> str:
        """Return a stable identity string for cache-key generation."""
        stat = file_path.stat()
        return "|".join(
            [
                str(file_path.resolve()),
                str(stat.st_mtime),
                str(stat.st_size),
            ]
        )

    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Create a directory if it does not already exist."""
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clean_temp_files(temp_dir: Path) -> None:
        """Remove a temporary directory tree when it exists."""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned temporary directory: {temp_dir}")

    @staticmethod
    def validate_file(file_path: Path) -> bool:
        """Validate that a file exists, is a regular file, and is readable."""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
        if not os.access(file_path, os.R_OK):
            logger.error(f"File is not readable: {file_path}")
            return False
        return True
