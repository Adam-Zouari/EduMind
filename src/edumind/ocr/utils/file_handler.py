"""
File handling utilities
"""
import os
import shutil
from pathlib import Path
from typing import Optional
import hashlib
from .logger import get_logger

logger = get_logger(__name__)

class FileHandler:
    """Utility class for file operations"""
    
    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """Generate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Create directory if it doesn't exist"""
        directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def clean_temp_files(temp_dir: Path) -> None:
        """Remove temporary files"""
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned temporary directory: {temp_dir}")
    
    @staticmethod
    def validate_file(file_path: Path) -> bool:
        """Validate if file exists and is readable"""
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