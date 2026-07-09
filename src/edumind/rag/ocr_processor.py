"""
OCR Processor Module
Simple interface for loading OCR JSON output from collaborator.
"""

from typing import List, Dict, Any
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Loads OCR JSON output from collaborator.
    """

    def __init__(self):
        """Initialize the OCR Processor."""
        logger.info("OCRProcessor initialized")

    def load_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from OCR JSON file.
        Expected format: List of dictionaries with 'text' and metadata.

        Args:
            json_path: Path to the JSON file from OCR system

        Returns:
            List of document dictionaries
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        # Validate documents
        documents = []
        for idx, doc in enumerate(data):
            if 'text' not in doc:
                logger.warning(f"Document {idx} missing 'text' field, skipping")
                continue
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from JSON")
        return documents


if __name__ == "__main__":
    # Example usage
    processor = OCRProcessor()

    # Load OCR JSON from collaborator
    # documents = processor.load_from_json('path/to/ocr_output.json')
    print("OCRProcessor ready to load JSON files from OCR system")

