"""
Main pipeline for content extraction
"""
from pathlib import Path
from typing import Optional, Dict, Any
import time
from .format_detector import FormatDetector
from .base_extractor import ExtractionResult
from ..extractors.pdf_extractor import PDFExtractor
from ..extractors.docx_extractor import DOCXExtractor
from ..extractors.ocr_extractor import OCRExtractor
from ..extractors.web_extractor import WebExtractor
# Audio and Video extractors are lazy-loaded (imported when needed)
from ..processors.text_cleaner import TextCleaner
from ..processors.math_extractor import MathExtractor
from ..utils.logger import get_logger
from ..utils.file_handler import FileHandler
from ..config import PRESERVE_LATEX

logger = get_logger(__name__)

class DataIngestionPipeline:
    """Main pipeline for extracting content from various file formats"""
    
    def __init__(self):
        self.format_detector = FormatDetector()
        self.text_cleaner = TextCleaner()
        self.math_extractor = MathExtractor()
        
        # Initialize extractors
        self.extractors = {
            "pdf": PDFExtractor(),
            "docx": DOCXExtractor(),
            "image": OCRExtractor(),  # Uses PaddleOCR by default (from config)
            "web": WebExtractor(),
            "audio": None,  # Lazy load (heavy model)
            "video": None   # Lazy load (heavy model)
        }
        
        logger.info("Data Ingestion Pipeline initialized")
    
    def process_file(self, 
                    file_path: str | Path, 
                    clean_text: bool = True,
                    preserve_latex: bool = PRESERVE_LATEX,
                    **kwargs) -> ExtractionResult:
        """
        Process a file and extract its content
        
        Args:
            file_path: Path to the file
            clean_text: Whether to clean extracted text
            preserve_latex: Preserve LaTeX notation
            **kwargs: Additional arguments for extractors
            
        Returns:
            ExtractionResult object
        """
        file_path = Path(file_path)
        
        logger.info(f"Processing file: {file_path}")
        start_time = time.time()
        
        # Validate file
        if not FileHandler.validate_file(file_path):
            return ExtractionResult(
                text="",
                file_path=str(file_path),
                success=False,
                error="File validation failed"
            )
        
        try:
            # Detect format
            format_info = self.format_detector.detect(file_path)
            format_type = format_info['format_type']
            
            logger.info(f"Detected format: {format_type}")
            
            # Get appropriate extractor
            extractor = self._get_extractor(format_type)
            
            if not extractor:
                return ExtractionResult(
                    text="",
                    file_path=str(file_path),
                    success=False,
                    error=f"No extractor available for format: {format_type}"
                )
            
            # Extract content
            result = extractor.extract(file_path, **kwargs)
            
            # Clean text if requested
            if result.success and clean_text and result.text:
                if preserve_latex:
                    # Preserve math during cleaning
                    preserved_text, math_dict = self.math_extractor.preserve_math(result.text)
                    cleaned = self.text_cleaner.clean(preserved_text, preserve_latex=True)
                    result.text = self.math_extractor.restore_math(cleaned, math_dict)
                else:
                    result.text = self.text_cleaner.clean(result.text, preserve_latex=False)
                
                # Extract math expressions
                math_expressions = self.math_extractor.extract_latex(result.text)
                result.metadata['math_expressions'] = math_expressions
            
            # Add format info to metadata
            result.metadata['format_info'] = format_info
            result.metadata['file_size'] = FileHandler.get_file_size(file_path)
            result.metadata['file_hash'] = FileHandler.get_file_hash(file_path)
            
            total_time = time.time() - start_time
            logger.info(f"Processing completed in {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            return ExtractionResult(
                text="",
                file_path=str(file_path),
                success=False,
                error=str(e)
            )
    
    def process_batch(self, file_paths: list[str | Path], parallel: bool = True,
                      max_workers: int = 4, **kwargs) -> list[ExtractionResult]:
        """
        Process multiple files with optional parallel processing

        Args:
            file_paths: List of file paths
            parallel: Enable parallel processing (default: True)
            max_workers: Maximum number of parallel workers (default: 4)
            **kwargs: Additional arguments for extractors

        Returns:
            List of ExtractionResult objects
        """
        logger.info(f"Processing batch of {len(file_paths)} files (parallel={parallel})")

        if parallel and len(file_paths) > 1:
            # Parallel processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm

            results = [None] * len(file_paths)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(self.process_file, fp, **kwargs): i
                    for i, fp in enumerate(file_paths)
                }

                # Collect results as they complete
                for future in tqdm(as_completed(future_to_index), total=len(file_paths),
                                  desc="Processing files"):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing file {file_paths[index]}: {e}")
                        results[index] = ExtractionResult(
                            text="",
                            metadata={},
                            format_type="unknown",
                            file_path=str(file_paths[index]),
                            extraction_time=0,
                            success=False,
                            error=str(e)
                        )

            return results
        else:
            # Sequential processing
            results = []
            for file_path in file_paths:
                result = self.process_file(file_path, **kwargs)
                results.append(result)

            return results
    
    def _get_extractor(self, format_type: str):
        """Get or initialize extractor for format type"""
        if format_type not in self.extractors:
            logger.error(f"Unknown format type: {format_type}")
            return None
        
        extractor = self.extractors[format_type]
        
        # Lazy load heavy models
        if extractor is None:
            if format_type == "audio":
                logger.info("Loading Audio Extractor (Whisper)...")
                from ..extractors.audio_extractor import AudioExtractor
                self.extractors["audio"] = AudioExtractor()
                extractor = self.extractors["audio"]
            elif format_type == "video":
                logger.info("Loading Video Extractor (Whisper + FFmpeg)...")
                from ..extractors.video_extractor import VideoExtractor
                self.extractors["video"] = VideoExtractor()
                extractor = self.extractors["video"]

        return extractor
