"""
PDF extraction using PyMuPDF and pdfplumber
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any
import time
from ..core.base_extractor import BaseExtractor, ExtractionResult

class PDFExtractor(BaseExtractor):
    """Extract text and metadata from PDF files"""
    
    def __init__(self, preserve_layout: bool = True):
        super().__init__()
        self.preserve_layout = preserve_layout
    
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """Extract text from PDF"""
        start_time = time.time()
        self.logger.info(f"Extracting PDF: {file_path}")
        
        try:
            # Try PyMuPDF first (faster)
            text, metadata = self._extract_with_pymupdf(file_path)
            
            # If text is minimal, try pdfplumber (better for tables)
            if len(text.strip()) < 100:
                self.logger.info("Trying pdfplumber for better extraction")
                plumber_text, plumber_meta = self._extract_with_pdfplumber(file_path)
                if len(plumber_text) > len(text):
                    text = plumber_text
                    metadata.update(plumber_meta)
            
            extraction_time = time.time() - start_time
            
            return ExtractionResult(
                text=text,
                metadata=metadata,
                format_type="pdf",
                file_path=str(file_path),
                extraction_time=extraction_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _extract_with_pymupdf(self, file_path: Path) -> tuple[str, Dict]:
        """Extract using PyMuPDF"""
        doc = fitz.open(file_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            if self.preserve_layout:
                text_parts.append(page.get_text("text", sort=True))
            else:
                text_parts.append(page.get_text())
        
        text = "\n\n".join(text_parts)
        
        metadata = {
            "num_pages": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "extractor": "pymupdf"
        }
        
        doc.close()
        return text, metadata
    
    def _extract_with_pdfplumber(self, file_path: Path) -> tuple[str, Dict]:
        """Extract using pdfplumber (better for tables)"""
        text_parts = []
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                
                # Extract tables
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
            
            metadata = {
                "num_pages": len(pdf.pages),
                "num_tables": len(tables),
                "extractor": "pdfplumber"
            }
        
        text = "\n\n".join(text_parts)
        
        # Add table data to text
        if tables:
            text += "\n\n--- TABLES ---\n\n"
            for i, table in enumerate(tables):
                text += f"\nTable {i+1}:\n"
                for row in table:
                    text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
        
        return text, metadata