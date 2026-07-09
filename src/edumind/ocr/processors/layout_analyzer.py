"""
Layout analysis for document structure preservation
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextBlock:
    """Represents a block of text with position and content"""
    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float
    block_type: str  # 'title', 'paragraph', 'list', 'table', 'caption'


class LayoutAnalyzer:
    """Analyze document layout to preserve structure"""
    
    @staticmethod
    def analyze_layout(image: np.ndarray, ocr_data: Dict) -> List[TextBlock]:
        """
        Analyze document layout from OCR data
        
        Args:
            image: Input image
            ocr_data: OCR data from Tesseract (image_to_data output)
            
        Returns:
            List of TextBlock objects with structure information
        """
        blocks = []
        
        # Group words into text blocks
        current_block = None
        
        for i in range(len(ocr_data['text'])):
            if int(ocr_data['conf'][i]) < 0:
                continue
            
            text = ocr_data['text'][i].strip()
            if not text:
                continue
            
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            conf = ocr_data['conf'][i]
            
            # Determine block type based on position and formatting
            block_type = LayoutAnalyzer._classify_block_type(
                text, x, y, w, h, image.shape
            )
            
            block = TextBlock(x, y, w, h, text, conf, block_type)
            blocks.append(block)
        
        # Sort blocks by reading order (top to bottom, left to right)
        blocks = LayoutAnalyzer._sort_reading_order(blocks)
        
        return blocks
    
    @staticmethod
    def _classify_block_type(text: str, x: int, y: int, w: int, h: int, 
                            image_shape: Tuple) -> str:
        """Classify text block type based on features"""
        img_height, img_width = image_shape[:2]
        
        # Title detection (large text, near top, centered)
        if y < img_height * 0.2 and h > 30:
            return 'title'
        
        # List detection (starts with bullet or number)
        if text.startswith(('•', '-', '*', '1.', '2.', '3.')):
            return 'list'
        
        # Caption detection (small text, near bottom or near images)
        if h < 15 and (y > img_height * 0.8 or 'Figure' in text or 'Table' in text):
            return 'caption'
        
        # Default to paragraph
        return 'paragraph'
    
    @staticmethod
    def _sort_reading_order(blocks: List[TextBlock]) -> List[TextBlock]:
        """Sort blocks in natural reading order"""
        # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
        return sorted(blocks, key=lambda b: (b.y // 20, b.x))
    
    @staticmethod
    def reconstruct_text_with_structure(blocks: List[TextBlock]) -> str:
        """Reconstruct text preserving document structure"""
        output = []
        current_type = None
        
        for block in blocks:
            # Add spacing based on block type transitions
            if current_type and current_type != block.block_type:
                output.append('\n')
            
            # Format based on block type
            if block.block_type == 'title':
                output.append(f"\n# {block.text}\n")
            elif block.block_type == 'list':
                output.append(f"  {block.text}\n")
            elif block.block_type == 'caption':
                output.append(f"\n*{block.text}*\n")
            else:  # paragraph
                output.append(f"{block.text} ")
            
            current_type = block.block_type
        
        return ''.join(output).strip()
    
    @staticmethod
    def detect_columns(blocks: List[TextBlock], image_width: int) -> int:
        """Detect number of columns in document"""
        if not blocks:
            return 1
        
        # Analyze x-coordinates to find column boundaries
        x_positions = [b.x for b in blocks]
        
        # Use clustering to find columns
        # Simple approach: check for gaps in x-positions
        x_sorted = sorted(set(x_positions))
        gaps = []
        
        for i in range(len(x_sorted) - 1):
            gap = x_sorted[i + 1] - x_sorted[i]
            if gap > image_width * 0.1:  # Significant gap
                gaps.append(gap)
        
        # Number of columns = number of significant gaps + 1
        return len(gaps) + 1

