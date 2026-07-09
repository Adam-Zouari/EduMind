"""
Text cleaning and normalization with context-aware OCR error correction
"""
import re
import ftfy
from unidecode import unidecode
from typing import List, Dict, Tuple
from ..utils.logger import get_logger
from ..config import REMOVE_HEADERS_FOOTERS, NORMALIZE_WHITESPACE, MIN_TEXT_LENGTH

logger = get_logger(__name__)

# Expanded OCR error dictionary with context patterns
OCR_ERROR_PATTERNS = {
    # Character substitutions (context-aware)
    'letter_to_letter': {
        r'\brn\b': 'm',  # 'rn' looks like 'm'
        r'\bvv\b': 'w',  # 'vv' looks like 'w'
        r'\bcl\b': 'd',  # 'cl' looks like 'd'
        r'\bII\b': 'll',  # 'II' (capital i) looks like 'll'
    },
    # Number-like patterns (only in non-numeric contexts)
    'ambiguous_chars': {
        'O': '0',  # Letter O to zero (in numeric context)
        'l': '1',  # Letter l to one (in numeric context)
        'I': '1',  # Letter I to one (in numeric context)
        'S': '5',  # Letter S to five (in numeric context)
        'Z': '2',  # Letter Z to two (in numeric context)
    },
    # Common word corrections
    'common_words': {
        r'\btlie\b': 'the',
        r'\btbe\b': 'the',
        r'\banci\b': 'and',
        r'\bwlth\b': 'with',
        r'\bfrom\b': 'from',
        r'\bthls\b': 'this',
        r'\bthat\b': 'that',
        r'\bwhlch\b': 'which',
        r'\bwlll\b': 'will',
        r'\bcan\b': 'can',
        r'\bhas\b': 'has',
        r'\bhave\b': 'have',
    },
    # Punctuation errors
    'punctuation': {
        r'\s+([.,!?;:])': r'\1',  # Remove space before punctuation
        r'([.,!?;:])\s*([.,!?;:])': r'\1',  # Remove duplicate punctuation
        r',,': ',',
        r'\.\.': '.',
    }
}

class TextCleaner:
    """Clean and normalize extracted text with context-aware OCR error correction"""

    @staticmethod
    def clean(text: str, preserve_latex: bool = True, aggressive_ocr_fix: bool = False) -> str:
        """
        Clean and normalize text with improved OCR error correction

        Args:
            text: Raw extracted text
            preserve_latex: Keep LaTeX notation
            aggressive_ocr_fix: Apply aggressive OCR error correction (may over-correct)

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Fix encoding issues
        text = ftfy.fix_text(text)

        # Remove headers/footers (common patterns)
        if REMOVE_HEADERS_FOOTERS:
            text = TextCleaner._remove_headers_footers(text)

        # Normalize whitespace
        if NORMALIZE_WHITESPACE:
            text = TextCleaner._normalize_whitespace(text)

        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove multiple dashes/underscores (often used as dividers)
        text = re.sub(r'[-_]{4,}', '', text)

        # Fix common OCR errors (context-aware)
        text = TextCleaner._fix_ocr_errors_advanced(text, aggressive=aggressive_ocr_fix)

        # Preserve or remove LaTeX
        if not preserve_latex:
            text = re.sub(r'\$.*?\$', '', text)
            text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)

        return text.strip()
    
    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """Remove common header/footer patterns"""
        lines = text.split('\n')
        
        # Common header/footer patterns
        patterns = [
            r'^\s*page\s+\d+',
            r'^\s*\d+\s*$',
            r'confidential',
            r'proprietary',
            r'copyright\s+©',
        ]
        
        cleaned_lines = []
        for line in lines:
            is_header_footer = False
            for pattern in patterns:
                if re.search(pattern, line.lower()):
                    is_header_footer = True
                    break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text
    
    @staticmethod
    def _fix_ocr_errors(text: str) -> str:
        """Legacy OCR error fix (kept for backward compatibility)"""
        return TextCleaner._fix_ocr_errors_advanced(text, aggressive=False)

    @staticmethod
    def _fix_ocr_errors_advanced(text: str, aggressive: bool = False) -> str:
        """
        Fix common OCR errors with context awareness

        Args:
            text: Input text
            aggressive: If True, apply more corrections (may over-correct)
        """
        # 1. Fix letter-to-letter substitutions (safe)
        for pattern, replacement in OCR_ERROR_PATTERNS['letter_to_letter'].items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 2. Fix common word errors (safe)
        for pattern, replacement in OCR_ERROR_PATTERNS['common_words'].items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 3. Fix punctuation errors (safe)
        for pattern, replacement in OCR_ERROR_PATTERNS['punctuation'].items():
            text = re.sub(pattern, replacement, text)

        # 4. Context-aware number/letter corrections (only if aggressive)
        if aggressive:
            text = TextCleaner._fix_ambiguous_characters(text)

        return text

    @staticmethod
    def _fix_ambiguous_characters(text: str) -> str:
        """
        Fix ambiguous characters (O/0, l/1, I/1) based on context
        Only applies corrections in appropriate contexts
        """
        words = text.split()
        corrected_words = []

        for word in words:
            # Check if word looks like a number
            if TextCleaner._is_likely_number(word):
                # Apply letter-to-number corrections
                corrected = word
                for letter, number in OCR_ERROR_PATTERNS['ambiguous_chars'].items():
                    corrected = corrected.replace(letter, number)
                corrected_words.append(corrected)
            else:
                # Keep as is (likely a word)
                corrected_words.append(word)

        return ' '.join(corrected_words)

    @staticmethod
    def _is_likely_number(word: str) -> bool:
        """
        Determine if a word is likely meant to be a number

        Heuristics:
        - Contains mostly digits
        - Contains O, l, I mixed with digits
        - Matches common number patterns (dates, IDs, etc.)
        """
        # Remove common punctuation from numbers
        cleaned = word.replace(',', '').replace('.', '').replace('-', '').replace('/', '')

        if not cleaned:
            return False

        # Count digits and ambiguous characters
        digit_count = sum(1 for c in cleaned if c.isdigit())
        ambiguous_count = sum(1 for c in cleaned if c in 'OlISZ')
        total_chars = len(cleaned)

        # If mostly digits or digits + ambiguous chars, likely a number
        if (digit_count + ambiguous_count) / total_chars > 0.7:
            return True

        # Check for common number patterns
        number_patterns = [
            r'^\d{1,2}[Ol]\d+$',  # e.g., "2O23" (2023)
            r'^\d+[Ol]$',  # e.g., "10O" (100)
            r'^[Ol]\d+$',  # e.g., "O5" (05)
            r'^\d+[lI]\d+$',  # e.g., "1l5" (115)
        ]

        for pattern in number_patterns:
            if re.match(pattern, word):
                return True

        return False
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > MIN_TEXT_LENGTH]