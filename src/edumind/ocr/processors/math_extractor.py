"""
Extract and preserve mathematical notation
"""
import re
from typing import List, Dict
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MathExtractor:
    """Extract and preserve LaTeX mathematical notation"""
    
    @staticmethod
    def extract_latex(text: str) -> Dict[str, List[str]]:
        """
        Extract LaTeX expressions from text
        
        Returns:
            Dict with 'inline' and 'display' math expressions
        """
        # Inline math: $...$
        inline_math = re.findall(r'\$([^\$]+)\$', text)
        
        # Display math: $$...$$
        display_math = re.findall(r'\$\$([^\$]+)\$\$', text)
        
        # LaTeX environments: \begin{equation}...\end{equation}
        equation_math = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', text, re.DOTALL)
        
        # LaTeX environments: \begin{align}...\end{align}
        align_math = re.findall(r'\\begin\{align\}(.*?)\\end\{align\}', text, re.DOTALL)
        
        return {
            "inline": inline_math,
            "display": display_math + equation_math + align_math
        }
    
    @staticmethod
    def preserve_math(text: str) -> str:
        """
        Replace math expressions with placeholders to preserve them during cleaning
        """
        math_dict = {}
        counter = 0
        
        # Find all math expressions
        math_patterns = [
            (r'\$\$.*?\$\$', 'DISPLAYMATH'),
            (r'\$.*?\$', 'INLINEMATH'),
            (r'\\begin\{equation\}.*?\\end\{equation\}', 'EQUATION'),
            (r'\\begin\{align\}.*?\\end\{align\}', 'ALIGN')
        ]
        
        for pattern, prefix in math_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                placeholder = f"__{prefix}_{counter}__"
                math_dict[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                counter += 1
        
        return text, math_dict
    
    @staticmethod
    def restore_math(text: str, math_dict: Dict[str, str]) -> str:
        """Restore math expressions from placeholders"""
        for placeholder, expression in math_dict.items():
            text = text.replace(placeholder, expression)
        return text