"""
Form recognition and structured data extraction
"""
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FormField:
    """Represents a form field with label and value"""
    label: str
    value: str
    confidence: float
    field_type: str  # 'text', 'number', 'date', 'email', 'phone', 'checkbox'


class FormRecognizer:
    """Recognize and extract structured data from forms"""
    
    # Common form field patterns
    FIELD_PATTERNS = {
        'name': r'(?:name|nome|nom)\s*:?\s*([A-Za-z\s]+)',
        'email': r'(?:email|e-mail|correo)\s*:?\s*([\w\.-]+@[\w\.-]+\.\w+)',
        'phone': r'(?:phone|tel|telephone|telefono)\s*:?\s*([\d\s\-\(\)]+)',
        'date': r'(?:date|fecha|data)\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        'address': r'(?:address|direccion|adresse)\s*:?\s*([A-Za-z0-9\s,\.]+)',
        'id': r'(?:id|identification|dni|passport)\s*:?\s*([A-Z0-9]+)',
        'amount': r'(?:amount|total|suma)\s*:?\s*\$?\s*([\d,\.]+)',
    }
    
    @staticmethod
    def extract_form_fields(text: str) -> List[FormField]:
        """
        Extract structured form fields from text
        
        Args:
            text: Input text from OCR
            
        Returns:
            List of FormField objects
        """
        fields = []
        
        # Try each pattern
        for field_type, pattern in FormRecognizer.FIELD_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                label = match.group(0).split(':')[0].strip()
                value = match.group(1).strip()
                
                # Validate and classify the value
                validated_type, confidence = FormRecognizer._validate_field(
                    value, field_type
                )
                
                if confidence > 0.5:
                    field = FormField(
                        label=label,
                        value=value,
                        confidence=confidence,
                        field_type=validated_type
                    )
                    fields.append(field)
        
        # Detect checkbox fields
        checkbox_fields = FormRecognizer._detect_checkboxes(text)
        fields.extend(checkbox_fields)
        
        # Detect key-value pairs
        kv_fields = FormRecognizer._detect_key_value_pairs(text)
        fields.extend(kv_fields)
        
        return fields
    
    @staticmethod
    def _validate_field(value: str, expected_type: str) -> Tuple[str, float]:
        """Validate field value and return confidence score"""
        value = value.strip()
        
        if expected_type == 'email':
            if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
                return 'email', 0.95
            return 'text', 0.3
        
        elif expected_type == 'phone':
            # Remove formatting
            digits = re.sub(r'[^\d]', '', value)
            if 7 <= len(digits) <= 15:
                return 'phone', 0.9
            return 'text', 0.4
        
        elif expected_type == 'date':
            if re.match(r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', value):
                return 'date', 0.9
            return 'text', 0.3
        
        elif expected_type == 'amount':
            if re.match(r'^\$?\s*[\d,\.]+$', value):
                return 'number', 0.9
            return 'text', 0.4
        
        elif expected_type == 'id':
            if len(value) >= 5 and any(c.isdigit() for c in value):
                return 'id', 0.85
            return 'text', 0.5
        
        return expected_type, 0.7
    
    @staticmethod
    def _detect_checkboxes(text: str) -> List[FormField]:
        """Detect checkbox fields (☐, ☑, ✓, ✗, [X], [ ])"""
        checkbox_patterns = [
            (r'\[([X✓✗])\]\s*([A-Za-z\s]+)', True),
            (r'\[\s\]\s*([A-Za-z\s]+)', False),
            (r'☑\s*([A-Za-z\s]+)', True),
            (r'☐\s*([A-Za-z\s]+)', False),
        ]
        
        fields = []
        for pattern, is_checked in checkbox_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                label = match.group(1 if len(match.groups()) == 1 else 2).strip()
                value = 'checked' if is_checked else 'unchecked'
                
                field = FormField(
                    label=label,
                    value=value,
                    confidence=0.85,
                    field_type='checkbox'
                )
                fields.append(field)
        
        return fields
    
    @staticmethod
    def _detect_key_value_pairs(text: str) -> List[FormField]:
        """Detect generic key-value pairs (Label: Value)"""
        pattern = r'^([A-Za-z\s]+):\s*(.+)$'
        matches = re.finditer(pattern, text, re.MULTILINE)
        
        fields = []
        for match in matches:
            label = match.group(1).strip()
            value = match.group(2).strip()
            
            # Skip if already captured by specific patterns
            if len(label) > 50 or len(value) > 200:
                continue
            
            field = FormField(
                label=label,
                value=value,
                confidence=0.7,
                field_type='text'
            )
            fields.append(field)
        
        return fields
    
    @staticmethod
    def to_structured_dict(fields: List[FormField]) -> Dict:
        """Convert form fields to structured dictionary"""
        result = {}
        
        for field in fields:
            # Use label as key, value as value
            key = field.label.lower().replace(' ', '_')
            result[key] = {
                'value': field.value,
                'type': field.field_type,
                'confidence': field.confidence
            }
        
        return result

