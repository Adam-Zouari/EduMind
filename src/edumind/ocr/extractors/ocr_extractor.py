"""
OCR extraction using Tesseract and PaddleOCR with advanced preprocessing
"""
import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.base_extractor import BaseExtractor, ExtractionResult
from ..config import (
    TESSERACT_CMD, OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD,
    TEMP_DIR, OCR_USE_PADDLE, OCR_USE_GPU, OCR_ENABLE_CACHING, OCR_CACHE_DIR,
    OCR_ADAPTIVE_PREPROCESSING, OCR_ROTATION_CORRECTION,
    OCR_PERSPECTIVE_CORRECTION, OCR_QUALITY_THRESHOLD,
    OCR_USE_ANGLE_CLS
)

# Try to import PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: PaddleOCR not available. Only Tesseract will be used.")


import mlflow

class OCRExtractor(BaseExtractor):
    """Extract text from images using OCR with advanced preprocessing and caching"""

    # Class-level cache for model instances (shared across instances)
    _paddle_instance = None
    _cache_lock = None

    def __init__(self, use_paddle: Optional[bool] = None, confidence_threshold: Optional[float] = None,
                 languages: Optional[List[str]] = None, enable_caching: bool = OCR_ENABLE_CACHING):
        super().__init__()
        # Use config default if not explicitly specified
        if use_paddle is None:
            use_paddle = OCR_USE_PADDLE
        self.use_paddle = use_paddle and PADDLE_AVAILABLE
        self.confidence_threshold = confidence_threshold or OCR_CONFIDENCE_THRESHOLD
        self.languages = languages or OCR_LANGUAGES
        self.enable_caching = enable_caching
        self.cache_dir = OCR_CACHE_DIR if enable_caching else None

        # Initialize cache directory
        if self.enable_caching and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if use_paddle and not PADDLE_AVAILABLE:
            self.logger.warning("PaddleOCR requested but not available. Using Tesseract instead.")
            self.use_paddle = False

        if self.use_paddle:
            # Use cached PaddleOCR instance to avoid reloading model
            if OCRExtractor._paddle_instance is None:
                try:
                    import os
                    import paddle

                    # Suppress verbose PaddlePaddle logging
                    os.environ['FLAGS_allocator_strategy'] = 'auto_growth'

                    self.logger.info("Initializing PaddleOCR (cached instance)...")

                    # Set device (new API - use_gpu parameter removed)
                    use_gpu = False
                    if OCR_USE_GPU:
                        try:
                            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                                paddle.device.set_device('gpu:0')
                                use_gpu = True
                            else:
                                paddle.device.set_device('cpu')
                        except:
                            paddle.device.set_device('cpu')
                    else:
                        paddle.device.set_device('cpu')

                    OCRExtractor._paddle_instance = PaddleOCR(
                        use_angle_cls=OCR_USE_ANGLE_CLS,
                        lang='en'
                    )
                    self.logger.info(f"PaddleOCR initialized (angle_cls={OCR_USE_ANGLE_CLS})")
                except Exception as e:
                    self.logger.error(f"Failed to initialize PaddleOCR: {e}")
                    self.logger.warning("Falling back to Tesseract")
                    self.use_paddle = False

            self.paddle_ocr = OCRExtractor._paddle_instance

        if not self.use_paddle:
            # Only set Tesseract path if we're using Tesseract
            self.logger.info(f"Using Tesseract OCR: {TESSERACT_CMD}")
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    def extract(self, file_path: Path, **kwargs) -> ExtractionResult:
        """Extract text from image using OCR with caching and retry logic"""
        start_time = time.time()
        self.logger.info(f"Extracting text via OCR: {file_path}")

        # Check cache first
        if self.enable_caching:
            cached_result = self._get_cached_result(file_path)
            if cached_result:
                self.logger.info(f"Using cached OCR result for {file_path}")
                return cached_result

        try:
            # Assess image quality
            quality_score = self._assess_image_quality(file_path)
            self.logger.info(f"Image quality score: {quality_score:.2f}")

            # Preprocess image with adaptive techniques
            preprocessed, preprocessing_info = self._preprocess_image_advanced(file_path, quality_score)

            # Try extraction with retry logic
            text, confidence, attempt_info = self._extract_with_retry(preprocessed)

            # Validate extraction quality
            is_valid, validation_msg = self._validate_extraction(text, confidence)
            
            ocr_engine = "paddleocr" if self.use_paddle else "tesseract"

            metadata = {
                "ocr_engine": ocr_engine,
                "confidence": confidence,
                "languages": self.languages,
                "quality_score": quality_score,
                "preprocessing": preprocessing_info,
                "extraction_attempts": attempt_info,
                "validation": {"is_valid": is_valid, "message": validation_msg},
                "extractor": "ocr"
            }

            extraction_time = time.time() - start_time
            
            # Log to MLflow if active run exists
            try:
                if mlflow.active_run():
                    mlflow.log_metrics({
                        "ocr_processing_time": extraction_time,
                        "ocr_confidence": float(confidence),
                        "image_quality_score": quality_score
                    })
                    mlflow.log_param("ocr_engine", ocr_engine)
            except Exception as e:
                self.logger.warning(f"Failed to log to MLflow: {e}")

            result = ExtractionResult(
                text=text,
                metadata=metadata,
                format_type="image",
                file_path=str(file_path),
                extraction_time=extraction_time,
                success=is_valid
            )

            # Cache successful results
            if self.enable_caching and is_valid:
                self._cache_result(file_path, result)

            return result

        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return self._create_error_result(file_path, str(e))

    
    def _assess_image_quality(self, file_path: Path) -> float:
        """Assess image quality using Laplacian variance (sharpness)"""
        img = cv2.imread(str(file_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize to 0-100 scale (empirically determined thresholds)
        quality_score = min(100, (laplacian_var / 10) * 100)

        return quality_score

    def _preprocess_image_advanced(self, file_path: Path, quality_score: float) -> Tuple[np.ndarray, Dict]:
        """Advanced preprocessing with adaptive techniques based on image quality"""
        preprocessing_steps = []

        # Read image
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")

        original_img = img.copy()

        # 1. Rotation correction (if enabled)
        if OCR_ROTATION_CORRECTION:
            img, rotation_angle = self._correct_rotation(img)
            if abs(rotation_angle) > 0.5:
                preprocessing_steps.append(f"rotation_corrected_{rotation_angle:.1f}deg")

        # 2. Perspective correction (if enabled and needed)
        if OCR_PERSPECTIVE_CORRECTION:
            img, perspective_corrected = self._correct_perspective(img)
            if perspective_corrected:
                preprocessing_steps.append("perspective_corrected")

        # 3. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        preprocessing_steps.append("grayscale")

        # 4. Adaptive denoising based on quality
        if OCR_ADAPTIVE_PREPROCESSING:
            if quality_score < 50:
                # Low quality: aggressive denoising
                gray = cv2.fastNlMeansDenoising(gray, h=10)
                preprocessing_steps.append("aggressive_denoise")
            elif quality_score < 70:
                # Medium quality: moderate denoising
                gray = cv2.medianBlur(gray, 3)
                preprocessing_steps.append("moderate_denoise")
            else:
                # High quality: minimal denoising
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                preprocessing_steps.append("light_denoise")
        else:
            # Default denoising
            gray = cv2.medianBlur(gray, 3)
            preprocessing_steps.append("default_denoise")

        # 5. Adaptive thresholding
        if quality_score < 60:
            # Use adaptive thresholding for poor quality images
            threshold = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            preprocessing_steps.append("adaptive_threshold")
        else:
            # Use Otsu's thresholding for good quality images
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessing_steps.append("otsu_threshold")

        # 6. Morphological operations for very low quality
        if quality_score < 40:
            kernel = np.ones((2, 2), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            preprocessing_steps.append("morphological_closing")

        preprocessing_info = {
            "steps": preprocessing_steps,
            "quality_score": quality_score
        }

        return threshold, preprocessing_info
    
    def _correct_rotation(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct image rotation using text orientation"""
        try:
            # Convert to grayscale for OSD (Orientation and Script Detection)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Try to detect orientation using Tesseract OSD
            try:
                osd = pytesseract.image_to_osd(gray)
                rotation_angle = int([line for line in osd.split('\n') if 'Rotate' in line][0].split(':')[1].strip())

                if rotation_angle != 0:
                    # Rotate image
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated, rotation_angle
            except:
                # OSD failed, try alternative method using contours
                rotation_angle = self._detect_rotation_contours(gray)
                if abs(rotation_angle) > 0.5:
                    (h, w) = img.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return rotated, rotation_angle

            return img, 0.0
        except Exception as e:
            self.logger.warning(f"Rotation correction failed: {e}")
            return img, 0.0

    def _detect_rotation_contours(self, gray: np.ndarray) -> float:
        """Detect rotation angle using contour analysis"""
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                angles.append(angle)

            # Return median angle
            return np.median(angles) if angles else 0.0

        return 0.0

    def _correct_perspective(self, img: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Detect and correct perspective distortion"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return img, False

            # Find largest contour (likely the document)
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # If we found a quadrilateral, apply perspective transform
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)

                # Compute perspective transform
                (tl, tr, br, bl) = rect
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

                return warped, True

            return img, False
        except Exception as e:
            self.logger.warning(f"Perspective correction failed: {e}")
            return img, False

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _extract_with_retry(self, image: np.ndarray) -> Tuple[str, float, List[str]]:
        """Extract text with retry logic using different configurations"""
        attempts = []

        # Attempt 1: Standard extraction
        try:
            if self.use_paddle and self.paddle_ocr:
                text, confidence = self._extract_with_paddle(image)
            else:
                text, confidence = self._extract_with_tesseract(image)
        except Exception as e:
            self.logger.warning(f"Initial OCR extraction attempt failed: {e}")
            if self.use_paddle:
                self.logger.info("Falling back to Tesseract for this attempt...")
                text, confidence = self._extract_with_tesseract(image)
            else:
                text, confidence = "", 0.0

        attempts.append(f"standard (conf: {confidence:.2f})")

        # If confidence is low, try alternative preprocessing
        if confidence < self.confidence_threshold:
            self.logger.info(f"Low confidence ({confidence:.2f}), trying alternative preprocessing...")

            # Attempt 2: Inverted image (white text on black background)
            inverted = cv2.bitwise_not(image)
            try:
                if self.use_paddle and self.paddle_ocr:
                    text2, confidence2 = self._extract_with_paddle(inverted)
                else:
                    text2, confidence2 = self._extract_with_tesseract(inverted)
            except Exception as e:
                self.logger.warning(f"Alternative OCR extraction attempt failed: {e}")
                if self.use_paddle:
                    self.logger.info("Falling back to Tesseract for alternative attempt...")
                    text2, confidence2 = self._extract_with_tesseract(inverted)
                else:
                    text2, confidence2 = "", 0.0

            attempts.append(f"inverted (conf: {confidence2:.2f})")

            if confidence2 > confidence:
                text, confidence = text2, confidence2

        return text, confidence, attempts

    def _extract_with_tesseract(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text using Tesseract with configurable confidence threshold"""
        # Get text with confidence
        lang_str = '+'.join(self.languages)
        data = pytesseract.image_to_data(
            image,
            lang=lang_str,
            output_type=pytesseract.Output.DICT,
            config='--psm 3'  # Fully automatic page segmentation
        )

        # Filter by confidence
        text_parts = []
        confidences = []

        for i, conf in enumerate(data['conf']):
            if conf > self.confidence_threshold:
                text = data['text'][i].strip()
                if text:
                    text_parts.append(text)
                    confidences.append(conf)

        text = " ".join(text_parts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return text, avg_confidence
    
    def _extract_with_paddle(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text using PaddleOCR with GPU support"""
        try:
            # Comprehensive diagnostics
            self.logger.debug(f"Input image - shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            
            # Image format conversion - PaddleOCR prefers uint8 in BGR format
            processed_image = image
            if image.dtype != np.uint8:
                self.logger.debug(f"Converting image from {image.dtype} to uint8")
                if image.max() <= 1.0:
                    processed_image = (image * 255).astype(np.uint8)
                else:
                    processed_image = image.astype(np.uint8)
            
            # Ensure image is in correct format (HWC with 3 channels)
            if len(processed_image.shape) == 2:
                self.logger.debug("Converting grayscale to BGR")
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            elif processed_image.shape[2] == 4:
                self.logger.debug("Converting RGBA to BGR")
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGBA2BGR)
            
            # Make image contiguous in memory (can prevent some numpy errors)
            processed_image = np.ascontiguousarray(processed_image)
            
            # Initial extraction attempt with detailed error tracking
            try:
                result = self.paddle_ocr.ocr(processed_image)
            except (IndexError, TypeError, ValueError) as e:
                import traceback
                error_trace = traceback.format_exc()
                self.logger.error(f"PaddleOCR internal error: {type(e).__name__}: {e}")
                self.logger.debug(f"Full traceback:\n{error_trace}")
                raise
            
            text_parts = []
            confidences = []

            # First check if result is a list (standard PaddleOCR format)
            if result and isinstance(result, list) and len(result) > 0:
                page_result = result[0]
                
                # Handle PaddlePaddle-X OCRResult object (newer API)
                if page_result and hasattr(page_result, '__class__') and 'OCRResult' in page_result.__class__.__name__:
                    self.logger.debug(f"Detected PaddlePaddle-X OCRResult object")
                    try:
                        # OCRResult is a dictionary-like object with json() method
                        # Try to extract using json() first
                        if hasattr(page_result, 'json') and callable(page_result.json):
                            result_data = page_result.json()
                            self.logger.debug(f"OCRResult json data type: {type(result_data)}")
                            
                            # Parse the JSON data structure
                            if isinstance(result_data, dict):
                                # Look for text in common keys
                                for key in ['rec_text', 'text', 'texts', 'results', 'ocr_results']:
                                    if key in result_data:
                                        data = result_data[key]
                                        if isinstance(data, list):
                                            for item in data:
                                                if isinstance(item, str):
                                                    text_parts.append(item)
                                                    confidences.append(100.0)
                                                elif isinstance(item, dict) and 'text' in item:
                                                    text_parts.append(str(item['text']))
                                                    conf = item.get('score', item.get('confidence', 1.0)) * 100
                                                    confidences.append(conf)
                                        elif isinstance(data, str):
                                            text_parts.append(data)
                                            confidences.append(100.0)
                                        break
                                
                                # If still no text, log the keys to help debug
                                if not text_parts:
                                    self.logger.warning(f"No text found in OCRResult JSON. Keys: {list(result_data.keys())}")
                                    self.logger.debug(f"OCRResult JSON sample: {str(result_data)[:500]}")
                        
                        # Fallback: try accessing as dictionary
                        elif hasattr(page_result, 'items'):
                            for key, value in page_result.items():
                                if isinstance(value, str) and len(value) > 0:
                                    text_parts.append(value)
                                    confidences.append(100.0)
                                elif isinstance(value, list):
                                    for item in value:
                                        if isinstance(item, str):
                                            text_parts.append(item)
                                            confidences.append(100.0)
                        
                        # Last resort: try str() method
                        elif hasattr(page_result, 'str') and callable(page_result.str):
                            result_str = page_result.str()
                            if result_str and len(result_str) > 10:
                                text_parts.append(result_str)
                                confidences.append(100.0)
                        
                        if not text_parts:
                            # Log available attributes to help debug
                            attrs = [attr for attr in dir(page_result) if not attr.startswith('_')]
                            self.logger.warning(f"Unable to extract text from OCRResult. Available attributes: {attrs}")
                    except Exception as e:
                        import traceback
                        self.logger.error(f"Failed to parse PaddlePaddle-X OCRResult: {e}")
                        self.logger.debug(traceback.format_exc())
                
                # Handle standard list format
                elif page_result and isinstance(page_result, list):
                    for line in page_result:
                        try:
                            # Standard format: [box, (text, confidence)]
                            if isinstance(line, (list, tuple)) and len(line) >= 2:
                                content = line[1]
                                if isinstance(content, (list, tuple)) and len(content) >= 2:
                                    text = content[0]
                                    conf = content[1]
                                    
                                    if conf * 100 > self.confidence_threshold:
                                        text_parts.append(str(text))
                                        confidences.append(float(conf) * 100)
                                else:
                                    self.logger.debug(f"Non-standard line content format: {content}")
                            else:
                                self.logger.debug(f"Non-standard line format: {line}")
                        except (IndexError, TypeError) as e:
                            self.logger.debug(f"Failed to parse PaddleOCR line element: {e}")
                            continue
                elif page_result is not None:
                    self.logger.warning(f"Unexpected page result type: {type(page_result)}")

            text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_confidence
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
            raise  # Re-raise to be caught by the caller

    def _validate_extraction(self, text: str, confidence: float) -> Tuple[bool, str]:
        """Validate extraction quality"""
        if not text or len(text.strip()) == 0:
            return False, "No text extracted"

        if confidence < self.confidence_threshold:
            return False, f"Low confidence: {confidence:.2f} < {self.confidence_threshold}"

        # Check for minimum meaningful content
        words = text.split()
        if len(words) < 3:
            return False, f"Too few words extracted: {len(words)}"

        # Check for excessive special characters (sign of poor OCR)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.5:
            return False, f"Too many special characters: {special_char_ratio:.2%}"

        return True, "Extraction validated successfully"

    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key from file path and modification time"""
        stat = file_path.stat()
        key_str = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_result(self, file_path: Path) -> Optional[ExtractionResult]:
        """Retrieve cached OCR result if available"""
        if not self.cache_dir:
            return None

        try:
            cache_key = self._get_cache_key(file_path)
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Reconstruct ExtractionResult
                return ExtractionResult(
                    text=data['text'],
                    metadata=data['metadata'],
                    format_type=data['format_type'],
                    file_path=data['file_path'],
                    extraction_time=data['extraction_time'],
                    success=data['success'],
                    error=data.get('error')
                )
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")

        return None

    def _cache_result(self, file_path: Path, result: ExtractionResult):
        """Cache OCR result for future use"""
        if not self.cache_dir:
            return

        try:
            cache_key = self._get_cache_key(file_path)
            cache_file = self.cache_dir / f"{cache_key}.json"

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            self.logger.debug(f"Cached result for {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {e}")