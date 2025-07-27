"""
Utility functions for the Document to Markdown Converter
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results."""
    
    @staticmethod
    def enhance_image_quality(image: Image.Image) -> Image.Image:
        """
        Apply image enhancements to improve OCR accuracy.
        
        Args:
            image: PIL Image to enhance
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return image
            
        except Exception as e:
            logger.warning(f"Error enhancing image: {str(e)}")
            return image
    
    @staticmethod
    def deskew_image(image: Image.Image) -> Image.Image:
        """
        Automatically correct skewed/rotated documents.
        
        Args:
            image: PIL Image to deskew
            
        Returns:
            Deskewed PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = np.rad2deg(theta) - 90
                    angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Only rotate if angle is significant
                    if abs(median_angle) > 0.5:
                        # Rotate image
                        rotated = image.rotate(-median_angle, expand=True, fillcolor='white')
                        logger.info(f"Deskewed image by {median_angle:.2f} degrees")
                        return rotated
            
            return image
            
        except Exception as e:
            logger.warning(f"Error deskewing image: {str(e)}")
            return image
    
    @staticmethod
    def remove_noise(image: Image.Image) -> Image.Image:
        """
        Remove noise from scanned documents.
        
        Args:
            image: PIL Image to denoise
            
        Returns:
            Denoised PIL Image
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
            
            # Convert back to PIL
            denoised_pil = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            
            return denoised_pil
            
        except Exception as e:
            logger.warning(f"Error removing noise: {str(e)}")
            return image
    
    @staticmethod
    def auto_crop_margins(image: Image.Image) -> Image.Image:
        """
        Automatically crop unnecessary margins from document.
        
        Args:
            image: PIL Image to crop
            
        Returns:
            Cropped PIL Image
        """
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Convert to numpy array
            img_array = np.array(gray)
            
            # Find non-white regions
            non_white = img_array < 240
            
            # Find bounding box
            rows = np.any(non_white, axis=1)
            cols = np.any(non_white, axis=0)
            
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # Add small margin
                margin = 20
                rmin = max(0, rmin - margin)
                rmax = min(img_array.shape[0], rmax + margin)
                cmin = max(0, cmin - margin)
                cmax = min(img_array.shape[1], cmax + margin)
                
                # Crop image
                cropped = image.crop((cmin, rmin, cmax, rmax))
                logger.info(f"Auto-cropped image margins")
                return cropped
            
            return image
            
        except Exception as e:
            logger.warning(f"Error cropping margins: {str(e)}")
            return image

class MarkdownPostProcessor:
    """Post-processing utilities for Markdown content."""
    
    @staticmethod
    def clean_markdown(content: str) -> str:
        """
        Clean and standardize Markdown content.
        
        Args:
            content: Raw Markdown content
            
        Returns:
            Cleaned Markdown content
        """
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Fix header spacing
        content = re.sub(r'(#+)\s*(.+)', r'\1 \2', content)
        
        # Fix list formatting
        content = re.sub(r'^\s*[-*+]\s+', '- ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', content, flags=re.MULTILINE)
        
        # Remove trailing whitespace
        content = '\n'.join(line.rstrip() for line in content.split('\n'))
        
        # Ensure single newline at end
        content = content.strip() + '\n'
        
        return content
    
    @staticmethod
    def fix_common_ocr_errors(content: str, language: str = "english") -> str:
        """
        Fix common OCR errors based on language.
        
        Args:
            content: Markdown content with potential OCR errors
            language: Language of the content ("english" or "nepali")
            
        Returns:
            Content with corrected OCR errors
        """
        if language.lower() == "english":
            # Common English OCR corrections
            corrections = {
                r'\b0\b': 'O',  # Zero to O
                r'\bl\b': 'I',  # lowercase l to uppercase I
                r'\brn\b': 'm', # rn to m
                r'\bvv\b': 'w', # double v to w
                r'\b1\b(?=[a-zA-Z])': 'l', # 1 to l before letters
                r'(?<=[a-zA-Z])\b1\b': 'l', # 1 to l after letters
            }
            
            for pattern, replacement in corrections.items():
                content = re.sub(pattern, replacement, content)
        
        elif language.lower() == "nepali":
            # Nepali/Devanagari specific corrections
            # Add Devanagari-specific OCR error corrections here
            pass
        
        return content
    
    @staticmethod
    def add_table_of_contents(content: str) -> str:
        """
        Generate table of contents from headers.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with TOC prepended
        """
        lines = content.split('\n')
        toc_lines = []
        
        for line in lines:
            if line.strip().startswith('#'):
                # Extract header level and text
                match = re.match(r'^(#+)\s*(.+)', line.strip())
                if match:
                    level = len(match.group(1))
                    text = match.group(2).strip()
                    
                    # Create anchor link
                    anchor = re.sub(r'[^\w\s-]', '', text.lower())
                    anchor = re.sub(r'[-\s]+', '-', anchor)
                    
                    # Add to TOC
                    indent = '  ' * (level - 1)
                    toc_lines.append(f"{indent}- [{text}](#{anchor})")
        
        if toc_lines:
            toc = "## Table of Contents\n\n" + '\n'.join(toc_lines) + "\n\n---\n\n"
            return toc + content
        
        return content

class FileValidator:
    """File validation utilities."""
    
    SUPPORTED_IMAGE_FORMATS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
    SUPPORTED_PDF_FORMATS = {'pdf'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    def validate_file(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate uploaded file.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if len(file_bytes) > FileValidator.MAX_FILE_SIZE:
            return False, f"File too large. Maximum size is {FileValidator.MAX_FILE_SIZE // (1024*1024)}MB"
        
        if len(file_bytes) == 0:
            return False, "File is empty"
        
        # Check file extension
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if ext in FileValidator.SUPPORTED_PDF_FORMATS:
            # Validate PDF
            if not file_bytes.startswith(b'%PDF'):
                return False, "Invalid PDF file"
        
        elif ext in FileValidator.SUPPORTED_IMAGE_FORMATS:
            # Validate image
            try:
                Image.open(io.BytesIO(file_bytes))
            except Exception:
                return False, f"Invalid {ext.upper()} image file"
        
        else:
            supported = ', '.join(FileValidator.SUPPORTED_IMAGE_FORMATS | FileValidator.SUPPORTED_PDF_FORMATS)
            return False, f"Unsupported file format. Supported: {supported}"
        
        return True, ""
    
    @staticmethod
    def get_file_info(file_bytes: bytes, filename: str) -> dict:
        """
        Extract file information.
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with file information
        """
        info = {
            'filename': filename,
            'size': len(file_bytes),
            'size_mb': len(file_bytes) / (1024 * 1024),
            'type': 'unknown'
        }
        
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if ext == 'pdf':
            info['type'] = 'pdf'
            # Try to get page count
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(file_bytes, first_page=1, last_page=1)
                # This is a rough estimate - we'd need to actually convert to get exact count
                info['estimated_pages'] = 'Multiple'
            except:
                info['estimated_pages'] = 'Unknown'
        
        elif ext in FileValidator.SUPPORTED_IMAGE_FORMATS:
            info['type'] = 'image'
            info['estimated_pages'] = 1
            
            try:
                image = Image.open(io.BytesIO(file_bytes))
                info['dimensions'] = f"{image.width}x{image.height}"
                info['mode'] = image.mode
            except:
                pass
        
        return info

class APIManager:
    """Enhanced API management with retry logic and rate limiting."""
    
    def __init__(self, api_key: str, max_retries: int = 3, base_delay: float = 1.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.request_count = 0
        self.last_request_time = 0
    
    def rate_limit(self):
        """Implement rate limiting."""
        import time
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Minimum delay between requests
        min_delay = 0.5
        if time_since_last < min_delay:
            time.sleep(min_delay - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def make_request_with_retry(self, request_func, *args, **kwargs):
        """
        Make API request with exponential backoff retry.
        
        Args:
            request_func: Function to call for the API request
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the API request
        """
        import time
        
        for attempt in range(self.max_retries + 1):
            try:
                self.rate_limit()
                result = request_func(*args, **kwargs)
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a retryable error
                retryable_errors = ['rate limit', 'timeout', 'temporary', 'server error']
                is_retryable = any(err in error_msg for err in retryable_errors)
                
                if attempt < self.max_retries and is_retryable:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API request failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed or non-retryable error
                    logger.error(f"API request failed after {attempt + 1} attempts: {str(e)}")
                    raise

class ProgressTracker:
    """Enhanced progress tracking with time estimates."""
    
    def __init__(self):
        self.start_time = None
        self.page_times = []
        self.current_page = 0
        self.total_pages = 0
    
    def start(self, total_pages: int):
        """Start tracking progress."""
        import time
        self.start_time = time.time()
        self.total_pages = total_pages
        self.current_page = 0
        self.page_times = []
    
    def update(self, page_number: int):
        """Update progress for completed page."""
        import time
        current_time = time.time()
        
        if self.start_time and page_number > 0:
            # Calculate time for this page
            if len(self.page_times) == page_number - 1:
                # First page or consecutive page
                page_time = current_time - (self.start_time + sum(self.page_times))
                self.page_times.append(page_time)
        
        self.current_page = page_number
    
    def get_eta(self) -> Optional[str]:
        """Get estimated time to completion."""
        if not self.page_times or self.current_page == 0:
            return None
        
        # Calculate average time per page
        avg_time = sum(self.page_times) / len(self.page_times)
        
        # Estimate remaining time
        remaining_pages = self.total_pages - self.current_page
        eta_seconds = remaining_pages * avg_time
        
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            seconds = int(eta_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        import time
        
        stats = {
            'total_pages': self.total_pages,
            'completed_pages': self.current_page,
            'progress_percent': (self.current_page / self.total_pages * 100) if self.total_pages > 0 else 0,
            'eta': self.get_eta()
        }
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            stats['elapsed_time'] = f"{int(elapsed)}s"
            
            if self.page_times:
                stats['avg_time_per_page'] = f"{sum(self.page_times) / len(self.page_times):.1f}s"
        
        return stats
