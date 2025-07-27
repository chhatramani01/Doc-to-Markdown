import streamlit as st
import google.generativeai as genai
from pdf2image import convert_from_bytes
from PIL import Image
import io
import base64
import time
import os
from typing import List, Optional, Tuple
import zipfile
from docx import Document
import tempfile
import logging
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

# Import utility modules (assuming they're in the same directory)
try:
    from utils import (
        ImagePreprocessor, MarkdownPostProcessor, 
        FileValidator, APIManager, ProgressTracker
    )
except ImportError:
    # If utils module isn't available, create minimal stubs
    class ImagePreprocessor:
        @staticmethod
        def enhance_image_quality(image): return image
        @staticmethod
        def deskew_image(image): return image
        @staticmethod
        def remove_noise(image): return image
        @staticmethod
        def auto_crop_margins(image): return image
    
    class MarkdownPostProcessor:
        @staticmethod
        def clean_markdown(content): return content
        @staticmethod
        def fix_common_ocr_errors(content, language="english"): return content
        @staticmethod
        def add_table_of_contents(content): return content
    
    class FileValidator:
        @staticmethod
        def validate_file(file_bytes, filename): return True, ""
        @staticmethod
        def get_file_info(file_bytes, filename): return {"filename": filename, "size": len(file_bytes)}
    
    class APIManager:
        def __init__(self, api_key, **kwargs): pass
        def make_request_with_retry(self, func, *args, **kwargs): return func(*args, **kwargs)
    
    class ProgressTracker:
        def start(self, total): pass
        def update(self, page): pass
        def get_eta(self): return     def process_document(self, file_bytes: bytes, file_type: str, language: Language, 
                        status_callback=None, enable_toc: bool = False) -> Tuple[str, ProcessingStatus]:
        """Process entire document and return Markdown content."""
        status = ProcessingStatus()
        markdown_content = []
        
        try:
            # Convert to images
            if file_type == "pdf":
                images = self.pdf_to_images(file_bytes)
            else:
                # Single image file
                image = Image.open(io.BytesIO(file_bytes))
                images = [image]
            
            status.total_pages = len(images)
            self.progress_tracker.start(len(images))
            
            if status_callback:
                status_callback(status)
            
            # Process each page
            for i, image in enumerate(images):
                status.current_page = i + 1
                status.status_message = f"Processing page {i + 1} of {len(images)}..."
                
                # Update progress tracker
                self.progress_tracker.update(i + 1)
                stats = self.progress_tracker.get_stats()
                status.processing_stats = stats
                
                if status_callback:
                    status_callback(status)
                
                # Preprocess image
                processed_image = self.preprocess_image(image)
                
                # Skip blank pages
                if self.is_blank_page(processed_image):
                    logger.info(f"Skipping blank page {i + 1}")
                    status.processed_pages += 1
                    continue
                
                # Check image quality
                quality_score = self.get_image_quality_score(processed_image)
                if quality_score < 0.2:
                    logger.warning(f"Low quality detected for page {i + 1} (score: {quality_score:.2f})")
                
                # Process with Gemini
                page_content = self.process_image_with_gemini(processed_image, language)
                
                if page_content:
                    # Post-process the content
                    cleaned_content = self.postprocessor.clean_markdown(page_content)
                    fixed_content = self.postprocessor.fix_common_ocr_errors(
                        cleaned_content, language.value.lower()
                    )
                    
                    markdown_content.append(fixed_content)
                    status.processed_pages += 1
                    logger.info(f"Successfully processed page
        def get_stats(self): return {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Language(Enum):
    ENGLISH = "English"
    NEPALI = "Nepali"

class OutputFormat(Enum):
    MARKDOWN = "md"
    TEXT = "txt"
    DOCX = "docx"

@dataclass
class ProcessingStatus:
    total_pages: int = 0
    processed_pages: int = 0
    failed_pages: int = 0
    current_page: int = 0
    status_message: str = ""
    processing_stats: dict = None
    
    def __post_init__(self):
        if self.processing_stats is None:
            self.processing_stats = {}
    
    @property
    def progress_percentage(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100

class DocumentProcessor:
    def __init__(self, api_key: str, enable_preprocessing: bool = True):
        """Initialize the document processor with Gemini API key."""
        self.api_key = api_key
        self.enable_preprocessing = enable_preprocessing
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.api_manager = APIManager(api_key, max_retries=3, base_delay=1.0)
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = MarkdownPostProcessor()
        self.progress_tracker = ProgressTracker()
        
    def get_language_prompt(self, language: Language) -> str:
        """Get language-specific prompt for Gemini API."""
        base_prompt = """
You are an expert document digitization assistant. Your task is to convert this scanned document image into clean, properly structured Markdown format.

CRITICAL REQUIREMENTS:
1. **Preserve Structure**: Maintain all headings, subheadings, paragraphs, lists, and formatting hierarchy
2. **Clean Output**: Remove artifacts, noise, and scanning imperfections
3. **Accurate Text**: Ensure high OCR accuracy and fix obvious scanning errors
4. **Markdown Formatting**: Use proper Markdown syntax (# ## ### for headings, * for lists, etc.)
5. **Layout Preservation**: Maintain the logical flow and organization of the original document

FORMATTING RULES:
- Use # for main titles, ## for sections, ### for subsections
- Preserve bullet points and numbered lists with proper indentation
- Maintain paragraph breaks and spacing
- Handle tables with proper Markdown table syntax if present
- Preserve emphasis (bold/italic) where clearly indicated
- Skip headers/footers unless they contain important content
- If you see mathematical equations or formulas, format them appropriately
- For code blocks or technical content, use proper code formatting
"""
        
        language_specific = {
            Language.ENGLISH: """
6. **Language**: Process English text with attention to proper grammar and spelling
7. **Technical Terms**: Preserve technical terminology and proper nouns accurately
8. **Formatting**: Use standard English punctuation and capitalization rules
9. **Context**: Understand context to fix common OCR errors (e.g., 'rn' → 'm', '0' → 'O')

Return ONLY the Markdown content, no explanations or metadata.
""",
            Language.NEPALI: """
6. **Language**: Process Nepali text (देवनागरी script) with attention to proper grammar and spelling
7. **Script Accuracy**: Ensure accurate Devanagari character recognition and proper Unicode encoding
8. **Cultural Context**: Preserve Nepali linguistic conventions and formatting standards
9. **Mixed Content**: Handle English words/numbers within Nepali text appropriately
10. **Diacritics**: Pay special attention to correct diacritical marks and conjunct consonants

Return ONLY the Markdown content in proper Devanagari script, no explanations or metadata.
"""
        }
        
        return base_prompt + language_specific[language]
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy."""
        if not self.enable_preprocessing:
            return image
        
        try:
            # Apply preprocessing steps
            image = self.preprocessor.auto_crop_margins(image)
            image = self.preprocessor.deskew_image(image)
            image = self.preprocessor.remove_noise(image)
            image = self.preprocessor.enhance_image_quality(image)
            
            return image
        except Exception as e:
            logger.warning(f"Error in preprocessing: {str(e)}")
            return image
    
    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF bytes to list of PIL Images."""
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG')
            logger.info(f"Successfully converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    def is_blank_page(self, image: Image.Image, threshold: float = 0.95) -> bool:
        """Check if an image is mostly blank/white."""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            # Calculate percentage of white pixels
            pixels = list(gray.getdata())
            white_pixels = sum(1 for pixel in pixels if pixel > 240)
            white_ratio = white_pixels / len(pixels)
            return white_ratio > threshold
        except Exception as e:
            logger.warning(f"Error checking blank page: {str(e)}")
            return False
    
    def get_image_quality_score(self, image: Image.Image) -> float:
        """Calculate image quality score for processing optimization."""
        try:
            # Convert to grayscale
            gray = image.convert('L')
            
            # Calculate various quality metrics
            import numpy as np
            img_array = np.array(gray)
            
            # Contrast (standard deviation)
            contrast = np.std(img_array)
            
            # Sharpness (using Laplacian variance)
            from scipy import ndimage
            laplacian_var = ndimage.laplace(img_array).var()
            
            # Combine metrics (normalize to 0-1 scale)
            quality_score = min(1.0, (contrast / 50.0 + laplacian_var / 1000.0) / 2.0)
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Error calculating image quality: {str(e)}")
            return 0.5  # Default medium quality
    
    def process_image_with_gemini(self, image: Image.Image, language: Language) -> Optional[str]:
        """Process a single image with Gemini API."""
        def _make_request():
            prompt = self.get_language_prompt(language)
            response = self.model.generate_content([prompt, image])
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini API")
                return None
        
        try:
            # Use API manager for retry logic
            result = self.api_manager.make_request_with_retry(_make_request)
            return result
                
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            return None
    
                    markdown_content.append(fixed_content)
                    status.processed_pages += 1
                    logger.info(f"Successfully processed page {i + 1} (quality: {quality_score:.2f})")
                else:
                    status.failed_pages += 1
                    logger.warning(f"Failed to process page {i + 1}")
                    markdown_content.append(f"<!-- Page {i + 1}: Processing failed -->")
                
                # Small delay to avoid rate limiting (handled by APIManager now)
                time.sleep(0.1)
            
            status.status_message = "Processing complete!"
            
            # Combine all pages
            final_content = "\n\n---\n\n".join(filter(None, markdown_content))
            
            # Add table of contents if requested
            if enable_toc and final_content:
                final_content = self.postprocessor.add_table_of_contents(final_content)
            
            # Final cleanup
            final_content = self.postprocessor.clean_markdown(final_content)
            
            return final_content, status
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            status.status_message = f"Error: {str(e)}"
            raise

def convert_to_format(markdown_content: str, output_format: OutputFormat) -> bytes:
    """Convert markdown content to specified format."""
    if output_format == OutputFormat.MARKDOWN:
        return markdown_content.encode('utf-8')
    
    elif output_format == OutputFormat.TEXT:
        # Simple markdown to text conversion
        import re
        text = markdown_content
        # Remove markdown formatting
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'---+', '', text)  # Remove separators
        return text.encode('utf-8')
    
    elif output_format == OutputFormat.DOCX:
        # Convert to DOCX
        doc = Document()
        
        lines = markdown_content.split('\n')
        for line in lines.strip():
            line = line.strip()
            if not line or line == '---':
                continue
                
            if line.startswith('###'):
                # H3
                doc.add_heading(line[3:].strip(), level=3)
            elif line.startswith('##'):
                # H2
                doc.add_heading(line[2:].strip(), level=2)
            elif line.startswith('#'):
                # H1
                doc.add_heading(line[1:].strip(), level=1)
            else:
                # Regular paragraph
                doc.add_paragraph(line)
        
        # Save to bytes
        doc_buffer = io.BytesIO()
        doc.save(doc_buffer)
        doc_buffer.seek(0)
        return doc_buffer.getvalue()

def main():
    st.set_page_config(
        page_title="Document to Markdown Converter",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    .feature-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📄 Document to Markdown Converter")
    st.markdown("Convert scanned PDFs and images to clean Markdown using **Gemini 2.5 Flash Lite**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'current_processing' not in st.session_state:
        st.session_state.current_processing = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key",
            placeholder="Enter your API key here..."
        )
        
        if api_key:
            st.success("✅ API Key configured")
        
        st.markdown("---")
        
        # Language selection
        language = st.selectbox(
            "🌐 Document Language",
            options=[Language.ENGLISH, Language.NEPALI],
            format_func=lambda x: f"{x.value} {'🇺🇸' if x == Language.ENGLISH else '🇳🇵'}",
            help="Select the primary language of your documents"
        )
        
        # Processing options
        st.subheader("🔧 Processing Options")
        
        enable_preprocessing = st.checkbox(
            "📸 Enable Image Preprocessing",
            value=True,
            help="Apply image enhancement, deskewing, and noise reduction"
        )
        
        enable_toc = st.checkbox(
            "📑 Generate Table of Contents",
            value=False,
            help="Automatically generate TOC from document headings"
        )
        
        # Output format selection
        output_formats = st.multiselect(
            "📁 Output Formats",
            options=[OutputFormat.MARKDOWN, OutputFormat.TEXT, OutputFormat.DOCX],
            default=[OutputFormat.MARKDOWN],
            format_func=lambda x: f".{x.value} {get_format_icon(x)}",
            help="Choose which formats to generate"
        )
        
        st.markdown("---")
        
        # Quality settings
        st.subheader("⚡ Quality Settings")
        
        quality_preset = st.selectbox(
            "Quality Preset",
            options=["Fast", "Balanced", "High Quality"],
            index=1,
            help="Balance between speed and accuracy"
        )
        
        # Advanced settings expander
        with st.expander("🛠️ Advanced Settings"):
            blank_threshold = st.slider(
                "Blank Page Threshold",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Threshold for detecting blank pages (higher = more strict)"
            )
            
            max_retries = st.number_input(
                "API Retry Attempts",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of retry attempts for failed API calls"
            )
        
        st.markdown("---")
        
        # Instructions
        st.markdown("### 📋 Quick Start")
        st.markdown("""
        1. **🔑 Enter your Gemini API key**
        2. **🌐 Select document language**
        3. **📁 Upload your files**
        4. **⚙️ Configure options**
        4. **🚀 Click 'Process Documents'**
        5. **💾 Download results**
        """)
        
        # API key help
        with st.expander("❓ How to get API key"):
            st.markdown("""
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Click "Create API Key"
            3. Copy the generated key
            4. Paste it in the field above
            """)

def get_format_icon(format_type: OutputFormat) -> str:
    """Get emoji icon for file format."""
    icons = {
        OutputFormat.MARKDOWN: "📝",
        OutputFormat.TEXT: "📄",
        OutputFormat.DOCX: "📘"
    }
    return icons.get(format_type, "📁")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 File Upload")
      
