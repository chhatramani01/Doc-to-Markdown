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
    
    @property
    def progress_percentage(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100

class DocumentProcessor:
    def __init__(self, api_key: str):
        """Initialize the document processor with Gemini API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
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
"""
        
        language_specific = {
            Language.ENGLISH: """
6. **Language**: Process English text with attention to proper grammar and spelling
7. **Technical Terms**: Preserve technical terminology and proper nouns accurately
8. **Formatting**: Use standard English punctuation and capitalization rules

Return ONLY the Markdown content, no explanations or metadata.
""",
            Language.NEPALI: """
6. **Language**: Process Nepali text (देवनागरी script) with attention to proper grammar and spelling
7. **Script Accuracy**: Ensure accurate Devanagari character recognition and proper Unicode encoding
8. **Cultural Context**: Preserve Nepali linguistic conventions and formatting standards
9. **Mixed Content**: Handle English words/numbers within Nepali text appropriately

Return ONLY the Markdown content in proper Devanagari script, no explanations or metadata.
"""
        }
        
        return base_prompt + language_specific[language]
    
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
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()
    
    def process_image_with_gemini(self, image: Image.Image, language: Language) -> Optional[str]:
        """Process a single image with Gemini API."""
        try:
            prompt = self.get_language_prompt(language)
            
            # Generate content with image
            response = self.model.generate_content([prompt, image])
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini API")
                return None
                
        except Exception as e:
            logger.error(f"Error processing image with Gemini: {str(e)}")
            return None
    
    def process_document(self, file_bytes: bytes, file_type: str, language: Language, 
                        status_callback=None) -> Tuple[str, ProcessingStatus]:
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
            if status_callback:
                status_callback(status)
            
            # Process each page
            for i, image in enumerate(images):
                status.current_page = i + 1
                status.status_message = f"Processing page {i + 1} of {len(images)}..."
                
                if status_callback:
                    status_callback(status)
                
                # Skip blank pages
                if self.is_blank_page(image):
                    logger.info(f"Skipping blank page {i + 1}")
                    status.processed_pages += 1
                    continue
                
                # Process with Gemini
                page_content = self.process_image_with_gemini(image, language)
                
                if page_content:
                    markdown_content.append(page_content)
                    status.processed_pages += 1
                    logger.info(f"Successfully processed page {i + 1}")
                else:
                    status.failed_pages += 1
                    logger.warning(f"Failed to process page {i + 1}")
                    markdown_content.append(f"<!-- Page {i + 1}: Processing failed -->")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            
            status.status_message = "Processing complete!"
            
            # Combine all pages
            final_content = "\n\n---\n\n".join(markdown_content)
            
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
        layout="wide"
    )
    
    st.title("📄 Document to Markdown Converter")
    st.markdown("Convert scanned PDFs and images to clean Markdown using Gemini 2.5 Flash Lite")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        # Language selection
        language = st.selectbox(
            "Document Language",
            options=[Language.ENGLISH, Language.NEPALI],
            format_func=lambda x: x.value
        )
        
        # Output format selection
        output_formats = st.multiselect(
            "Output Formats",
            options=[OutputFormat.MARKDOWN, OutputFormat.TEXT, OutputFormat.DOCX],
            default=[OutputFormat.MARKDOWN],
            format_func=lambda x: f".{x.value}"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Enter your Gemini API key
        2. Select document language
        3. Upload PDF or image files
        4. Choose output formats
        5. Click 'Process Documents'
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF or image files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload scanned PDFs or images to convert to Markdown"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            # Show file details
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size:,} bytes)")
    
    with col2:
        st.header("🎛️ Processing Controls")
        
        process_button = st.button(
            "🚀 Process Documents",
            disabled=not (api_key and uploaded_files),
            type="primary"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key")
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one file")
    
    # Processing section
    if process_button and api_key and uploaded_files:
        try:
            processor = DocumentProcessor(api_key)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_container = st.container()
            results_container = st.container()
            
            all_results = {}
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"📄 Processing: {uploaded_file.name}")
                
                # Determine file type
                file_type = "pdf" if uploaded_file.type == "application/pdf" else "image"
                
                # Status tracking
                def update_status(status: ProcessingStatus):
                    with status_container:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Pages", status.total_pages)
                        with col2:
                            st.metric("Processed", status.processed_pages)
                        with col3:
                            st.metric("Failed", status.failed_pages)
                        
                        if status.total_pages > 0:
                            progress_bar.progress(status.progress_percentage / 100)
                            st.info(f"📊 Progress: {status.progress_percentage:.1f}%")
                        
                        if status.status_message:
                            st.write(f"🔄 {status.status_message}")
                
                # Process document
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_bytes = uploaded_file.read()
                    markdown_content, final_status = processor.process_document(
                        file_bytes, file_type, language, update_status
                    )
                
                # Store results
                all_results[uploaded_file.name] = {
                    'content': markdown_content,
                    'status': final_status
                }
                
                # Show results
                with results_container:
                    st.success(f"✅ Completed: {uploaded_file.name}")
                    
                    # Show preview
                    with st.expander(f"📖 Preview: {uploaded_file.name}"):
                        # Editable text area
                        edited_content = st.text_area(
                            "Edit Markdown (optional)",
                            value=markdown_content,
                            height=300,
                            key=f"edit_{file_idx}"
                        )
                        all_results[uploaded_file.name]['content'] = edited_content
                    
                    # Download buttons
                    st.subheader("💾 Download Options")
                    download_cols = st.columns(len(output_formats))
                    
                    for i, format_type in enumerate(output_formats):
                        with download_cols[i]:
                            converted_content = convert_to_format(edited_content, format_type)
                            
                            st.download_button(
                                label=f"⬇️ .{format_type.value}",
                                data=converted_content,
                                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}.{format_type.value}",
                                mime=f"text/{format_type.value}" if format_type != OutputFormat.DOCX else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
            
            # Bulk download option
            if len(uploaded_files) > 1:
                st.subheader("📦 Bulk Download")
                
                # Create ZIP file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, result in all_results.items():
                        base_name = filename.rsplit('.', 1)[0]
                        
                        for format_type in output_formats:
                            converted_content = convert_to_format(result['content'], format_type)
                            zip_file.writestr(
                                f"{base_name}.{format_type.value}",
                                converted_content
                            )
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Download All Files (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="converted_documents.zip",
                    mime="application/zip"
                )
            
            st.success("🎉 All documents processed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Google Gemini 2.5 Flash Lite")

if __name__ == "__main__":
    main()
