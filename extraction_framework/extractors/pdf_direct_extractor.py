"""Direct PDF extractor - sends PDF directly to LLM without conversion"""
from pathlib import Path
from typing import List
from .base_extractor import BaseExtractor


class PDFDirectExtractor(BaseExtractor):
    """Send PDF directly to LLM without text/image extraction"""
    
    def __init__(self):
        super().__init__("PDF-Direct")
    
    def extract_text(self, pdf_path: Path) -> str:
        """Return marker indicating PDF should be sent directly"""
        return f"PDF_DIRECT_MODE: {pdf_path}"
    
    def extract_pages(self, pdf_path: Path) -> List[str]:
        """Return marker for direct PDF mode (one item per page would be wasteful)"""
        return [f"PDF_DIRECT_MODE: {pdf_path}"]
    
    def get_pdf_bytes(self, pdf_path: Path) -> bytes:
        """Get PDF file bytes for direct upload"""
        with open(pdf_path, 'rb') as f:
            return f.read()
    
    def get_metadata(self):
        """Return extractor metadata"""
        return {
            "name": self.name,
            "format": "Direct PDF",
            "library": "native",
            "features": ["direct_upload", "preserves_formatting", "minimal_preprocessing"]
        }
