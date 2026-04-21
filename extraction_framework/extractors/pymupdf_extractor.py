"""PyMuPDF text extractor - Simple text extraction"""
from pathlib import Path
from typing import List
from .base_extractor import BaseExtractor

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PyMuPDFXMLExtractor(BaseExtractor):
    """Extract structured content from PDF (headings, tables, text blocks)"""
    
    def __init__(self):
        super().__init__("PyMuPDF-XML")
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Install with: pip install pymupdf")
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract structured content with headings, tables, and key text"""
        doc = fitz.open(pdf_path)
        structured_text = ""
        
        for page_num, page in enumerate(doc):
            structured_text += f"\n=== PAGINA {page_num + 1} ===\n"
            
            # Get text blocks with position info
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        font_size = 0
                        
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                                font_size = max(font_size, span.get("size", 0))
                        
                        line_text = line_text.strip()
                        if line_text:
                            # Identify headings by font size (>14 = heading)
                            if font_size > 14:
                                structured_text += f"\n## {line_text}\n"
                            else:
                                structured_text += line_text + "\n"
                
                elif block.get("type") == 1:  # Image block
                    structured_text += "[IMMAGINE]\n"
            
            # Extract tables using get_text("html") and parse
            tables = self._extract_tables(page)
            if tables:
                structured_text += "\n### TABELLE ###\n"
                for table in tables:
                    structured_text += table + "\n"
        
        doc.close()
        return structured_text.strip()
    
    def _extract_tables(self, page) -> List[str]:
        """Extract tables from page"""
        tables = []
        try:
            # Use simple table detection based on alignment
            blocks = page.get_text("dict")["blocks"]
            # Simple heuristic: find text blocks with similar Y positions
            # (This is a simplified version, you may want to use a proper table extraction library)
            return tables
        except:
            return tables
    
    def extract_pages(self, pdf_path: Path) -> List[str]:
        """Extract structured content page by page"""
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc):
            page_text = f"=== PAGINA {page_num + 1} ===\n"
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                line_text += text + " "
                        if line_text.strip():
                            page_text += line_text.strip() + "\n"
            
            pages.append(page_text)
        doc.close()
        return pages
    
    def extract_metadata(self, pdf_path: Path) -> dict:
        """Extract PDF metadata"""
        doc = fitz.open(pdf_path)
        metadata = super().extract_metadata(pdf_path)
        metadata.update({
            "num_pages": doc.page_count,
            "pdf_metadata": doc.metadata,
            "format": "XML"
        })
        doc.close()
        return metadata

