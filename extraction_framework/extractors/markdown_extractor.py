"""Markdown extractor using pymupdf4llm for high-quality LLM payloads"""
from pathlib import Path
from typing import List
import pymupdf4llm
from .base_extractor import BaseExtractor


class MarkdownExtractor(BaseExtractor):
    """Convert PDF to Markdown for optimal LLM context ingestion"""
    
    def __init__(self):
        super().__init__("Markdown-PyMuPDF")
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract full text as Markdown string"""
        return pymupdf4llm.to_markdown(str(pdf_path))
    
    def extract_pages(self, pdf_path: Path) -> List[str]:
        """Extract text page by page (Markdown per page)"""
        # pymupdf4llm doesn't have a direct "per page" markdown method as simple as to_markdown,
        # but we can use the library's page range features if needed.
        # For this PoC, we extract the whole doc and split by common markers if necessary,
        # or use fitz for page separation and pymupdf4llm for the content.
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        for i in range(len(doc)):
            # Extract single page as markdown
            page_md = pymupdf4llm.to_markdown(doc, pages=[i])
            pages.append(page_md)
        doc.close()
        return pages
