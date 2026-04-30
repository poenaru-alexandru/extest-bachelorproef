"""Markdown extractor using pymupdf4llm for high-quality LLM payloads"""
from pathlib import Path
from typing import List
import pymupdf4llm
from .base_extractor import BaseExtractor


class MarkdownExtractor(BaseExtractor):
    """Convert PDF to Markdown for optimal LLM context ingestion"""
    
    def __init__(self):
        super().__init__("PyMuPDF4LLM")
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract full text as Markdown string"""
        return pymupdf4llm.to_markdown(str(pdf_path))