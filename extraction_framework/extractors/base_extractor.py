"""Base class for PDF text extractors"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path


class BaseExtractor(ABC):
    """Abstract base class for PDF extractors"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def extract_text(self, pdf_path: Path) -> str:
        """Extract full text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        pass
