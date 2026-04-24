"""Extractor factory and registry"""
from typing import List
from .base_extractor import BaseExtractor
from .markdown_extractor import MarkdownExtractor

def get_all_extractors() -> List[BaseExtractor]:
    """Get all available PDF extractors
    
    Returns:
        List of initialized extractor instances
    """
    return [MarkdownExtractor()]
