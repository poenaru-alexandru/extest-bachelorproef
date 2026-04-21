"""Extractor factory and registry"""
from typing import List, Type
from .base_extractor import BaseExtractor
from .pymupdf_extractor import PyMuPDFXMLExtractor
from .pdf_direct_extractor import PDFDirectExtractor


def get_all_extractors() -> List[BaseExtractor]:
    """Get all available PDF extractors
    
    Returns:
        List of initialized extractor instances
    """
    extractors = []
    extractor_classes = [
        PyMuPDFXMLExtractor,
        PDFDirectExtractor,
    ]
    
    for extractor_config in extractor_classes:
        try:
            if isinstance(extractor_config, tuple):
                extractor_class, kwargs = extractor_config
                extractors.append(extractor_class(**kwargs))
            else:
                extractors.append(extractor_config())
        except (ImportError, Exception) as e:
            # Skip extractors that fail to initialize due to missing optional deps.
            print(f"Warning: Extractor not available: {e}")
            continue
    
    return extractors


def get_extractor_by_name(name: str, **kwargs) -> BaseExtractor:
    """Get extractor by name
    
    Args:
        name: Name of the extractor (case insensitive)
        **kwargs: Additional parameters for the extractor
        
    Returns:
        Initialized extractor instance
        
    Raises:
        ValueError: If extractor not found
    """
    name_lower = name.lower().replace(" ", "-").replace("_", "-")
    
    extractor_map = {
        "pymupdf-xml": PyMuPDFXMLExtractor,
        "xml": PyMuPDFXMLExtractor,  # Alias
        "pdf-direct": PDFDirectExtractor,
        "direct": PDFDirectExtractor,  # Alias
    }
    
    if name_lower not in extractor_map:
        available = ", ".join(sorted(set(extractor_map.keys())))
        raise ValueError(f"Extractor '{name}' not found. Available: {available}")
    
    try:
        return extractor_map[name_lower](**kwargs)
    except ImportError as e:
        raise ImportError(f"Extractor '{name}' not available: {e}")
