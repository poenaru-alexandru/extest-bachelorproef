"""Base class for LLM providers"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, name: str, model: str, api_key: Optional[str] = None):
        self.name = name
        self.model = model
        self.api_key = api_key
    
    @abstractmethod
    def extract_structured_data(
        self, 
        text: str, 
        schema: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> tuple[BaseModel, Dict[str, int]]:
        """Extract structured data using the LLM
        
        Args:
            text: Identical payload string for all models (standardized Markdown)
            schema: Pydantic model defining output structure
            system_prompt: Optional system prompt override
            
        Returns:
            Tuple of (extracted data instance, token usage dict)
        """
        pass
    
    def __str__(self):
        return f"{self.name} ({self.model})"
