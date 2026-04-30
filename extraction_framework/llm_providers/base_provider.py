"""Base class for LLM providers"""
import json
import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel


class ExtractionError(Exception):
    """Exception raised when LLM extraction parsing fails, carrying partial telemetry and raw output."""
    def __init__(self, message: str, raw_content: str = None, token_usage: Dict[str, Any] = None):
        super().__init__(message)
        self.raw_content = raw_content
        self.token_usage = token_usage or {}


def build_extraction_messages(text: str, schema: Type[BaseModel]) -> list[dict]:
    """
    Centralized prompt builder. 
    Returns the exact message array needed based on the model's capabilities.
    """
    instructions = schema.__doc__ or 'Extract structured data from the text.'
    
    # Identify the root field name (usually 'periodes' in this project)
    # This helps models that might be confused by the nesting
    root_keys = list(schema.model_json_schema().get("properties", {}).keys())
    root_key_hint = f"\nEnsure the response is a JSON object with a '{root_keys[0]}' key containing the list of extracted items." if root_keys else ""

    system_content = f"{instructions}{root_key_hint}"
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Document text to analyze:\n{text}"}
    ]


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
        pass
    
    def __str__(self):
        return f"{self.name} ({self.model})"