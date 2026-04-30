"""LLM provider factory and registry"""
from typing import Dict, List, Optional, Any
from .base_provider import BaseLLMProvider
from .llama_cpp_provider import LlamaCppProvider
from .huggingface_provider import HuggingFaceProvider
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = {"local", "huggingface"}


def _load_providers_config() -> Dict[str, Dict[str, Any]]:
    """Load provider configuration from LLM_PROVIDERS JSON env var
    
    Returns:
        Dictionary with provider configurations
        
    Raises:
        ValueError: If LLM_PROVIDERS is not set or invalid JSON
    """
    llm_providers_json = os.getenv("LLM_PROVIDERS")
    
    if not llm_providers_json:
        raise ValueError(
            "LLM_PROVIDERS environment variable not set. "
            "Please configure providers in .env using JSON format. "
            "Example: LLM_PROVIDERS='{\"huggingface\": {\"api_key\": \"hf_...\", \"base_url\": \"https://api-inference.huggingface.co/models/\", \"models\": [\"model1\"]}, \"local\": {\"base_url\": \"http://localhost:11434/v1\", \"models\": [\"local1\"]}}'"
        )
    
    try:
        config = json.loads(llm_providers_json)
        if not isinstance(config, dict):
            raise ValueError("LLM_PROVIDERS must be a JSON object")

        normalized_config = {k.lower(): v for k, v in config.items()}
        unsupported = set(normalized_config.keys()) - SUPPORTED_PROVIDERS
        if unsupported:
            raise ValueError(
                "Unsupported provider(s) in LLM_PROVIDERS: "
                f"{', '.join(sorted(unsupported))}. "
                "Only 'local' and 'huggingface' are supported."
            )

        missing = SUPPORTED_PROVIDERS - set(normalized_config.keys())
        if missing:
            raise ValueError(
                "Missing required provider(s) in LLM_PROVIDERS: "
                f"{', '.join(sorted(missing))}. "
                "Both 'local' and 'huggingface' must be configured."
            )

        for provider_name, provider_config in normalized_config.items():
            if not isinstance(provider_config, dict):
                raise ValueError(f"Configuration for '{provider_name}' must be a JSON object")
            base_url = provider_config.get("base_url")
            if not base_url:
                raise ValueError(f"Provider '{provider_name}' must define 'base_url'")
            models = provider_config.get("models", [])

        return normalized_config
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM_PROVIDERS JSON: {e}")


_PROVIDERS_CONFIG = _load_providers_config()


def get_provider(
    provider_name: str, 
    model: Optional[str] = None, 
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> BaseLLMProvider:
    """Get LLM provider by name with configuration from .env
    
    Args:
        provider_name: Name of the provider ('local', 'huggingface')
        model: Model name/identifier (if None, uses first from config)
        api_key: API key override (if None, uses config)
        base_url: Base URL override (if None, uses config)
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If provider not found or not configured
    """
    provider_name_lower = provider_name.lower()
    
    # Get configuration for this provider
    provider_config = _PROVIDERS_CONFIG.get(provider_name_lower, {})
    
    # Use provided values or fallback to config
    api_key = api_key or provider_config.get("api_key")
    base_url = base_url or provider_config.get("base_url")
    
    # If model not specified, use first from config
    if model is None:
        models = provider_config.get("models", [])
        model = models[0] if models else None
    
    # Return LlamaCppProvider for local models
    if provider_name_lower == "local":
        # Resolve absolute path to model
        # Base dir is the project root (BP/)
        base_dir = Path(__file__).parent.parent.parent.parent
        model_path = base_dir / "local_models" / (model if model and model.endswith(".gguf") else f"{model}.gguf" if model else "")
        
        return LlamaCppProvider(
            model_path=str(model_path)
        )
    
    # Return HuggingFaceProvider for HF endpoints
    elif provider_name_lower == "huggingface":
        return HuggingFaceProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider_name=provider_name
        )
    
    else:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. "
            f"Allowed providers: local, huggingface"
        )


def get_available_providers() -> List[str]:
    """Get list of configured providers from .env
    
    Returns:
        List of provider names that have configurations
    """
    return list(_PROVIDERS_CONFIG.keys())


def get_provider_models(provider_name: str) -> List[str]:
    """Get list of models for a specific provider
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        List of model names for this provider
    """
    provider_config = _PROVIDERS_CONFIG.get(provider_name.lower(), {})
    return provider_config.get("models", [])


def get_all_providers_config() -> Dict[str, Dict[str, Any]]:
    """Get complete provider configuration
    
    Returns:
        Dictionary with all provider configurations
    """
    return _PROVIDERS_CONFIG.copy()
