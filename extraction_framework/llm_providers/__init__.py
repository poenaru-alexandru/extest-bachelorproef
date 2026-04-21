"""LLM provider factory and registry"""
from typing import Dict, List, Optional, Any
from .base_provider import BaseLLMProvider
from .openai_provider import OpenAIProvider
import os
import json
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = {"nvidia", "local", "huggingface"}


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
            "Example: LLM_PROVIDERS='{\"nvidia\": {\"api_key\": \"nvapi-...\", \"base_url\": \"https://integrate.api.nvidia.com/v1\", \"models\": [\"model1\", \"model2\", \"model3\"]}, \"local\": {\"base_url\": \"http://localhost:11434/v1\", \"models\": [\"local1\", \"local2\", \"local3\"]}}'"
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
                "Only 'nvidia' and 'local' are supported."
            )

        missing = SUPPORTED_PROVIDERS - set(normalized_config.keys())
        if missing:
            raise ValueError(
                "Missing required provider(s) in LLM_PROVIDERS: "
                f"{', '.join(sorted(missing))}. "
                "Both 'nvidia' and 'local' must be configured."
            )

        for provider_name, provider_config in normalized_config.items():
            if not isinstance(provider_config, dict):
                raise ValueError(f"Configuration for '{provider_name}' must be a JSON object")
            base_url = provider_config.get("base_url")
            if not base_url:
                raise ValueError(f"Provider '{provider_name}' must define 'base_url'")
            models = provider_config.get("models", [])
            # if not isinstance(models, list) or len(models) != 3:
            #     raise ValueError(
            #         f"Provider '{provider_name}' must define exactly 3 models in 'models'"
            #     )

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
        provider_name: Name of the provider ('nvidia' or 'local')
        model: Model name/identifier (if None, uses first from config)
        api_key: API key override (if None, uses config)
        base_url: Base URL override (if None, uses config)
        
    Returns:
        Initialized provider instance
        
    Raises:
        ValueError: If provider not found or not configured
    """
    provider_name_lower = provider_name.lower()
    if provider_name_lower not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. "
            f"Allowed providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    
    # Get configuration for this provider
    provider_config = _PROVIDERS_CONFIG.get(provider_name_lower, {})
    
    if not provider_config:
        raise ValueError(
            f"Provider '{provider_name}' not configured in .env. "
            f"Available providers: {', '.join(get_available_providers())}"
        )
    
    # Use provided values or fallback to config
    api_key = api_key or provider_config.get("api_key")
    base_url = base_url or provider_config.get("base_url")
    
    # If model not specified, use first from config
    if model is None:
        models = provider_config.get("models", [])
        model = models[0] if models else None
    
    # Both NVIDIA and local endpoints are OpenAI-compatible.
    return OpenAIProvider(
        model=model or provider_config["models"][0],
        api_key=api_key,
        base_url=base_url,
        provider_name=provider_name
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
