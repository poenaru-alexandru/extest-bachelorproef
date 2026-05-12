"""LLM provider factory and registry"""
from typing import Dict, List, Optional, Any
from .base_provider import BaseLLMProvider
from .llama_cpp_provider import LlamaCppProvider
from .huggingface_provider import HuggingFaceProvider
from .local_server_manager import LocalServerManager
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_PROVIDERS = {"local", "huggingface"}


def _load_providers_config() -> Dict[str, Dict[str, Any]]:
    """Load provider configuration from LLM_PROVIDERS JSON env var."""
    llm_providers_json = os.getenv("LLM_PROVIDERS")

    if not llm_providers_json:
        raise ValueError(
            "LLM_PROVIDERS environment variable not set. "
            "Please configure providers in .env using JSON format. "
            "Example: LLM_PROVIDERS='{\"huggingface\": {\"api_key\": \"hf_...\", \"base_url\": \"https://api-inference.huggingface.co/models/\", \"models\": [\"model1\"]}, \"local\": {\"models\": [\"model.gguf\"]}}'"
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
            # base_url is required for huggingface, optional for local (managed by LocalServerManager)
            if provider_name != "local":
                base_url = provider_config.get("base_url")
                if not base_url:
                    raise ValueError(f"Provider '{provider_name}' must define 'base_url'")

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
    """Get LLM provider by name with configuration from .env.

    For 'local' providers the caller is responsible for having started
    LocalServerManager before making any inference calls.
    """
    provider_name_lower = provider_name.lower()

    provider_config = _PROVIDERS_CONFIG.get(provider_name_lower, {})

    api_key = api_key or provider_config.get("api_key")
    base_url = base_url or provider_config.get("base_url")

    if model is None:
        models = provider_config.get("models", [])
        model = models[0] if models else None

    if provider_name_lower == "local":
        # The server URL is derived from LLAMA_SERVER_PORT; the caller manages server lifecycle.
        port = int(os.getenv("LLAMA_SERVER_PORT", "8080"))
        server_url = f"http://localhost:{port}/v1"
        return LlamaCppProvider(base_url=server_url, model_name=model or "")

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


def resolve_local_model_path(model_name: str) -> str:
    """Resolve a local model name to its absolute .gguf file path.

    Models are expected at <repo_root>/local_models/<model_name>.gguf
    """
    base_dir = Path(__file__).parent.parent.parent  # → repo root
    model_file = model_name if model_name.endswith(".gguf") else f"{model_name}.gguf"
    return str(base_dir / "local_models" / model_file)


def get_available_providers() -> List[str]:
    return list(_PROVIDERS_CONFIG.keys())


def get_provider_models(provider_name: str) -> List[str]:
    provider_config = _PROVIDERS_CONFIG.get(provider_name.lower(), {})
    return provider_config.get("models", [])


def get_all_providers_config() -> Dict[str, Dict[str, Any]]:
    return _PROVIDERS_CONFIG.copy()
