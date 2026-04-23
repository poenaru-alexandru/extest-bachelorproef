"""Provider for local GGUF models using llama-cpp-python"""
import json
from pathlib import Path
from typing import Dict, Tuple, Type, Optional
from pydantic import BaseModel
from .base_provider import BaseLLMProvider

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LlamaCppProvider(BaseLLMProvider):
    """Local GGUF model provider using llama-cpp-python
    
    Recommended Text-Only Models:
    - Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
    - nemotron-nano-9b-v2-q4_k_m.gguf
    (Note: Vision models like Qwen-VL require a specialized Llava handler)
    """
    
    def __init__(
        self, 
        model_path: str,
        n_ctx: int = 8192,  # Safer default to prevent Out-Of-Memory errors
        n_gpu_layers: int = -1,  # -1 for all layers on GPU
        verbose: bool = False
    ):
        # We use the filename as the model name
        model_name = Path(model_path).name
        super().__init__("local", model_name)
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"[LlamaCpp] Loading model: {model_name} (n_ctx={n_ctx})")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            n_threads=8 # Standard for local runs
        )
    
    def extract_structured_data(
        self, 
        text: str, 
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> Tuple[BaseModel, Dict[str, int]]:
        """Extract structured data using local GGUF model and native JSON grammars"""
        
        if system_prompt is None:
            # We use the Italian system prompt to match the OpenAI provider for consistency
            # Notice we removed the heavy "ONLY OUTPUT JSON" begging, as the grammar handles it natively.
            system_prompt = (
                f"Sei un agente specializzato nell'estrazione di dati. {schema.__doc__ or 'Estrai dati strutturati.'}\n"
                f"Il tuo compito è estrarre le informazioni dal documento fornito e restituirle popolando i campi."
            )

        # Build the message array. llama-cpp-python will automatically apply the correct 
        # template (ChatML, Llama 3, etc.) based on the GGUF file's internal metadata.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            # Let llama-cpp-python handle the prompt templating and JSON enforcement natively
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_object",
                    "schema": schema.model_json_schema()
                }
            )
            
            # The output is guaranteed by the grammar to be a JSON string matching the schema.
            # No regex cleanup is required.
            raw_content = response["choices"][0]["message"]["content"]
            parsed_json = json.loads(raw_content)
            
            token_usage = {
                'input': response["usage"]["prompt_tokens"],
                'output': response["usage"]["completion_tokens"],
                'total': response["usage"]["total_tokens"]
            }
            
            return schema.model_validate(parsed_json), token_usage
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as {schema.__name__}. Error: {e}")

    def supports_inline_files(self) -> bool:
        return False
        
    def close(self):
        """Unload model from memory"""
        if hasattr(self, 'llm'):
            del self.llm
            import gc
            gc.collect()