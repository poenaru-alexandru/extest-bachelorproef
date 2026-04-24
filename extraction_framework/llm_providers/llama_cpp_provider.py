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
        n_ctx: int = 16384,  # Increased default for larger document payloads
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
            flash_attn=True,  # Enable flash attention for better performance if supported
            verbose=verbose,
            n_threads=8 # Standard for local runs
        )
    
    def extract_structured_data(
        self, 
        text: str, 
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> Tuple[BaseModel, Dict[str, int]]:
        """Extract structured data using local GGUF model and native JSON grammars with streaming telemetry"""
        import time

        if system_prompt is None:
            system_prompt = (
                f"Sei un agente specializzato nell'estrazione di dati. {schema.__doc__ or 'Estrai dati strutturati.'}\n"
                f"Il tuo compito è estrarre le informazioni dal documento fornito e restituirle popolando i campi."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        try:
            # Start timing
            start_time = time.time()
            ttft = None
            accumulated_content = ""

            # Use stream=True for telemetry
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_object",
                    "schema": schema.model_json_schema()
                },
                stream=True
            )
            
            prompt_tokens = 0
            completion_tokens = 0

            for chunk in response:
                if ttft is None:
                    ttft = time.time() - start_time
                
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    accumulated_content += delta["content"]
                    completion_tokens += 1 # Rough estimate if usage not in chunk
                
                # Try to extract actual usage if provided in chunk
                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"]["prompt_tokens"]
                    completion_tokens = chunk["usage"]["completion_tokens"]

            end_time = time.time()
            total_time = end_time - start_time
            gen_time = total_time - (ttft or 0)

            parsed_json = json.loads(accumulated_content)
            
            token_usage = {
                'ttft_seconds': ttft,
                'generation_seconds': gen_time,
                'total_inference_seconds': total_time,
                'input': prompt_tokens,
                'output': completion_tokens,
                'total': prompt_tokens + completion_tokens
            }
            
            return schema.model_validate(parsed_json), token_usage
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as {schema.__name__}. Error: {e}. Raw content: {accumulated_content[:500]}...")

        
    def close(self):
        """Unload model from memory"""
        if hasattr(self, 'llm'):
            del self.llm
            import gc
            gc.collect()