"""Provider for local GGUF models using llama-cpp-python"""
import json
from pathlib import Path
from typing import Dict, Tuple, Type, Optional
from pydantic import BaseModel
from .base_provider import BaseLLMProvider, ExtractionError

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LlamaCppProvider(BaseLLMProvider):
    
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

        # Clean up the schema to avoid confusing the model
        schema_dict = schema.model_json_schema()
        schema_dict.pop('$schema', None)
        schema_dict.pop('title', None)
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
        
        if system_prompt is None:
            class_name = schema.__name__
            system_prompt = (
                f"Sei un assistente esperto nell'estrazione di dati strutturati. {schema.__doc__ or ''}\n\n"
                "IL TUO COMPITO:\n"
                "Estrai i dati dal testo fornito e popolali NELLA STRUTTURA JSON seguente.\n"
                "DEVI restituire un oggetto JSON popolato con i valori reali, NON la definizione dello schema.\n\n"
                f"STRUTTURA JSON DA POPOLARE:\n{schema_json}\n\n"
                "REGOLE MANDATORIE:\n"
                "1. Usa ESATTAMENTE i nomi dei campi definiti sopra.\n"
                "2. NON inventare nuovi nomi di campi.\n"
                "3. Restituisci SOLO il JSON popolato con i dati.\n"
                f"4. NON racchiudere il JSON in una chiave '{class_name}'.\n"
                "5. Se un dato non è presente, usa null."
            )

        if "gemma" in self.model.lower():
            messages = [
                {"role": "user", "content": (
                    f"{system_prompt}\n\n"
                    f"Analizza il seguente testo ed estrai i dati:\n\n"
                    f"TESTO:\n{text}"
                )}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Testo del documento da analizzare:\n{text}"}
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

            if prompt_tokens == 0:
                try:
                    # Try to use the model's native tokenizer for accuracy
                    prompt_text = "\n".join([m.get("content", "") for m in messages])
                    prompt_tokens = len(self.llm.tokenize(prompt_text.encode("utf-8")))
                except Exception:
                    # Fallback heuristic
                    input_chars = sum(len(m.get("content", "")) for m in messages)
                    prompt_tokens = int(input_chars / 4)

            parsed_json = json.loads(accumulated_content)
            
            # Fallback for models that wrap the output in the class name despite instructions
            class_name = schema.__name__
            if class_name in parsed_json and len(parsed_json) == 1:
                parsed_json = parsed_json[class_name]

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
            raise ExtractionError(
                message=f"Failed to parse LLM response as {schema.__name__}. Error: {e}.",
                raw_content=accumulated_content,
                token_usage=token_usage if 'token_usage' in locals() else {}
            )

        
    def close(self):
        """Unload model from memory"""
        if hasattr(self, 'llm'):
            del self.llm
            import gc
            gc.collect()
