"""Provider for local GGUF models using llama-cpp-python"""
import json
from pathlib import Path
from typing import Dict, Tuple, Type, Optional
from pydantic import BaseModel
from .base_provider import BaseLLMProvider, ExtractionError, build_extraction_messages

try:
    from llama_cpp import Llama
    from llama_cpp.llama_grammar import LlamaGrammar
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LlamaCppProvider(BaseLLMProvider):
    
    def __init__(
        self, 
        model_path: str,
        n_ctx: int = 32768,
    ):
        model_name = Path(model_path).name
        super().__init__("local", model_name)
        
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not installed.")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"[LlamaCpp] Loading model: {model_name} (n_ctx={n_ctx})")
        
        # BEST CAPACITY CONFIGURATION FOR RTX 4060 8GB
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            n_batch=2048,           
            n_ubatch=2048,        
            flash_attn=True,        
            # type_k=llama_cpp.GGML_TYPE_Q4_0,
            # type_v=llama_cpp.GGML_TYPE_Q4_0,
            # use_mmap=False,
            # offload_kqv=True,
            verbose=True
        ) 
        
    def extract_structured_data(
        self, 
        text: str, 
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> Tuple[BaseModel, Dict[str, int]]:
        """Extract structured data using local GGUF model and native JSON grammars with streaming telemetry"""
        import time

        # Centralized Prompt Engine (Strict Grammar = True)
        messages = build_extraction_messages(text, schema)
        
        # Pass the raw, nested schema strictly to the grammar engine
        schema_json = schema.model_json_schema()
        
        try:
            start_time = time.time()
            ttft = None
            accumulated_content = ""

            # MANDATORY 8GB SAFEGUARD: This MUST run before create_chat_completion
            self.llm.reset()

            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0,
                stream=True,
                response_format={
                    "type": "json_object",
                    "value": schema_json
                }
            )
            
            prompt_tokens = 0
            completion_tokens = 0

            for raw_chunk in response:
                if ttft is None:
                    ttft = time.time() - start_time
                
                # 2. Convert the Pydantic object back to a dict to preserve your logic
                chunk = raw_chunk.model_dump() if hasattr(raw_chunk, 'model_dump') else dict(raw_chunk)
                
                delta = chunk["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    accumulated_content += delta["content"]
                    completion_tokens += 1 
                
                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"]["prompt_tokens"]
                    completion_tokens = chunk["usage"]["completion_tokens"]
            end_time = time.time()
            total_time = end_time - start_time
            gen_time = total_time - (ttft or 0)

            if prompt_tokens == 0:
                try:
                    prompt_text = "\n".join([m.get("content", "") for m in messages])
                    prompt_tokens = len(self.llm.tokenize(prompt_text.encode("utf-8")))
                except Exception:
                    input_chars = sum(len(m.get("content", "")) for m in messages)
                    prompt_tokens = int(input_chars / 4)

            parsed_json = json.loads(accumulated_content)
            
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
                message=f"Failed to parse LLM response. Error: {e}.",
                raw_content=accumulated_content if 'accumulated_content' in locals() else "",
                token_usage=token_usage if 'token_usage' in locals() else {}
            )

    def close(self):
        """Unload model from memory"""
        if hasattr(self, 'llm'):
            del self.llm
            import gc
            gc.collect()