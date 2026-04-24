"""Hugging Face provider using the official huggingface_hub package"""
import json
import time
import re
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from .base_provider import BaseLLMProvider

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from ecologits import EcoLogits
    ECOLOGITS_AVAILABLE = True
except ImportError:
    ECOLOGITS_AVAILABLE = False


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face provider using official huggingface_hub for accurate EcoLogits tracking"""
    
    def __init__(
        self, 
        model: str, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # Kept in signature for factory compatibility but ignored
        provider_name: str = "huggingface"
    ):
        super().__init__(provider_name, model, api_key)
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("huggingface_hub not installed. Install with: pip install huggingface_hub")
        
        kwargs = {"api_key": api_key}
            
        # Parse HuggingFace model string which might be "model_id" or "model_id:provider"
        # For example "google/gemma-2-9b:fastest" or "meta-llama/Llama-3.1-8B-Instruct:novita"
        # However, huggingface_hub also supports setting the provider in InferenceClient explicitly.
        
        # Keep the exact model string for the API calls as users specified it in .env
        self.model = model
        
        # For Gemma, we force provider="auto" in the InferenceClient
        if "gemma" in self.model.lower():
            kwargs["provider"] = "auto"
            
        self.client = InferenceClient(**kwargs)
            
    def extract_structured_data(
        self, 
        text: str, 
        schema: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> tuple[BaseModel, Dict[str, Any]]:
        """Extract structured data using Hugging Face InferenceClient"""
        
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        class_name = schema.__name__
        
        # ONE STANDARD PROMPT - No fallbacks
        standard_prompt = (
            f"Sei un agente specializzato nell'estrazione di dati. {schema.__doc__ or 'Estrai dati strutturati.'}\n\n"
            f"Il tuo compito è estrarre le informazioni dal documento fornito e restituirle come un oggetto JSON valido.\n"
            f"Schema di output ({class_name}):\n{schema_json}\n\n"
            "Vincoli:\n"
            "- Restituisci SOLO l'oggetto JSON popolato.\n"
            "- Nessun preambolo, nessuna spiegazione, nessun blocco di codice markdown.\n"
            "- Se un campo non viene trovato, usa null."
        )
            
        messages = [
            {"role": "system", "content": standard_prompt},
            {"role": "user", "content": text}
        ]
        
        # --- EXECUTION ---
        start_time = time.time()
        ttft = None
        accumulated_content = ""
        final_impacts = None
        
        try:
            if "gemma" in self.model.lower():
                # Gemma requires text_generation instead of chat.completions
                combined_prompt = f"{standard_prompt}\n\nTesto da analizzare:\n{text}\n\nJSON Output:\n"
                response = self.client.text_generation(
                    combined_prompt,
                    model=self.model,
                    temperature=0
                )
                accumulated_content = response
                ttft = time.time() - start_time
            else:
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, 
                    stream=True, 
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                for chunk in response_stream:
                    if ttft is None:
                        ttft = time.time() - start_time
                    
                    if chunk.choices and chunk.choices[0].delta.content:
                        accumulated_content += chunk.choices[0].delta.content
                        
                    if hasattr(chunk, 'impacts') and chunk.impacts:
                        final_impacts = chunk.impacts
                        
        except Exception as stream_e:
            raise ValueError(f"Failed during HF API call: {stream_e}. Accumulated so far: {accumulated_content[:200]}")
            
        end_time = time.time()
        total_time = end_time - start_time
        gen_time = total_time - (ttft or 0)
        
        # Fallback Token Math (Heuristic)
        input_chars = sum(len(m.get("content", "")) for m in messages)
        output_chars = len(accumulated_content)
        input_tokens = int(input_chars / 4)
        output_tokens = int(output_chars / 4)
        
        token_usage = {
            'ttft_seconds': ttft,
            'generation_seconds': gen_time,
            'total_inference_seconds': total_time,
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens
        }
        
        # Integrate sustainability data if usage was official
        if final_impacts and ECOLOGITS_AVAILABLE:
            try:
                token_usage.update({
                    'energy_kwh': final_impacts.energy.value.mean,
                    'co2_kg': final_impacts.gwp.value.mean,
                    'energy_source': 'ecologits'
                })
            except Exception:
                pass
                
        # Attempt to clean markdown if the model ignored instructions
        clean_json = accumulated_content.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r'^```json\s*', '', clean_json)
            clean_json = re.sub(r'^```\s*', '', clean_json)
            clean_json = re.sub(r'\s*```$', '', clean_json)
            
        try:
            parsed_json = json.loads(clean_json)
            return schema.model_validate(parsed_json), token_usage
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as {schema.__name__}. Error: {e}. Raw content: {accumulated_content[:500]}...")
