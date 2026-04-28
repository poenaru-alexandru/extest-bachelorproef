"""Hugging Face provider using the official huggingface_hub package"""
import json
import time
import re
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from .base_provider import BaseLLMProvider, ExtractionError

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
        self.model = model
        self.client = InferenceClient(**kwargs)
    def extract_structured_data(
        self, 
        text: str, 
        schema: type[BaseModel],
        system_prompt: Optional[str] = None
    ) -> tuple[BaseModel, Dict[str, Any]]:
        """Extract structured data using Hugging Face InferenceClient"""
        
        # Clean up the schema to avoid confusing the model
        schema_dict = schema.model_json_schema()
        schema_dict.pop('$schema', None)
        schema_dict.pop('title', None)
        schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
        
        class_name = schema.__name__
        
        if system_prompt is None:
            system_prompt = (
                f"Sei un assistente esperto nell'estrazione di dati. {schema.__doc__ or ''}\n\n"
                "IL TUO COMPITO:\n"
                "Estrai i dati dal testo fornito e inseriscili NELLA STRUTTURA JSON descritta qui sotto.\n"
                "DEVI restituire un oggetto JSON popolato con i valori reali, NON la definizione dello schema.\n\n"
                f"STRUTTURA JSON DA POPOLARE:\n{schema_json}\n\n"
                "REGOLE MANDATORIE:\n"
                "1. Usa ESATTAMENTE i nomi dei campi definiti sopra.\n"
                "2. NON inventare nuovi nomi di campi.\n"
                "3. Restituisci SOLO il JSON popolato con i dati.\n"
                f"4. NON racchiudere il JSON in una chiave '{class_name}'.\n"
                "5. Se un dato non è presente, usa null."
            )
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Analizza il seguente testo ed estrai i dati seguendo la struttura richiesta.\n"
                f"Ricorda: restituisci solo il JSON popolato, non lo schema.\n\n"
                f"TESTO:\n{text}"
            )}
        ]
        
        # --- EXECUTION ---
        start_time = time.time()
        ttft = None
        accumulated_content = ""
        final_impacts = None
        
        try:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages, 
                stream=True, 
                temperature=0,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            for chunk in response_stream:
                if ttft is None:
                    ttft = time.time() - start_time
                
                choices = chunk.get("choices", []) if isinstance(chunk, dict) else getattr(chunk, "choices", [])
                
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else getattr(choice, "delta", None)
                    
                    if delta:
                        content = delta.get("content", "") if isinstance(delta, dict) else getattr(delta, "content", "")
                        if content:
                            accumulated_content += content
                    
                print(f"Received chunk: {chunk.impacts}")  # Debugging output
                        
        except Exception as stream_e:
            end_time = time.time()
            total_time = end_time - start_time
            gen_time = total_time - (ttft or 0)
            input_chars = sum(len(m.get("content", "")) for m in messages)
            output_chars = len(accumulated_content)
            input_tokens = int(input_chars / 4)
            output_tokens = int(output_chars / 4)
            partial_usage = {
                'ttft_seconds': ttft,
                'generation_seconds': gen_time,
                'total_inference_seconds': total_time,
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            }
            raise ExtractionError(
                message=f"Failed during HF API call: {stream_e}.",
                raw_content=accumulated_content,
                token_usage=partial_usage
            )
            
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
            
            # Fallback for models that wrap the output in the class name despite instructions
            if class_name in parsed_json and len(parsed_json) == 1:
                parsed_json = parsed_json[class_name]
                
            return schema.model_validate(parsed_json), token_usage
        except Exception as e:
            raise ExtractionError(
                message=f"Failed to parse LLM response as {schema.__name__}. Error: {e}.",
                raw_content=accumulated_content,
                token_usage=token_usage
            )
