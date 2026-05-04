"""Hugging Face provider using the official huggingface_hub package"""
import json
import time
import re
from typing import Optional, List, Any, Dict
from pydantic import BaseModel
from .base_provider import BaseLLMProvider, ExtractionError, build_extraction_messages

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

# ---------------------------------------------------------------------------
# Regional emission factors — kgCO2eq per kWh
# Update these from your source (e.g. nowtricity.com) and note the snapshot date.
# ---------------------------------------------------------------------------
EMISSION_FACTOR_ITA = 0.300   # Italy
EMISSION_FACTOR_BEL = 0.200   # Belgium
EMISSION_FACTOR_FRA = 0.100   # France
EMISSION_FACTOR_DEU = 0.400   # Germany
EMISSION_FACTOR_USA = 0.500   # USA
EMISSION_FACTOR_WOR = 0.150   # World average


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face provider using official huggingface_hub for accurate EcoLogits tracking"""
    
    def __init__(
        self, 
        model: str, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None, 
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
        
        # Centralized Prompt Engine
        messages = build_extraction_messages(text, schema)
        class_name = schema.__name__
        
# --- EXECUTION ---
        start_time = time.time()
        ttft = None
        accumulated_content = ""
        schema_json = schema.model_json_schema()
        
        # Trackers
        input_tokens = 0
        output_tokens = 0
        final_impacts = None

        try:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages, 
                stream=True, 
                temperature=0,
                stream_options={"include_usage": True}, # Forces exact usage tracking
                response_format={
                    "type": "json_object",
                    "value": schema_json
                }
            )
            
            for chunk in response_stream:
                if ttft is None:
                    ttft = time.time() - start_time
                
                # 1. Content Extraction
                choices = chunk.get("choices", []) if isinstance(chunk, dict) else getattr(chunk, "choices", [])
                if choices:
                    choice = choices[0]
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else getattr(choice, "delta", None)
                    if delta:
                        content = delta.get("content", "") if isinstance(delta, dict) else getattr(delta, "content", "")
                        if content:
                            accumulated_content += content
                
                # 2. Exact Token Usage Tracking (Yielded in the final chunk)
                usage = chunk.get("usage", None) if isinstance(chunk, dict) else getattr(chunk, "usage", None)
                if usage:
                    if isinstance(usage, dict):
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                    else:
                        input_tokens = getattr(usage, "prompt_tokens", input_tokens)
                        output_tokens = getattr(usage, "completion_tokens", output_tokens)

                # 3. EcoLogits Tracking (Calculated incrementally, complete on final chunk)
                # Uses usage-phase only (electricity during inference) — excludes embodied carbon
                # (hardware manufacturing). impacts.usage.energy has no embodied component;
                # impacts.energy is the total and would inflate CO2 numbers.
                if hasattr(chunk, 'impacts') and chunk.impacts:
                    usage = getattr(chunk.impacts, "usage", None)
                    energy_val = getattr(usage, "energy", None) if usage else None
                    gwp_val = getattr(usage, "gwp", None) if usage else None
                    final_impacts = {
                        "energy_kwh": energy_val.mean if hasattr(energy_val, "mean") else energy_val,
                        "co2_kg": gwp_val.mean if hasattr(gwp_val, "mean") else gwp_val,
                    }
                                            
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
        
        base_energy_kwh = final_impacts["energy_kwh"] if final_impacts else None

        regional_emissions = {}
        if base_energy_kwh is not None:
            for zone, factor in [
                ("ITA", EMISSION_FACTOR_ITA),
                ("BEL", EMISSION_FACTOR_BEL),
                ("FRA", EMISSION_FACTOR_FRA),
                ("DEU", EMISSION_FACTOR_DEU),
                ("USA", EMISSION_FACTOR_USA),
                ("WOR", EMISSION_FACTOR_WOR),
            ]:
                regional_emissions[zone] = base_energy_kwh * factor

        # Compile the exact metrics
        token_usage = {
            'ttft_seconds': ttft,
            'generation_seconds': gen_time,
            'total_inference_seconds': total_time,
            'input': input_tokens,
            'output': output_tokens,
            'total': input_tokens + output_tokens,
            'energy_kwh': base_energy_kwh,
            'co2_kg': final_impacts.get("co2_kg") if final_impacts else None,
            'energy_source': 'ecologits' if final_impacts else None,
            'ecologits_impacts': final_impacts,
            'regional_cloud_projections': regional_emissions if regional_emissions else None,
        }
                
        # Clean and parse the accumulated JSON
        clean_json = accumulated_content.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r'^```json\s*', '', clean_json)
            clean_json = re.sub(r'^```\s*', '', clean_json)
            clean_json = re.sub(r'\s*```$', '', clean_json)
            
        try:
            parsed_json = json.loads(clean_json)
            
            if isinstance(parsed_json, list) and len(parsed_json) > 0:
                parsed_json = parsed_json[0]
                
            if class_name in parsed_json and len(parsed_json) == 1:
                parsed_json = parsed_json[class_name]
                
            # Return both the validated schema and the comprehensive usage data
            return schema.model_validate(parsed_json), token_usage
            
        except Exception as e:
            raise ExtractionError(
                message=f"Failed to parse LLM response as {schema.__name__}. Error: {e}.",
                raw_content=accumulated_content,
                token_usage=token_usage # Pass the exact usage even on parse failure
            )