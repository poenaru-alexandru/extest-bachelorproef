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
    from ecologits import electricity_mix_repository
    ECOLOGITS_AVAILABLE = True
except ImportError:
    ECOLOGITS_AVAILABLE = False


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
                    input_tokens = getattr(usage, "prompt_tokens", input_tokens)
                    output_tokens = getattr(usage, "completion_tokens", output_tokens)

                # 3. EcoLogits Tracking (Calculated incrementally, complete on final chunk)
                if hasattr(chunk, 'impacts') and chunk.impacts:
                    final_impacts = {
                        "energy_kwh": getattr(chunk.impacts.energy, "value", None),
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
        
        # Compile the exact metrics
        token_usage = {
            'ttft_seconds': ttft,
            'generation_seconds': gen_time,
            'total_inference_seconds': total_time,
            'input': input_tokens,          # Exact count from API
            'output': output_tokens,        # Exact count from API
            'total': input_tokens + output_tokens,
            'ecologits_impacts': final_impacts # Structured emissions data
        }

        base_energy_kwh = final_impacts["energy_kwh"]

        target_regions = ["ITA", "FRA", "USA", "WOR"] # Italy, France, USA, World Average

        regional_emissions = {}

        # 3. Calculate the hypothetical emissions for each region
        for zone in target_regions:
            # Fetch the specific grid data for the region
            mix = electricity_mix_repository.electricity_mixes.find_electricity_mix(zone=zone)
            
            if mix:
                # Multiply the constant energy by the regional GWP multiplier
                calculated_gwp = base_energy_kwh * mix.gwp
                regional_emissions[zone] = calculated_gwp

        # 4. Add this context to your final payload
        token_usage["regional_cloud_projections"] = regional_emissions
                
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