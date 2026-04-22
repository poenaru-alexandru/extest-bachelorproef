"""Generic OpenAI-compatible provider"""
from typing import Optional, List, Any
from pydantic import BaseModel
from .base_provider import BaseLLMProvider
import base64

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from ecologits import EcoLogits
    # Initialize EcoLogits for OpenAI-compatible providers with Italian electricity mix
    EcoLogits.init(providers=["openai"], electricity_mix_zone="ITA")
    ECOLOGITS_AVAILABLE = True
except ImportError:
    ECOLOGITS_AVAILABLE = False


class OpenAIProvider(BaseLLMProvider):
    """Generic OpenAI-compatible provider for configured endpoints"""
    
    def __init__(
        self, 
        model: str = "gpt-4o", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider_name: str = "OpenAI"
    ):
        super().__init__(provider_name, model, api_key)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Install with: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def _build_content(
        self,
        text: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> List[dict]:
        """Build content list for strictly text-based extraction
        
        Args:
            text: Text content (XML, plain text)
            prompt: User prompt/instruction
            
        Returns:
            List of content parts for OpenAI API
        """
        content = []
        
        # Combine prompt and text into one text block
        text_parts = []
        if prompt:
            text_parts.append(prompt)
        if text:
            text_parts.append(text)
            
        if text_parts:
            content.append({"type": "text", "text": "\n\n".join(text_parts)})
        
        return content

    def _extract_tokens(self, response) -> dict:
        """Extract token usage and sustainability data from response"""
        usage = response.usage if hasattr(response, 'usage') else None
        token_usage = {
            'input': usage.prompt_tokens if usage else 0,
            'output': usage.completion_tokens if usage else 0,
            'total': usage.total_tokens if usage else 0
        }
        
        # Extract EcoLogits impacts if available
        if ECOLOGITS_AVAILABLE and hasattr(response, 'impacts'):
            try:
                impacts = response.impacts
                # EcoLogits returns a RangeValue with mean, low, high. We use mean.
                token_usage.update({
                    'energy_kwh': impacts.energy.value.mean,  # kWh (mean estimate)
                    'co2_kg': impacts.gwp.value.mean,        # kgCO2eq (mean estimate)
                    'energy_source': 'ecologits'
                })
            except Exception as e:
                print(f"Warning: Failed to extract EcoLogits data: {e}")
                
        return token_usage

    def extract_structured_data(
        self, 
        text: Optional[str] = None,
        schema: type[BaseModel] = None,
        system_prompt: Optional[str] = None,
        image_data_list: Optional[List[dict]] = None,
        pdf_bytes: Optional[bytes] = None
    ) -> tuple[BaseModel, dict]:
        """Extract structured data using OpenAI API with fallback parsing
        
        Args:
            text: Text content (XML, plain text)
            schema: Pydantic model for structured output
            system_prompt: Optional system prompt
            image_data_list: Ignored (multimodal removed)
            pdf_bytes: Ignored
            
        Returns:
            Tuple of (parsed_data, token_usage)
        """
        import json
        import re

        if system_prompt is None:
            # Use schema docstring and add strict JSON instructions
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            
            # Extract main class name to be explicit
            class_name = schema.__name__
            
            system_prompt = (
                f"Sei un agente specializzato nell'estrazione di dati. {schema.__doc__ or 'Estrai dati strutturati.'}\n\n"
                f"Il tuo compito è estrarre le informazioni dal documento fornito e restituirle come un oggetto JSON valido che segua lo schema riportato di seguito.\n"
                f"CRITICO: NON restituire la definizione dello schema stesso. Estrai i valori REALI dal testo e popola i campi.\n\n"
                f"Schema di output ({class_name}):\n{schema_json}\n\n"
                "Vincoli:\n"
                "- Restituisci SOLO l'oggetto JSON popolato.\n"
                "- Nessun preambolo, nessuna spiegazione, nessun blocco di codice markdown (es. ```json).\n"
                "- Se un campo non viene trovato nel testo, usa null.\n"
                "- Se lo schema definisce una lista, estrai TUTTI gli elementi trovati nel documento."
            )
        
        # Build content
        content = self._build_content(
            text=text,
            prompt=system_prompt
        )

        # Set model-specific parameters based on user instructions
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0
        }

        # Identify model type
        model_lower = self.model.lower()
        
        if "nemotron-nano" in model_lower:
            params.update({
                "frequency_penalty": 0,
                "presence_penalty": 0
            })
        
        # Only use json_object for models that support it reliably
        # Qwen models on HF Router often don't support response_format="json_object"
        if ("llama-3.1" in model_lower or "gpt-4o" in model_lower) and "qwen" not in model_lower:
            params["response_format"] = {"type": "json_object"}
        
        # Qwen specific overrides
        if "qwen" in model_lower:
            params["temperature"] = 0  # Very low for extraction

        
        # Use regular chat completion
        response = self.client.chat.completions.create(**params)

        
        raw_content = response.choices[0].message.content
        
        # Attempt to clean markdown if the model ignored instructions
        clean_json = raw_content.strip()
        if clean_json.startswith("```"):
            clean_json = re.sub(r'^```json\s*', '', clean_json)
            clean_json = re.sub(r'^```\s*', '', clean_json)
            clean_json = re.sub(r'\s*```$', '', clean_json)
        
        try:
            parsed_json = json.loads(clean_json)
            return schema.model_validate(parsed_json), self._extract_tokens(response)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response as {schema.__name__}. Error: {e}. Content: {raw_content[:200]}...")


    
    def extract_text(self, text: str, prompt: str) -> tuple[str, dict]:
        """Extract/filter text using OpenAI (for preselection)
        
        Args:
            text: Input text to process
            prompt: Instruction prompt
            
        Returns:
            Tuple of (filtered_text, token_usage)
        """
        content = self._build_content(text=text, prompt=prompt)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}]
        )
        
        return response.choices[0].message.content, self._extract_tokens(response)
    
    def supports_inline_files(self) -> bool:
        """OpenAI supports inline files (images, PDFs) via base64"""
        return True
