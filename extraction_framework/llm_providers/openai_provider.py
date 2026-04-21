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
        """Extract token usage from response"""
        usage = response.usage if hasattr(response, 'usage') else None
        return {
            'input': usage.prompt_tokens if usage else 0,
            'output': usage.completion_tokens if usage else 0,
            'total': usage.total_tokens if usage else 0
        }

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
            system_prompt = (
                f"{schema.__doc__ or 'Extract structured data.'}\n"
                f"You MUST return ONLY a valid JSON object matching this schema:\n{schema_json}\n"
                "Do not include any preamble, explanation, or markdown code blocks like ```json."
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
            params["temperature"] = 0.1  # Very low for extraction

        
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
