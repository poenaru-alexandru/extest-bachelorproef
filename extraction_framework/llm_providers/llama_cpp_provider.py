"""Provider for local GGUF models via native llama-server HTTP API."""
import json
import time
from typing import Dict, Optional, Tuple, Type

from pydantic import BaseModel

from .base_provider import BaseLLMProvider, ExtractionError, build_extraction_messages

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


class LlamaCppProvider(BaseLLMProvider):
    """Talks to a running llama-server instance via its OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str):
        super().__init__("local", model_name)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        # api_key is required by the openai client but ignored by llama-server
        self._client = OpenAI(base_url=base_url, api_key="not-needed")

    def extract_structured_data(
        self,
        text: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> Tuple[BaseModel, Dict]:
        """Run a single inference call and return the validated model + telemetry dict."""
        messages = build_extraction_messages(text, schema)
        schema_json = schema.model_json_schema()

        tracker = None
        if CODECARBON_AVAILABLE:
            tracker = OfflineEmissionsTracker(
                project_name="local_inference",
                measure_power_secs=0.1,
                save_to_file=False,
                log_level="error",
                country_iso_code="ITA",
            )
            tracker.start()

        start_time = time.time()
        ttft: Optional[float] = None
        accumulated_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        end_time = start_time  # fallback if stream fails before assigning

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema_json,
                        "strict": True,
                    },
                },
            )

            for chunk in stream:
                # Record time-to-first-token on first chunk that carries content
                if ttft is None and chunk.choices:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        ttft = time.time() - start_time

                if chunk.choices:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        accumulated_content += delta_content

                # Usage arrives in the final chunk (stream_options.include_usage)
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

            end_time = time.time()

        finally:
            if tracker:
                tracker.stop()

        total_time = end_time - start_time
        gen_time = total_time - (ttft or 0)

        impacts = self._read_codecarbon(tracker)

        try:
            parsed_json = json.loads(accumulated_content)
            # llama-server sometimes wraps the result under the schema class name
            class_name = schema.__name__
            if class_name in parsed_json and len(parsed_json) == 1:
                parsed_json = parsed_json[class_name]
            result = schema.model_validate(parsed_json)
        except Exception as e:
            token_usage = self._build_token_usage(
                ttft, gen_time, total_time, prompt_tokens, completion_tokens, impacts
            )
            raise ExtractionError(
                message=f"Failed to parse LLM response: {e}",
                raw_content=accumulated_content,
                token_usage=token_usage,
            )

        token_usage = self._build_token_usage(
            ttft, gen_time, total_time, prompt_tokens, completion_tokens, impacts
        )
        return result, token_usage

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_codecarbon(self, tracker) -> Dict:
        if tracker is None:
            return {}
        d = getattr(tracker, "final_emissions_data", None)
        return {
            "energy_kwh":     d.energy_consumed if d else None,
            "co2_kg":         d.emissions       if d else None,
            "cpu_energy_kwh": d.cpu_energy      if d else None,
            "gpu_energy_kwh": d.gpu_energy      if d else None,
            "ram_energy_kwh": d.ram_energy      if d else None,
        }

    def _build_token_usage(
        self, ttft, gen_time, total_time, prompt_tokens, completion_tokens, impacts
    ) -> Dict:
        return {
            "ttft_seconds":            ttft,
            "generation_seconds":      gen_time,
            "total_inference_seconds": total_time,
            "input":                   prompt_tokens,
            "output":                  completion_tokens,
            "total":                   prompt_tokens + completion_tokens,
            "energy_kwh":              impacts.get("energy_kwh"),
            "co2_kg":                  impacts.get("co2_kg"),
            "cpu_energy_kwh":          impacts.get("cpu_energy_kwh"),
            "gpu_energy_kwh":          impacts.get("gpu_energy_kwh"),
            "ram_energy_kwh":          impacts.get("ram_energy_kwh"),
            "energy_source":           "codecarbon" if impacts else None,
        }
