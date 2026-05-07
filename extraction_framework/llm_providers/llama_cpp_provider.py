"""Provider for local GGUF models via native llama-server HTTP API."""
import json
import logging
import time
from typing import Dict, Optional, Tuple, Type

import requests
from pydantic import BaseModel

from .base_provider import BaseLLMProvider, ExtractionError, build_extraction_messages

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

logger = logging.getLogger(__name__)
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

# Power Usage Effectiveness — multiply measured IT energy by this to get total facility energy.
# 1.0 = bare metal / home setup (no datacenter overhead). Fill in your calculated value.
PUE_LOCAL = 1.08


class LlamaCppProvider(BaseLLMProvider):
    """Talks to a running llama-server instance via its native HTTP API."""

    def __init__(self, base_url: str, model_name: str):
        super().__init__("local", model_name)
        self._base_url = base_url  # e.g. http://localhost:8080/v1

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
            tracker = EmissionsTracker(
                project_name="local_inference",
                measure_power_secs=0.5,
                save_to_file=False,
                pue=PUE_LOCAL,
		log_level="warning"
            )
            tracker.start()

        start_time = time.time()
        end_time = start_time

        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0,
                    "stream": False,
		    "max_tokens": 1024,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": schema_json,
                            "strict": True,
                        },
                    },
                },
            )
            response.raise_for_status()
            end_time = time.time()
            data = response.json()
        finally:
            if tracker:
                tracker.stop()

        total_time = end_time - start_time

        # llama-server returns server-side timings — more accurate than client-side wall clock
        timings = data.get("timings", {})
        ttft = timings["prompt_ms"] / 1000 if timings.get("prompt_ms") else None
        gen_time = timings["predicted_ms"] / 1000 if timings.get("predicted_ms") else total_time - (ttft or 0)

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        content = data["choices"][0]["message"]["content"]
        impacts = self._read_codecarbon(tracker)

        try:
            parsed_json = json.loads(content)
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
                raw_content=content,
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
            "cpu_energy_kwh": d.cpu_energy      if d else None,
            "gpu_energy_kwh": d.gpu_energy      if d else None,
            "ram_energy_kwh": d.ram_energy      if d else None,
        }

    def _build_token_usage(
        self, ttft, gen_time, total_time, prompt_tokens, completion_tokens, impacts
    ) -> Dict:
        energy_kwh = impacts.get("energy_kwh")
        co2_kg = None
        regional = None
        if energy_kwh is not None:
            adjusted = energy_kwh * PUE_LOCAL
            co2_kg = adjusted * EMISSION_FACTOR_ITA
            regional = {
                "ITA": adjusted * EMISSION_FACTOR_ITA,
                "BEL": adjusted * EMISSION_FACTOR_BEL,
                "FRA": adjusted * EMISSION_FACTOR_FRA,
                "DEU": adjusted * EMISSION_FACTOR_DEU,
                "USA": adjusted * EMISSION_FACTOR_USA,
                "WOR": adjusted * EMISSION_FACTOR_WOR,
            }
        return {
            "ttft_seconds":               ttft,
            "generation_seconds":         gen_time,
            "total_inference_seconds":    total_time,
            "input":                      prompt_tokens,
            "output":                     completion_tokens,
            "total":                      prompt_tokens + completion_tokens,
            "energy_kwh":                 energy_kwh,
            "co2_kg":                     co2_kg,
            "cpu_energy_kwh":             impacts.get("cpu_energy_kwh"),
            "gpu_energy_kwh":             impacts.get("gpu_energy_kwh"),
            "ram_energy_kwh":             impacts.get("ram_energy_kwh"),
            "energy_source":              "codecarbon" if impacts else None,
            "regional_cloud_projections": regional,
        }
