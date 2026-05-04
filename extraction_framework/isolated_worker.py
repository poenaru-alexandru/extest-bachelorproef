"""
Isolated worker for running local LLM inference.
This script is designed to be called via subprocess, load a model,
process a single text payload, track emissions, and terminate.
This ensures perfect VRAM garbage collection by the OS.
"""
import sys
import json
import time
from pathlib import Path
import os
os.environ["GGML_CUDA_NO_GRAPH_CACHE"] = "1"
os.environ["GGML_CUDA_DISABLE_GRAPHS"] = "1"

# Add parent directory to path if needed depending on your run location
sys.path.insert(0, str(Path(__file__).parent.parent))

from codecarbon import OfflineEmissionsTracker
from extraction_framework.llm_providers import get_provider
from extraction_framework.Test.modello import FactuurModel

def run_worker(payload_path: str, model_name: str, output_path: str):
    print(f"DEBUG: My Process ID is {os.getpid()}")
    # 1. Load the pre-extracted text (No PyMuPDF CPU energy tracked!)
    with open(payload_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # 2. Boot the model (Provider handles the 8GB VRAM configuration)
    print(f"[Worker] Booting local model: {model_name}...")
    provider = get_provider("local", model_name)

    # 3. Setup pure inference tracking
    tracker = OfflineEmissionsTracker(
        project_name=f"Inference_{Path(payload_path).stem}",
        measure_power_secs=0.1, # CRITICAL: Sub-second tracking for fast local LLMs
        save_to_file=False,
        log_level="error",
        country_iso_code="ITA"
    )
    
    print(f"[Worker] Tracking started. Running inference...")
    tracker.start()

    # 4. Run the pure inference call
    try:
        extracted, token_usage = provider.extract_structured_data(
            text=document_text, 
            schema=FactuurModel
        )
        success = True
        error = None
        data = extracted.model_dump()
    except Exception as e:
        success = False
        error = str(e)
        data = None
        token_usage = getattr(e, 'token_usage', {})

    # 5. Stop tracking and extract data safely
    tracker.stop()
    d = getattr(tracker, 'final_emissions_data', None)
    
    # 6. Standardize emissions payload to EXACTLY match EcoLogits
    emissions_payload = {
        "energy_kwh": d.energy_consumed if d else None,
        "gwp_kgCO2eq": d.emissions if d else None, # Map CodeCarbon CO2 to GWP
        "adpe_kgSbeq": None, # Cannot be tracked via telemetry
        "pe_mj": None,       # Cannot be tracked via telemetry
        
        # Keep hardware-specific telemetry as a nested bonus for local runs
        "local_telemetry": {
            "cpu_energy_kwh": d.cpu_energy if d else None,
            "gpu_energy_kwh": d.gpu_energy if d else None,
            "ram_energy_kwh": d.ram_energy if d else None,
        }
    }

    # Inject into token_usage to maintain the same schema as HuggingFaceProvider
    if isinstance(token_usage, dict):
        token_usage['impacts'] = emissions_payload

    # 7. Package the unified results for the orchestrator
    result_payload = {
        "success": success,
        "error": error,
        "data": data,
        "token_usage": token_usage 
    }

    # 8. Write to the output bridge file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f)
    print(f"[Worker] Inference complete. Terminating process to flush VRAM.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python isolated_worker.py <payload_path> <model_name> <output_path>")
        sys.exit(1)
        
    run_worker(sys.argv[1], sys.argv[2], sys.argv[3])