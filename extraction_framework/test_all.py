"""
Batch test script to run extractions on multiple models.
"""
import sys
from pathlib import Path
import os
import json
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize global colorized print override
import extraction_framework.console

from extraction_framework.test_runner import TestRunner
from extraction_framework.llm_providers import get_all_providers_config
from extraction_framework.extractors import get_all_extractors
from extraction_framework.Test.modello import BachelorProefModel

# --- CONFIGURATION ---
# Set to True/False to enable/disable provider categories
PROVIDERS_TO_TEST = {
    "cloud": True,  # For huggingface, etc.
    "local": True   # For llama-cpp
}
# ---------------------

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    BASE_DIR = Path(__file__).parent.parent
    TEST_DIR = BASE_DIR / "Test"
    RESULTS_DIR = BASE_DIR / "extraction_framework" / "results"
    
    print(f"Project Root: {BASE_DIR}")
    print(f"Test Directory: {TEST_DIR}")
    
    runner = TestRunner(results_dir=RESULTS_DIR)
    
    # 1. Gather all PDF files from the flat Test folder
    pdf_files = list(TEST_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in '{TEST_DIR}'")
        
    if not pdf_files:
        print("No PDF files found to test. Exiting.")
        return

    # 2. Gather all LLM configurations
    llm_configs = []
    providers_config = get_all_providers_config()
    
    for provider_name, config_data in providers_config.items():
        category = "local" if provider_name.lower() == "local" else "cloud"
        if not PROVIDERS_TO_TEST.get(category, False):
            print(f"Skipping provider category: {category} ({provider_name})")
            continue
            
        models = config_data.get("models", [])
        api_key = config_data.get("api_key")
        base_url = config_data.get("base_url")
        
        for model_name in models:
            llm_configs.append({
                "provider": provider_name,
                "model": model_name,
                "api_key": api_key,
                "base_url": base_url
            })
            
    if not llm_configs:
        print("No LLM configurations found. Please check your .env file. Exiting.")
        return
        
    print(f"Total models to test: {len(llm_configs)}")
    
    # 3. Get extractors
    extractors = [e.name for e in get_all_extractors()]
    print(f"Using extractors: {extractors}")

    # 4. Run the test suite
    print("\nStarting batch extraction...")
    all_results = runner.run_test_suite(
        pdf_files=pdf_files,
        extractors=extractors,
        llm_configs=llm_configs,
        extraction_model=BachelorProefModel
    )
    
    # 5. Summary
    total_results = sum(len(results) for results in all_results.values())
    successful = sum(1 for results in all_results.values() for res in results if res.success)
    
    print(f"\n{'='*60}")
    print(f"BATCH TEST COMPLETED")
    print(f"{'='*60}")
    print(f"Total Extractions: {total_results}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_results - successful}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()