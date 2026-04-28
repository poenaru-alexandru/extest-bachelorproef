"""Main test runner orchestrating all extraction strategies"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize global colorized print override
import extraction_framework.console

from extraction_framework.extractors import get_all_extractors
from extraction_framework.extractors.markdown_extractor import MarkdownExtractor
from extraction_framework.llm_providers import get_provider
from extraction_framework.scoring import ResultScorer, ExtractionResult
import logging

try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

try:
    from ecologits import EcoLogits
    EcoLogits.init(providers=["huggingface_hub"])
    
    # Manual registration for EcoLogits to handle custom model names/aliases
    try:
        from ecologits.tracers.huggingface_tracer import llm_impacts
        repo = llm_impacts.__globals__['models']
        
        # Mapping for common aliases
        custom_models = [
            {
                'provider': 'huggingface_hub',
                'name': 'qwen/qwen3-vl-8b-instruct',
                'architecture': {'type': 'dense', 'parameters': 9}
            },
            {
                'provider': 'huggingface_hub',
                'name': 'meta-llama/llama-3.1-8b-instruct',
                'architecture': {'type': 'dense', 'parameters': 8.03}
            }
        ]
        
        for m_data in custom_models:
            repo.add_model(m_data)
        print("[EcoLogits] Successfully registered custom model aliases")
    except Exception as e:
        print(f"[EcoLogits] Warning: Could not register custom model aliases: {e}")
        
    ECOLOGITS_AVAILABLE = True
except ImportError:
    ECOLOGITS_AVAILABLE = False


class TestRunner:
    """Run extraction tests with various configurations"""
    
    def __init__(
        self, 
        results_dir: Path = None,
        ground_truth_dir: Path = None,
        model_loader = None
    ):
        self.results_dir = results_dir or Path(__file__).parent / "results"
        self.ground_truth_dir = ground_truth_dir or Path(__file__).parent / "ground_truth"
        self.scorer = ResultScorer(self.results_dir)
        self.model_loader = model_loader
    
    def _resolve_model(self, test_folder: str, extraction_model: Optional[type], extraction_model_module):
        """Helper to resolve the extraction model and module for a folder"""
        current_model = extraction_model
        current_module = extraction_model_module
        if current_model is None and self.model_loader:
            try:
                current_model = self.model_loader.get_model_for_test(test_folder)
                current_module = self.model_loader.get_module_for_test(test_folder)
            except Exception as e:
                print(f"Warning: Could not load model for {test_folder}: {e}")
        return current_model, current_module

    def run_extraction(
        self,
        pdf_path: Path,
        extractor_name: str,
        llm_provider: str,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        extraction_model: Optional[type] = None,
        extraction_model_module = None,
        use_preselection: bool = False
    ) -> ExtractionResult:
        """Run a single extraction test by instantiating a temporary provider"""
        try:
            # Resolve model if needed
            test_folder = pdf_path.parent.name
            current_model, current_module = self._resolve_model(test_folder, extraction_model, extraction_model_module)
            
            # Instantiate provider (and ensure it's closed if local)
            provider = get_provider(llm_provider, llm_model, llm_api_key)
            try:
                return self._run_extraction_with_provider(
                    pdf_path=pdf_path,
                    extractor_name=extractor_name,
                    provider=provider,
                    extraction_model=current_model,
                    extraction_model_module=current_module
                )
            finally:
                if hasattr(provider, 'close'):
                    provider.close()
        except Exception as e:
            return ExtractionResult(
                pdf_file=str(pdf_path),
                extractor_name=extractor_name,
                llm_provider=llm_provider,
                llm_model=llm_model or "unknown",
                extraction_time=0,
                success=False,
                error=f"Initialization error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
    
    def run_test_suite(
        self,
        pdf_files: List[Path],
        extractors: Optional[List[str]] = None,
        llm_configs: Optional[List[Dict[str, str]]] = None,
        extraction_model: Optional[type] = None,
        extraction_model_module = None
    ) -> Dict[str, List[ExtractionResult]]:
        """Run complete test suite with Model-first batching for local models
        
        Args:
            pdf_files: List of PDF files to test
            extractors: List of extractor names (None = all available)
            llm_configs: List of LLM configurations
            extraction_model: Pydantic model for extraction
            extraction_model_module: Module containing model and rules
            
        Returns:
            Dictionary mapping PDF files to their results
        """
        if extractors is None:
            extractors = [e.name for e in get_all_extractors()]
        
            
        all_results = {str(pdf): [] for pdf in pdf_files}
        
        # Separate configs into local and cloud to optimize loading
        local_configs = [c for c in llm_configs if c["provider"].lower() == "local"]
        cloud_configs = [c for c in llm_configs if c["provider"].lower() != "local"]
        
        # 1. PROCESS LOCAL MODELS (Model-first to avoid reload overhead)
        for llm_config in local_configs:
            model_name = llm_config["model"]
            print(f"\n{'#'*60}")
            print(f"BATCH RUN: Local Model {model_name}")
            print(f"{'#'*60}")
            
            provider = None
            try:
                # Try to instantiate provider once
                provider = get_provider("local", model_name)
                for pdf_file in pdf_files:
                    test_folder = pdf_file.parent.name
                    # Resolve models
                    current_model, current_module = self._resolve_model(test_folder, extraction_model, extraction_model_module)
                    for extractor_name in extractors:
                        print(f"\n--- Testing: {pdf_file.name} | {extractor_name} | {model_name} ---")
                        result = self._run_extraction_with_provider(
                            pdf_path=pdf_file,
                            extractor_name=extractor_name,
                            provider=provider,
                            extraction_model=current_model,
                            extraction_model_module=current_module
                        )
                        all_results[str(pdf_file)].append(result)
                        self.scorer.save_result(result)
                
            except Exception as e:
                print(f"Batch load failed for model {model_name}: {e}")
                # Create failed results for everything we intended to test with this model
                for pdf_file in pdf_files:
                    for extractor_name in extractors:
                        fail_res = ExtractionResult(
                            pdf_file=str(pdf_file),
                            extractor_name=extractor_name,
                            llm_provider="local",
                            llm_model=model_name,
                            extraction_time=0,
                            success=False,
                            error=f"Model load error: {str(e)}",
                            timestamp=datetime.now().isoformat()
                        )
                        all_results[str(pdf_file)].append(fail_res)
                        self.scorer.save_result(fail_res)
            finally:
                # Unload model after batch
                if provider and hasattr(provider, 'close'):
                    print(f"[LlamaCpp] Unloading model {model_name}")
                    provider.close()

        # 2. PROCESS CLOUD MODELS (Original Document-first logic)
        for pdf_file in pdf_files:
            test_folder = pdf_file.parent.name
            
            # Resolve models if not fixed
            current_model = extraction_model
            current_module = extraction_model_module
            if current_model is None and self.model_loader:
                try:
                    current_model = self.model_loader.get_model_for_test(test_folder)
                    current_module = self.model_loader.get_module_for_test(test_folder)
                except: pass

            for llm_config in cloud_configs:
                for extractor_name in extractors:
                    print(f"\n--- Testing: {pdf_file.name} | {extractor_name} | {llm_config['model']} ---")
                    result = self.run_extraction(
                        pdf_file,
                        extractor_name,
                        llm_config["provider"],
                        llm_config["model"],
                        llm_config.get("api_key"),
                        extraction_model=current_model,
                        extraction_model_module=current_module
                    )
                    all_results[str(pdf_file)].append(result)
                    self.scorer.save_result(result)
        
        return all_results

    def _run_extraction_with_provider(
        self,
        pdf_path: Path,
        extractor_name: str,
        provider,
        extraction_model: type,
        extraction_model_module = None
    ) -> ExtractionResult:
        """Internal helper to run extraction using a pre-instantiated provider"""
        start_time = time.time()
        llm_provider = provider.name
        llm_model = provider.model
        
        try:
            # 1. Get designated extractor (Standardized on Markdown for scientific rigor)
            extractor = MarkdownExtractor()

            if hasattr(extractor, 'set_extraction_model'):
                extractor.set_extraction_model(extraction_model)
            
            # 2. STANDARDIZED PAYLOAD EXTRACTION (Extract EXACTLY ONCE per document)
            # This ensures all models receive the identical payload string for valid comparison.
            document_text = extractor.extract_text(pdf_path)

            # 3. Word count check
            word_count = len(document_text.split())
            if word_count <= 70:
                raise ValueError(f"Document too small for scientific comparison ({word_count} words)")

            # 4. LLM Inference & Emissions Tracking
            emissions_data = {}
            if CODECARBON_AVAILABLE and llm_provider.lower() == "local":
                tracker = OfflineEmissionsTracker(
                    project_name=f"ExTest_{llm_model}", 
                    measure_power_secs=1, 
                    save_to_file=False, 
                    logging_logger=logging.getLogger(__name__), 
                    country_iso_code="ITA"
                )
                tracker.start()
                try:
                    # Pass the identical document_text payload to local model
                    extracted, token_usage = provider.extract_structured_data(text=document_text, schema=extraction_model)
                finally:
                    tracker.stop()
                    if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                        d = tracker.final_emissions_data
                        emissions_data = {
                            'energy_kwh': d.energy_consumed, 
                            'co2_kg': d.emissions, 
                            'cpu_energy_kwh': d.cpu_energy, 
                            'gpu_energy_kwh': d.gpu_energy, 
                            'ram_energy_kwh': d.ram_energy, 
                            'energy_source': 'codecarbon'
                        }
            else:
                # Pass identical document_text payload to cloud model (EcoLogits tracked inside)
                extracted, token_usage = provider.extract_structured_data(text=document_text, schema=extraction_model)

            # 5. Finalization
            extraction_time = time.time() - start_time
            return ExtractionResult(
                pdf_file=str(pdf_path), 
                extractor_name=extractor.name, 
                llm_provider=llm_provider, 
                llm_model=llm_model,
                extraction_time=extraction_time,
                
                # Streaming Telemetry
                ttft_seconds=token_usage.get('ttft_seconds'),
                generation_seconds=token_usage.get('generation_seconds'),
                total_inference_seconds=token_usage.get('total_inference_seconds'),
                
                success=True, 
                extracted_data=extracted.model_dump(), 
                timestamp=datetime.now().isoformat(),
                input_tokens=token_usage.get('input', 0), 
                output_tokens=token_usage.get('output', 0), 
                total_tokens=token_usage.get('total', 0),
                energy_kwh=emissions_data.get('energy_kwh') or token_usage.get('energy_kwh'),
                co2_kg=emissions_data.get('co2_kg') or token_usage.get('co2_kg'),
                cpu_energy_kwh=emissions_data.get('cpu_energy_kwh'), 
                gpu_energy_kwh=emissions_data.get('gpu_energy_kwh'), 
                ram_energy_kwh=emissions_data.get('ram_energy_kwh'),
                energy_source=emissions_data.get('energy_source') or token_usage.get('energy_source')
            )
        except Exception as e:
            from extraction_framework.llm_providers.base_provider import ExtractionError
            token_usage = getattr(e, 'token_usage', {})
            raw_content = getattr(e, 'raw_content', None)
            extracted_data = {"raw_content": raw_content} if raw_content else None
            
            # Retrieve emissions data if it was set before exception
            emissions = locals().get('emissions_data', {})
            
            return ExtractionResult(
                pdf_file=str(pdf_path), 
                extractor_name=extractor_name, 
                llm_provider=llm_provider, 
                llm_model=llm_model, 
                extraction_time=time.time() - start_time, 
                success=False, 
                error=str(e), 
                extracted_data=extracted_data,
                timestamp=datetime.now().isoformat(),
                ttft_seconds=token_usage.get('ttft_seconds'),
                generation_seconds=token_usage.get('generation_seconds'),
                total_inference_seconds=token_usage.get('total_inference_seconds'),
                input_tokens=token_usage.get('input', 0),
                output_tokens=token_usage.get('output', 0),
                total_tokens=token_usage.get('total', 0),
                energy_kwh=emissions.get('energy_kwh') or token_usage.get('energy_kwh'),
                co2_kg=emissions.get('co2_kg') or token_usage.get('co2_kg'),
                cpu_energy_kwh=emissions.get('cpu_energy_kwh'), 
                gpu_energy_kwh=emissions.get('gpu_energy_kwh'), 
                ram_energy_kwh=emissions.get('ram_energy_kwh'),
                energy_source=emissions.get('energy_source') or token_usage.get('energy_source')
            )


if __name__ == "__main__":
    import os
    import json
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Example usage
    runner = TestRunner()
    
    # Find all PDFs in test directory
    test_dir = Path(__file__).parent.parent / "Test" / "bolletta_ee_cenpi"
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in test directory")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Define LLM configurations dynamically from .env
    env_config = os.getenv("LLM_PROVIDERS")
    llm_configs = []
    
    if env_config:
        try:
            parsed_config = json.loads(env_config)
            for provider_key, config_data in parsed_config.items():
                if not isinstance(config_data, dict):
                    continue
                api_key = config_data.get("api_key")
                base_url = config_data.get("base_url")
                models = config_data.get("models", [])
                
                for model_name in models:
                    llm_configs.append({
                        "provider": provider_key,
                        "model": model_name,
                        "api_key": api_key,
                        "base_url": base_url
                    })
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM_PROVIDERS JSON: {e}")
            sys.exit(1)
    else:
        print("LLM_PROVIDERS environment variable not found. Please define it in your .env file.")
        sys.exit(1)
        
    print(f"Loaded {len(llm_configs)} LLM configurations.")
    
    # Run tests
    results = runner.run_test_suite(
        pdf_files[:1],  # Test first PDF only for now
        extractors=["markdown"],  # Test one extractor for now
        llm_configs=llm_configs
    )
