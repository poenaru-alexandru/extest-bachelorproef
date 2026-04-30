"""Main test runner orchestrating all extraction strategies"""
import sys
import subprocess
import json
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

from Test.modello import BachelorProefModel

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
                'name': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
                'architecture': {'type': 'dense', 'parameters': 7.61}
            },
            {
                'provider': 'huggingface_hub',
                'name': 'meta-llama/llama-3.1-8b-instruct',
                'architecture': {'type': 'dense', 'parameters': 8.03}
            },
            {
                'provider': 'huggingface_hub',
                'name': 'swiss-ai/Apertus-8B-Instruct-2509',
                'architecture': {'type': 'dense', 'parameters': 8}
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
        ground_truth_dir: Path = None
    ):
        self.results_dir = results_dir or Path(__file__).parent / "results"
        self.ground_truth_dir = ground_truth_dir or Path(__file__).parent / "ground_truth"
        self.scorer = ResultScorer(self.results_dir)
        self.text_cache = {}  # Cache for extracted document text: (pdf_path, extractor_name) -> text

    def run_extraction(
        self,
        pdf_path: Path,
        extractor_name: str,
        llm_provider: str,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        extraction_model: Optional[type] = BachelorProefModel,
        use_preselection: bool = False
    ) -> ExtractionResult:
        """Run a single cloud extraction test by instantiating a temporary provider"""
        try:
            # Instantiate provider 
            provider = get_provider(llm_provider, llm_model, llm_api_key)
            try:
                return self._run_extraction_with_provider(
                    pdf_path=pdf_path,
                    extractor_name=extractor_name,
                    provider=provider,
                    extraction_model=extraction_model
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
        extraction_model: Optional[type] = BachelorProefModel
    ) -> Dict[str, List[ExtractionResult]]:
        """Run complete test suite
        
        Args:
            pdf_files: List of PDF files to test
            extractors: List of extractor names (None = all available)
            llm_configs: List of LLM configurations
            extraction_model: Pydantic model for extraction
            
        Returns:
            Dictionary mapping PDF files to their results
        """
        if extractors is None:
            extractors = [e.name for e in get_all_extractors()]
        
        # Create a session-specific results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_results_dir = self.results_dir / f"results_{timestamp}"
        session_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TestRunner] Created session results directory: {session_results_dir.name}")
        
        # Update scorer to use the session directory for this suite run
        self.scorer.results_dir = session_results_dir
            
        all_results = {str(pdf): [] for pdf in pdf_files}
        
        # Separate configs into local and cloud to optimize loading
        local_configs = [c for c in llm_configs if c["provider"].lower() == "local"]
        cloud_configs = [c for c in llm_configs if c["provider"].lower() != "local"]
        
        # ---------------------------------------------------------
        # 1. PROCESS CLOUD MODELS (Original Native Logic)
        # ---------------------------------------------------------
        for pdf_file in pdf_files:
            for llm_config in cloud_configs:
                for extractor_name in extractors:
                    print(f"\n--- Testing CLOUD: {pdf_file.name} | {extractor_name} | {llm_config['model']} ---")
                    result = self.run_extraction(
                        pdf_file,
                        extractor_name,
                        llm_config["provider"],
                        llm_config["model"],
                        llm_config.get("api_key"),
                        extraction_model=extraction_model
                    )
                    all_results[str(pdf_file)].append(result)
                    filepath = self.scorer.save_result(result)
                    
                    if result.success:
                        print(f"✓ Success: {pdf_file.name} saved to {filepath.name}")
                    else:
                        print(f"❌ Failed: {pdf_file.name} - {result.error}")

        # ---------------------------------------------------------
        # 2. PROCESS LOCAL MODELS (Isolated Worker Method)
        # ---------------------------------------------------------
        for llm_config in local_configs:
            model_name = llm_config["model"]
            print(f"\n{'#'*60}")
            print(f"BATCH RUN: Local Model {model_name} (ISOLATED)")
            print(f"{'#'*60}")
            
            for pdf_file in pdf_files:
                for extractor_name in extractors:
                    start_time = time.time()
                    
                    # A. Standardized Payload Extraction & Caching
                    cache_key = (str(pdf_file), extractor_name)
                    if cache_key not in self.text_cache:
                        extractor = MarkdownExtractor()
                        if hasattr(extractor, 'set_extraction_model'):
                            extractor.set_extraction_model(extraction_model)
                            
                        document_text = extractor.extract_text(pdf_file)
                        self.text_cache[cache_key] = (document_text, extractor.name)
                        
                        # Save the payload so the isolated worker can read it without parsing PDF
                        debug_dir = Path(__file__).parent.parent / "debug_payloads"
                        debug_dir.mkdir(exist_ok=True)
                        payload_file = debug_dir / f"{pdf_file.stem}_{extractor.name}.txt"
                        with open(payload_file, "w", encoding="utf-8") as f:
                            f.write(document_text)
                    
                    real_extractor_name = self.text_cache[cache_key][1]
                    payload_file = Path(__file__).parent.parent / "debug_payloads" / f"{pdf_file.stem}_{real_extractor_name}.txt"
                    worker_output_file = Path(__file__).parent.parent / "debug_payloads" / f"{pdf_file.stem}_worker_result.json"

                    # B. Word count check to prevent processing junk
                    with open(payload_file, "r", encoding="utf-8") as f:
                        word_count = len(f.read().split())
                    if word_count <= 70:
                        print(f"❌ Failed: {pdf_file.name} - Document too small ({word_count} words)")
                        continue

                    print(f"\n--- Booting ISOLATED worker: {pdf_file.name} | {real_extractor_name} | {model_name} ---")
                    
                    # C. Launch the sacrificial worker process
                    subprocess.run([
                        sys.executable, # Uses the current python environment safely
                        "isolated_worker.py", 
                        str(payload_file), 
                        model_name, 
                        str(worker_output_file)
                    ])
                    
                    # D. Worker is dead. VRAM is perfectly flushed. Reconstruct results.
                    if not worker_output_file.exists():
                        print(f"❌ Failed: {pdf_file.name} - Worker crashed catastrophically without returning data.")
                        continue
                        
                    with open(worker_output_file, "r", encoding="utf-8") as f:
                        worker_data = json.load(f)
                    
                    result = ExtractionResult(
                        pdf_file=str(pdf_file), 
                        extractor_name=real_extractor_name, 
                        llm_provider="local", 
                        llm_model=model_name,
                        extraction_time=time.time() - start_time,
                        success=worker_data["success"],
                        error=worker_data.get("error"),
                        extracted_data=worker_data.get("data"),
                        timestamp=datetime.now().isoformat(),
                        ttft_seconds=worker_data.get("token_usage", {}).get('ttft_seconds'),
                        generation_seconds=worker_data.get("token_usage", {}).get('generation_seconds'),
                        total_inference_seconds=worker_data.get("token_usage", {}).get('total_inference_seconds'),
                        input_tokens=worker_data.get("token_usage", {}).get('input', 0),
                        output_tokens=worker_data.get("token_usage", {}).get('output', 0),
                        total_tokens=worker_data.get("token_usage", {}).get('total', 0),
                        energy_kwh=worker_data.get("emissions", {}).get('energy_kwh'),
                        co2_kg=worker_data.get("emissions", {}).get('co2_kg'),
                        cpu_energy_kwh=worker_data.get("emissions", {}).get('cpu_energy_kwh'),
                        gpu_energy_kwh=worker_data.get("emissions", {}).get('gpu_energy_kwh'),
                        ram_energy_kwh=worker_data.get("emissions", {}).get('ram_energy_kwh'),
                        energy_source=worker_data.get("emissions", {}).get('energy_source')
                    )
                    
                    # Clean up the temp JSON file
                    worker_output_file.unlink(missing_ok=True)
                    
                    all_results[str(pdf_file)].append(result)
                    filepath = self.scorer.save_result(result)
                    
                    if result.success:
                        print(f"✓ Success: {pdf_file.name} saved to {filepath.name}")
                    else:
                        print(f"❌ Failed: {pdf_file.name} - {result.error}")
        
        return all_results

    def _run_extraction_with_provider(
        self,
        pdf_path: Path,
        extractor_name: str,
        provider,
        extraction_model: type
    ) -> ExtractionResult:
        """Internal helper strictly for Cloud models (EcoLogits tracking built into provider)"""
        start_time = time.time()
        llm_provider = provider.name
        llm_model = provider.model
        
        try:
            # 1. STANDARDIZED PAYLOAD EXTRACTION
            cache_key = (str(pdf_path), extractor_name)
            
            if cache_key in self.text_cache:
                document_text, real_extractor_name = self.text_cache[cache_key]
                print(f"[Extractor] Using cached text for {pdf_path.name}")
            else:
                extractor = MarkdownExtractor()
                if hasattr(extractor, 'set_extraction_model'):
                    extractor.set_extraction_model(extraction_model)
                
                document_text = extractor.extract_text(pdf_path)
                real_extractor_name = extractor.name
                self.text_cache[cache_key] = (document_text, real_extractor_name)
                print(f"[Extractor] Extracted document text with {len(document_text.split())} words")
                
                debug_dir = Path(__file__).parent.parent / "debug_payloads"
                debug_dir.mkdir(exist_ok=True)
                debug_file = debug_dir / f"{pdf_path.stem}_{real_extractor_name}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(document_text)

            # 2. Word count check
            word_count = len(document_text.split())
            if word_count <= 70:
                raise ValueError(f"Document too small for scientific comparison ({word_count} words)")

            # 3. Cloud LLM Inference (EcoLogits tracks this automatically via wrapper)
            extracted, token_usage = provider.extract_structured_data(text=document_text, schema=extraction_model)

            # 4. Finalization
            extraction_time = time.time() - start_time
            return ExtractionResult(
                pdf_file=str(pdf_path), 
                extractor_name=real_extractor_name, 
                llm_provider=llm_provider, 
                llm_model=llm_model,
                extraction_time=extraction_time,
                ttft_seconds=token_usage.get('ttft_seconds'),
                generation_seconds=token_usage.get('generation_seconds'),
                total_inference_seconds=token_usage.get('total_inference_seconds'),
                success=True, 
                extracted_data=extracted.model_dump(), 
                timestamp=datetime.now().isoformat(),
                input_tokens=token_usage.get('input', 0), 
                output_tokens=token_usage.get('output', 0), 
                total_tokens=token_usage.get('total', 0),
                energy_kwh=token_usage.get('energy_kwh'),
                co2_kg=token_usage.get('co2_kg'),
                energy_source=token_usage.get('energy_source')
            )
        except Exception as e:
            from extraction_framework.llm_providers.base_provider import ExtractionError
            token_usage = getattr(e, 'token_usage', {})
            raw_content = getattr(e, 'raw_content', None)
            extracted_data = {"raw_content": raw_content} if raw_content else None
            
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
                energy_kwh=token_usage.get('energy_kwh'),
                co2_kg=token_usage.get('co2_kg'),
                energy_source=token_usage.get('energy_source')
            )

if __name__ == "__main__":
    # Standard CLI check remains exactly the same...
    pass