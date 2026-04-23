"""Main test runner orchestrating all extraction strategies"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extraction_framework.extractors import get_all_extractors, get_extractor_by_name
from extraction_framework.llm_providers import get_provider
from extraction_framework.scoring import ResultScorer, ExtractionResult
from extraction_framework.page_validator import PageValidator, load_validation_rules_from_model
import logging

try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


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
    
    def set_scoring_rules_for_test(self, test_folder: str):
        """Update scorer with rules from test folder's regole.py
        
        Args:
            test_folder: Name of the test folder
        """
        if not self.model_loader:
            return
        
        unique_ids, ignored_fields = self.model_loader.load_scoring_rules(test_folder)
        
        # Create new scorer with custom rules
        self.scorer = ResultScorer(
            self.results_dir,
            ignored_fields=ignored_fields,
            unique_identifiers=unique_ids
        )
    
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
        
        if llm_configs is None:
            llm_configs = [{"provider": "local", "model": "llama3.1:8b"}]
            
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
                    self.set_scoring_rules_for_test(test_folder)
                    # Resolve models
                    current_model, current_module = self._resolve_model(test_folder, extraction_model, extraction_model_module)
                    for extractor_name in extractors:
                        print(f"\n--- Testing: {pdf_file.name} | {extractor_name} ---")
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
            
            # Use specific rules for this test folder
            self.set_scoring_rules_for_test(test_folder)
            
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
                    print(f"\n--- Testing: {pdf_file.name} | {extractor_name} | {llm_config['provider']} ---")
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
        # This is a refactored version of run_extraction that accepts a provider object
        # to avoid the get_provider call which triggers model reloading
        start_time = time.time()
        llm_provider = provider.name
        llm_model = provider.model
        
        try:
            # Re-use the core logic from run_extraction
            # (In a real refactoring, run_extraction would call this method)
            
            # Get extractor
            extractor = get_extractor_by_name(extractor_name)
            if hasattr(extractor, 'set_extraction_model'):
                extractor.set_extraction_model(extraction_model)
            
            # Pre-filtering (copied from run_extraction)
            filtered_pdf_path = pdf_path
            page_filter_stats = None
            if extraction_model_module:
                validation_rules = load_validation_rules_from_model(extraction_model_module)
                if validation_rules:
                    validator = PageValidator(validation_rules)
                    import fitz
                    doc = fitz.open(pdf_path)
                    pages_to_keep = []
                    for page_num in range(len(doc)):
                        if validator.validate_page(doc[page_num].get_text()):
                            pages_to_keep.append(page_num)
                    doc.close()
                    if pages_to_keep:
                        first, last = min(pages_to_keep), max(pages_to_keep)
                        filtered_range = list(range(first, last + 1))
                        if len(filtered_range) < len(doc):
                            import tempfile
                            temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                            temp_pdf.close()
                            filtered_pdf_path = Path(temp_pdf.name)
                            doc = fitz.open(pdf_path)
                            filtered_doc = fitz.open()
                            for p in filtered_range:
                                filtered_doc.insert_pdf(doc, from_page=p, to_page=p)
                            filtered_doc.save(filtered_pdf_path)
                            filtered_doc.close()
                            doc.close()
                            page_filter_stats = {'total_pages': len(doc), 'filtered_pages': len(filtered_range), 'total_removed': len(doc)-len(filtered_range)}

            # Word count check
            import fitz
            doc = fitz.open(filtered_pdf_path)
            text = "".join([p.get_text() for p in doc])
            doc.close()
            if len(text.split()) <= 70:
                raise ValueError("Document too small")

            # Extraction
            if extractor.name != "PDF-Direct":
                text = extractor.extract_text(filtered_pdf_path)

            # LLM Call (Using the provided provider)
            emissions_data = {}
            if CODECARBON_AVAILABLE and llm_provider.lower() == "local":
                tracker = OfflineEmissionsTracker(project_name=f"ExTest_{llm_model}", measure_power_secs=1, save_to_file=False, logging_logger=logging.getLogger(__name__), country_iso_code="ITA")
                tracker.start()
                try:
                    extracted, token_usage = provider.extract_structured_data(text=text, schema=extraction_model)
                finally:
                    tracker.stop()
                    if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                        d = tracker.final_emissions_data
                        emissions_data = {'energy_kwh': d.energy_consumed, 'co2_kg': d.emissions, 'cpu_energy_kwh': d.cpu_energy, 'gpu_energy_kwh': d.gpu_energy, 'ram_energy_kwh': d.ram_energy, 'energy_source': 'codecarbon'}
            else:
                extracted, token_usage = provider.extract_structured_data(text=text, schema=extraction_model)

            # Cleanup
            if filtered_pdf_path != pdf_path and filtered_pdf_path.exists():
                filtered_pdf_path.unlink()

            extraction_time = time.time() - start_time
            return ExtractionResult(
                pdf_file=str(pdf_path), extractor_name=extractor_name, llm_provider=llm_provider, llm_model=llm_model,
                extraction_time=extraction_time, success=True, extracted_data=extracted.model_dump(), timestamp=datetime.now().isoformat(),
                input_tokens=token_usage.get('input', 0), output_tokens=token_usage.get('output', 0), total_tokens=token_usage.get('total', 0),
                energy_kwh=emissions_data.get('energy_kwh') or token_usage.get('energy_kwh'),
                co2_kg=emissions_data.get('co2_kg') or token_usage.get('co2_kg'),
                cpu_energy_kwh=emissions_data.get('cpu_energy_kwh'), gpu_energy_kwh=emissions_data.get('gpu_energy_kwh'), ram_energy_kwh=emissions_data.get('ram_energy_kwh'),
                energy_source=emissions_data.get('energy_source') or token_usage.get('energy_source'),
                page_filter_stats=page_filter_stats
            )
        except Exception as e:
            if filtered_pdf_path != pdf_path and filtered_pdf_path.exists():
                filtered_pdf_path.unlink()
            return ExtractionResult(pdf_file=str(pdf_path), extractor_name=extractor_name, llm_provider=llm_provider, llm_model=llm_model, extraction_time=time.time()-start_time, success=False, error=str(e), timestamp=datetime.now().isoformat())


if __name__ == "__main__":
    # Example usage
    runner = TestRunner()
    
    # Find all PDFs in test directory
    test_dir = Path(__file__).parent.parent / "Test" / "bolletta_ee_cenpi"
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in test directory")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Define LLM configurations
    llm_configs = [
        {"provider": "openai", "model": "gpt-4o", "api_key": None},
        # Add more configurations as needed
    ]
    
    # Run tests
    results = runner.run_test_suite(
        pdf_files[:1],  # Test first PDF only for now
        extractors=["PyMuPDF"],  # Test one extractor for now
        llm_configs=llm_configs
    )
