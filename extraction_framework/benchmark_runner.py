"""Orchestrates extraction benchmarks across PDF files and LLM configurations."""
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# EcoLogits MUST be initialized before huggingface_hub is imported anywhere.
# On Linux, EcoLogits replaces the InferenceClient class in the huggingface_hub
# namespace; any module that imported it before init() runs holds a stale reference
# and will never see .impacts on response chunks.
try:
    from ecologits import EcoLogits
    EcoLogits.init(providers=["huggingface_hub"])
    _ECOLOGITS_PREINIT_OK = True
except ImportError:
    _ECOLOGITS_PREINIT_OK = False

# Initialize global colorized print override
import extraction_framework.console

from extraction_framework.extractors import get_all_extractors
from extraction_framework.extractors.markdown_extractor import MarkdownExtractor
from extraction_framework.llm_providers import get_provider, resolve_local_model_path
from extraction_framework.llm_providers.local_server_manager import LocalServerManager
from extraction_framework.llm_providers.llama_cpp_provider import LlamaCppProvider
from extraction_framework.scoring import ResultScorer, ExtractionResult
from extraction_framework.results_db import ResultsDB
import logging

from extraction_framework.Test.modello import FactuurModel

ECOLOGITS_AVAILABLE = _ECOLOGITS_PREINIT_OK

if ECOLOGITS_AVAILABLE:
    try:
        from ecologits.tracers.huggingface_tracer import llm_impacts
        repo = llm_impacts.__globals__['models']

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


class BenchmarkRunner:
    """Runs extraction benchmarks across PDF files and LLM configurations."""

    def __init__(
        self,
        results_dir: Path = None,
        ground_truth_dir: Path = None
    ):
        self.results_dir = results_dir or Path(__file__).parent / "results"
        self.ground_truth_dir = ground_truth_dir or Path(__file__).parent / "ground_truth"
        db = ResultsDB(self.results_dir / "results.db")
        self.scorer = ResultScorer(self.results_dir, db=db)
        self.text_cache = {}  # (pdf_path, extractor_name) -> (text, real_extractor_name)

        # When True: load → infer → unload for every single (pdf × model) call.
        # When False: load model once → infer all PDFs → unload → next model.
        self._reload_per_call = os.getenv("LLAMA_RELOAD_MODEL_PER_CALL", "true").lower() == "true"

    def run_extraction(
        self,
        pdf_path: Path,
        extractor_name: str,
        llm_provider: str,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        extraction_model: Optional[type] = FactuurModel,
        use_preselection: bool = False
    ) -> ExtractionResult:
        """Run a single cloud extraction test by instantiating a temporary provider."""
        try:
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
                success=False,
                error=f"Initialization error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def run_test_suite(
        self,
        pdf_files: List[Path],
        extractors: Optional[List[str]] = None,
        llm_configs: Optional[List[Dict[str, str]]] = None,
        extraction_model: Optional[type] = FactuurModel,
        run_number: Optional[int] = None
    ) -> Dict[str, List[ExtractionResult]]:
        """Run complete test suite across all PDF files and model configurations."""
        if extractors is None:
            extractors = [e.name for e in get_all_extractors()]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_results_dir = self.results_dir / f"results_{timestamp}"
        session_results_dir.mkdir(parents=True, exist_ok=True)

        run_label = f" (run {run_number})" if run_number is not None else ""
        print(f"[BenchmarkRunner] Session results: {session_results_dir.name}{run_label}")
        print(f"[BenchmarkRunner] Model reload mode: {'per-call' if self._reload_per_call else 'per-model batch'}")

        self.scorer.results_dir = session_results_dir

        all_results = {str(pdf): [] for pdf in pdf_files}

        local_configs = [c for c in llm_configs if c["provider"].lower() == "local"]
        cloud_configs = [c for c in llm_configs if c["provider"].lower() != "local"]

        # ---------------------------------------------------------
        # 1. CLOUD MODELS
        # ---------------------------------------------------------

        # Pre-extract all texts before any cloud inference so timing is clean
        if cloud_configs:
            print("[BenchmarkRunner] Pre-extracting texts for cloud models...")
            for pdf_file in pdf_files:
                for extractor_name in extractors:
                    self._get_cached_text(pdf_file, extractor_name, extraction_model)

        for pdf_file in pdf_files:
            for llm_config in cloud_configs:
                for extractor_name in extractors:
                    print(f"\n--- CLOUD: {pdf_file.name} | {extractor_name} | {llm_config['model']} ---")
                    result = self.run_extraction(
                        pdf_file,
                        extractor_name,
                        llm_config["provider"],
                        llm_config["model"],
                        llm_config.get("api_key"),
                        extraction_model=extraction_model
                    )
                    all_results[str(pdf_file)].append(result)
                    filepath = self.scorer.save_result(result, run_number=run_number)
                    self._print_result(result, filepath)

        # ---------------------------------------------------------
        # 2. LOCAL MODELS (via llama-server)
        # ---------------------------------------------------------

        # Pre-extract all texts before starting any server so inference timing is clean
        if local_configs:
            print("[BenchmarkRunner] Pre-extracting texts...")
            for pdf_file in pdf_files:
                for extractor_name in extractors:
                    self._get_cached_text(pdf_file, extractor_name, extraction_model)

        server = LocalServerManager()

        for llm_config in local_configs:
            model_name = llm_config["model"]
            model_path = resolve_local_model_path(model_name)

            print(f"\n{'#'*60}")
            print(f"LOCAL: {model_name}  |  reload_per_call={self._reload_per_call}{run_label}")
            print(f"{'#'*60}")

            if not self._reload_per_call:
                server.start(model_path)
                provider = LlamaCppProvider(server.base_url, model_name)

            for pdf_file in pdf_files:
                for extractor_name in extractors:
                    document_text, real_extractor_name = self.text_cache.get(
                        (str(pdf_file), extractor_name), (None, extractor_name)
                    )
                    if document_text is None:
                        continue

                    if len(document_text.split()) <= 70:
                        print(f"❌ Skipped: {pdf_file.name} — too small")
                        continue

                    print(f"\n--- LOCAL: {pdf_file.name} | {real_extractor_name} | {model_name} ---")

                    if self._reload_per_call:
                        server.start(model_path)
                        provider = LlamaCppProvider(server.base_url, model_name)

                    result = self._run_extraction_with_provider(
                        pdf_path=pdf_file,
                        extractor_name=real_extractor_name,
                        provider=provider,
                        extraction_model=extraction_model,
                    )

                    if self._reload_per_call:
                        server.stop()

                    all_results[str(pdf_file)].append(result)
                    filepath = self.scorer.save_result(result, run_number=run_number)
                    self._print_result(result, filepath)

            if not self._reload_per_call:
                server.stop()

        return all_results

    def _run_extraction_with_provider(
        self,
        pdf_path: Path,
        extractor_name: str,
        provider,
        extraction_model: type
    ) -> ExtractionResult:
        """Run extraction for one PDF with an already-initialised provider."""
        llm_provider = provider.name
        llm_model = provider.model

        try:
            # 1. Text extraction with caching
            cache_key = (str(pdf_path), extractor_name)

            if cache_key in self.text_cache:
                document_text, real_extractor_name = self.text_cache[cache_key]
                print(f"[Extractor] Cache hit for {pdf_path.name}")
            else:
                extractor = MarkdownExtractor()
                if hasattr(extractor, 'set_extraction_model'):
                    extractor.set_extraction_model(extraction_model)

                document_text = extractor.extract_text(pdf_path)
                real_extractor_name = extractor.name
                self.text_cache[cache_key] = (document_text, real_extractor_name)
                print(f"[Extractor] {pdf_path.name}: {len(document_text.split())} words")

                debug_dir = Path(__file__).parent.parent / "debug_payloads"
                debug_dir.mkdir(exist_ok=True)
                debug_file = debug_dir / f"{pdf_path.stem}_{real_extractor_name}.txt"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(document_text)

            # 2. Word count guard
            word_count = len(document_text.split())
            if word_count <= 70:
                raise ValueError(f"Document too small for scientific comparison ({word_count} words)")

            # 3. LLM inference
            extracted, token_usage = provider.extract_structured_data(
                text=document_text,
                schema=extraction_model
            )

            # 4. Build result
            extracted_dict = extracted.model_dump()
            return ExtractionResult(
                pdf_file=str(pdf_path),
                extractor_name=real_extractor_name,
                llm_provider=llm_provider,
                llm_model=llm_model,
                ttft_seconds=token_usage.get('ttft_seconds'),
                generation_seconds=token_usage.get('generation_seconds'),
                total_inference_seconds=token_usage.get('total_inference_seconds'),
                success=True,
                validation_score=self.scorer.validate_and_score(extracted_dict),
                extracted_data=extracted_dict,
                timestamp=datetime.now().isoformat(),
                input_tokens=token_usage.get('input', 0),
                output_tokens=token_usage.get('output', 0),
                total_tokens=token_usage.get('total', 0),
                raw_energy_kwh=token_usage.get('raw_energy_kwh'),
                energy_kwh_with_pue=token_usage.get('energy_kwh_with_pue'),
                co2_kg=token_usage.get('co2_kg'),
                cpu_energy_kwh=token_usage.get('cpu_energy_kwh'),
                gpu_energy_kwh=token_usage.get('gpu_energy_kwh'),
                ram_energy_kwh=token_usage.get('ram_energy_kwh'),
                energy_source=token_usage.get('energy_source'),
                regional_cloud_projections=token_usage.get('regional_cloud_projections'),
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
                success=False,
                validation_score=0.0,
                error=str(e),
                extracted_data=extracted_data,
                timestamp=datetime.now().isoformat(),
                ttft_seconds=token_usage.get('ttft_seconds'),
                generation_seconds=token_usage.get('generation_seconds'),
                total_inference_seconds=token_usage.get('total_inference_seconds'),
                input_tokens=token_usage.get('input', 0),
                output_tokens=token_usage.get('output', 0),
                total_tokens=token_usage.get('total', 0),
                raw_energy_kwh=token_usage.get('raw_energy_kwh'),
                energy_kwh_with_pue=token_usage.get('energy_kwh_with_pue'),
                co2_kg=token_usage.get('co2_kg'),
                cpu_energy_kwh=token_usage.get('cpu_energy_kwh'),
                gpu_energy_kwh=token_usage.get('gpu_energy_kwh'),
                ram_energy_kwh=token_usage.get('ram_energy_kwh'),
                energy_source=token_usage.get('energy_source'),
                regional_cloud_projections=token_usage.get('regional_cloud_projections'),
            )

    def _get_cached_text(
        self,
        pdf_path: Path,
        extractor_name: str,
        extraction_model: type
    ) -> Optional[str]:
        """Return cached extracted text, extracting and caching if not yet seen."""
        cache_key = (str(pdf_path), extractor_name)
        if cache_key in self.text_cache:
            return self.text_cache[cache_key][0]

        try:
            extractor = MarkdownExtractor()
            if hasattr(extractor, 'set_extraction_model'):
                extractor.set_extraction_model(extraction_model)
            document_text = extractor.extract_text(pdf_path)
            self.text_cache[cache_key] = (document_text, extractor.name)

            debug_dir = Path(__file__).parent.parent / "debug_payloads"
            debug_dir.mkdir(exist_ok=True)
            debug_file = debug_dir / f"{pdf_path.stem}_{extractor.name}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(document_text)

            return document_text
        except Exception as e:
            print(f"❌ Extraction failed for {pdf_path.name}: {e}")
            return None

    @staticmethod
    def _print_result(result: ExtractionResult, filepath: Path) -> None:
        name = Path(result.pdf_file).name
        if result.success:
            print(f"✓ {name} saved to {filepath.name}")
        else:
            print(f"❌ {name} — {result.error}")


if __name__ == "__main__":
    pass
