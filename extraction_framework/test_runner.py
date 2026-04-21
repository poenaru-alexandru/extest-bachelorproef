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
        """Run a single extraction test
        
        Args:
            pdf_path: Path to PDF file
            extractor_name: Name of PDF extractor to use
            llm_provider: Name of LLM provider
            llm_model: LLM model identifier (if None, uses env default)
            llm_api_key: API key for LLM provider (if None, uses env)
            extraction_model: Pydantic model for extraction
            extraction_model_module: Module containing model and PAGE_VALIDATION_RULES
            use_preselection: Deprecated and ignored
            
        Returns:
            Extraction result
        """
        start_time = time.time()
        
        # Use a default model if not provided
        if extraction_model is None:
            raise ValueError("Extraction model must be provided")
        
        try:
            is_pdf = pdf_path.suffix.lower() == ".pdf"
            if not is_pdf:
                raise ValueError(f"Unsupported file type: {pdf_path.suffix}. Only PDF is supported.")

            # Get extractor only for PDF processing pipelines.
            extractor = get_extractor_by_name(extractor_name)

            # If extractor supports it, pass the extraction model for context
            if hasattr(extractor, 'set_extraction_model'):
                extractor.set_extraction_model(extraction_model)
            
            # ====== STEP 1: PAGE PRE-FILTERING (COMMON TO ALL EXTRACTORS) ======
            # Apply regex validation to filter pages BEFORE any extraction
            page_filter_stats = None
            filtered_pdf_path = pdf_path  # Default: use original PDF
            
            if extraction_model_module:
                validation_rules = load_validation_rules_from_model(extraction_model_module)
                if validation_rules:
                    print(f"Pre-filtering PDF pages with regex validation...")
                    validator = PageValidator(validation_rules)
                    
                    # Get page ranges to keep using PyMuPDF for fast page detection
                    import fitz
                    doc = fitz.open(pdf_path)
                    pages_to_keep = []
                    
                    total_pages = len(doc)
                    
                    for page_num in range(total_pages):
                        page = doc[page_num]
                        page_text = page.get_text()
                        
                        # Check if page is valid
                        if validator.validate_page(page_text):
                            pages_to_keep.append(page_num)
                    
                    doc.close()
                    
                    # Calculate stats
                    # Filter consecutive invalid pages from head/tail, keep invalid in middle
                    if pages_to_keep:
                        first_valid = min(pages_to_keep)
                        last_valid = max(pages_to_keep)
                        filtered_pages = list(range(first_valid, last_valid + 1))
                    else:
                        filtered_pages = list(range(total_pages))  # Keep all if none valid
                    
                    removed_head = filtered_pages[0] if filtered_pages else 0
                    removed_tail = total_pages - (filtered_pages[-1] + 1) if filtered_pages else 0
                    
                    page_filter_stats = {
                        'total_pages': total_pages,
                        'validated_pages': len(pages_to_keep),
                        'filtered_pages': len(filtered_pages),
                        'removed_from_head': removed_head,
                        'removed_from_tail': removed_tail,
                        'total_removed': removed_head + removed_tail
                    }
                    
                    # Create filtered PDF if pages were removed
                    if page_filter_stats['total_removed'] > 0:
                        import tempfile
                        import os
                        
                        # Create temp filtered PDF
                        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                        temp_pdf.close()
                        filtered_pdf_path = Path(temp_pdf.name)
                        
                        # Copy only filtered pages
                        doc = fitz.open(pdf_path)
                        filtered_doc = fitz.open()
                        for page_num in filtered_pages:
                            filtered_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        filtered_doc.save(filtered_pdf_path)
                        filtered_doc.close()
                        doc.close()
                        
                        print(f"[PageFilter] Kept {len(filtered_pages)}/{total_pages} pages (removed {page_filter_stats['total_removed']} from head/tail)")
                    else:
                        print(f"[PageFilter] All {total_pages} pages kept (no filtering needed)")
            
            # ====== STEP 2: WORD COUNT & SCANNED CHECK (ALWAYS VIA PYMUPDF) ======
            # We always check word count to skip scanned/empty docs, regardless of extractor
            import fitz
            doc = fitz.open(filtered_pdf_path)
            full_text_for_check = ""
            for page in doc:
                full_text_for_check += page.get_text()
            doc.close()
            
            word_count = len(full_text_for_check.split())
            if word_count <= 70:
                print(f"⚠ Document has only {word_count} words (limit: >70). Skipping extraction (likely scanned or empty).")
                raise ValueError(f"Document too small or likely scanned ({word_count} words). Minimum 70 words required.")

            # ====== STEP 3: EXTRACTION ======
            print(f"Extracting text from filtered PDF using {extractor_name}...")
            
            # Get text content based on extractor
            if extractor.name == "PDF-Direct":
                # For PDF-Direct, the text is actually the PDF bytes (handled by provider)
                # But our current provider setup expects text or PDF bytes.
                # Since NIM doesn't support direct PDF, we fall back to full text for now
                # to keep the pipeline alive, or raise error.
                text = full_text_for_check
            elif hasattr(extractor, 'extract_text'):
                text = extractor.extract_text(filtered_pdf_path)
            else:
                text = full_text_for_check

            # ====== STEP 4: LLM EXTRACTION ======
            print(f"Extracting structured data using {llm_provider} ({llm_model or 'default'})...")
            provider = get_provider(llm_provider, llm_model, llm_api_key)
            
            extracted, token_usage = provider.extract_structured_data(
                text=text,
                schema=extraction_model
            )
            
            # ====== STEP 5: CLEANUP ======
            # Remove temp filtered PDF if created
            if filtered_pdf_path != pdf_path and filtered_pdf_path.exists():
                try:
                    filtered_pdf_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temp file {filtered_pdf_path}: {e}")
            
            extraction_time = time.time() - start_time
            
            # Build result with optional page filter stats
            result_data = {
                "pdf_file": str(pdf_path),
                "extractor_name": extractor_name,
                "llm_provider": llm_provider,
                "llm_model": llm_model or provider.model,
                "extraction_time": extraction_time,
                "success": True,
                "extracted_data": extracted.model_dump(),
                "timestamp": datetime.now().isoformat(),
                "input_tokens": token_usage.get('input', 0),
                "output_tokens": token_usage.get('output', 0),
                "total_tokens": token_usage.get('total', 0)
            }
            
            if page_filter_stats:
                result_data["page_filter_stats"] = page_filter_stats
            
            result = ExtractionResult(**result_data)
            
            print(f"✓ Extraction successful in {extraction_time:.2f}s | Tokens: {token_usage.get('input', 0)} in, {token_usage.get('output', 0)} out, {token_usage.get('total', 0)} total")
            return result
            
        except Exception as e:
            extraction_time = time.time() - start_time
            print(f"✗ Extraction failed: {e}")
            
            return ExtractionResult(
                pdf_file=str(pdf_path),
                extractor_name=extractor_name,
                llm_provider=llm_provider,
                llm_model=llm_model or "unknown",
                extraction_time=extraction_time,
                success=False,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def run_test_suite(
        self,
        pdf_files: List[Path],
        extractors: Optional[List[str]] = None,
        llm_configs: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, List[ExtractionResult]]:
        """Run complete test suite
        
        Args:
            pdf_files: List of PDF files to test
            extractors: List of extractor names (None = all available)
            llm_configs: List of LLM configurations
            
        Returns:
            Dictionary mapping PDF files to their results
        """
        if extractors is None:
            extractors = [e.name for e in get_all_extractors()]
        
        if llm_configs is None:
            llm_configs = [
                {"provider": "openai", "model": "gpt-4o"},
            ]
        
        all_results = {}
        
        for pdf_file in pdf_files:
            print(f"\n{'='*60}")
            print(f"Testing: {pdf_file.name}")
            print(f"{'='*60}")
            
            results = []
            
            for extractor_name in extractors:
                for llm_config in llm_configs:
                    result = self.run_extraction(
                        pdf_file,
                        extractor_name,
                        llm_config["provider"],
                        llm_config["model"],
                        llm_config.get("api_key")
                    )
                    results.append(result)
                    self.scorer.save_result(result)
            
            all_results[str(pdf_file)] = results
            
            # Generate comparison
            comparison = self.scorer.compare_results(results)
            print(f"\nComparison Summary:")
            print(f"  Successful: {comparison.successful_extractions}/{comparison.total_extractions}")
            print(f"  Avg Time: {comparison.avg_extraction_time:.2f}s")
            if comparison.best_extraction:
                print(f"  Best: {comparison.best_extraction.extractor_name} + "
                      f"{comparison.best_extraction.llm_provider}")
        
        return all_results


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
