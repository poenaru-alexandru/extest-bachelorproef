"""Scoring and comparison system for extraction results"""
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel
import json
from collections import defaultdict
from datetime import datetime


class FieldScore(BaseModel):
    """Score for a single field"""
    field_name: str
    value: Any
    agreement_count: int  # How many models agreed on this value
    total_models: int
    confidence: float  # agreement_count / total_models


class ExtractionResult(BaseModel):
    """Single extraction result with streaming telemetry"""
    pdf_file: str
    extractor_name: str
    llm_provider: str
    llm_model: str
    # Streaming Telemetry
    ttft_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None
    total_inference_seconds: Optional[float] = None
    
    success: bool
    error: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    timestamp: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Quality Score (rule-based structural validation, 0–100)
    validation_score: Optional[float] = None

    # Sustainability Metrics
    # raw_energy_kwh: bare IT energy before PUE correction
    #   local  → CodeCarbon sum(GPU+CPU+RAM), pue NOT passed to tracker
    #   cloud  → EcoLogits usage.energy (already includes datacenter PUE; raw not separately available)
    # energy_kwh_with_pue: facility-level energy used for fair local↔cloud comparison
    #   local  → raw_energy_kwh × PUE_LOCAL (1.08)
    #   cloud  → same as raw_energy_kwh (EcoLogits already embeds PUE)
    raw_energy_kwh: Optional[float] = None
    energy_kwh_with_pue: Optional[float] = None
    co2_kg: Optional[float] = None
    cpu_energy_kwh: Optional[float] = None
    gpu_energy_kwh: Optional[float] = None
    ram_energy_kwh: Optional[float] = None
    energy_source: Optional[str] = None  # 'codecarbon' or 'ecologits'
    regional_cloud_projections: Optional[Dict[str, float]] = None


class ComparisonResult(BaseModel):
    """Comparison of multiple extraction results"""
    pdf_file: str
    total_extractions: int
    successful_extractions: int
    field_scores: List[FieldScore]
    best_extraction: Optional[ExtractionResult] = None
    consensus_data: Optional[Dict[str, Any]] = None


class ResultScorer:
    """Score and compare extraction results"""

    def __init__(self, results_dir: Path, db=None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.db = db  # optional ResultsDB instance

    def save_result(self, result: ExtractionResult, run_number: Optional[int] = None) -> Path:
        """Save extraction result to JSON file and, if a DB is attached, to SQLite."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = Path(result.pdf_file).stem
        
        # Simplify model name
        model_lower = result.llm_model.lower() if result.llm_model else 'unknown'
        if 'llama' in model_lower:
            model_short = 'llama'
        elif 'mistral' in model_lower:
            model_short = 'mistral'
        elif 'gemma' in model_lower:
            model_short = 'gemma'
        elif 'qwen' in model_lower:
            model_short = 'qwen'
        elif 'apertus' in model_lower:
            model_short = 'apertus'
        else:
            model_short = model_lower.replace('/', '_').replace('\\', '_').replace(':', '_')
            
        provider_type = 'local' if result.llm_provider.lower() == 'local' else 'cloud'
        filename = f"{provider_type}-{model_short}-{pdf_name}-{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

        if self.db is not None:
            session_id = self.results_dir.name  # e.g. "results_20240101_120000"
            self.db.insert(result, session_id, run_number=run_number)

        return filepath
    
    def load_results_for_pdf(self, pdf_file: str) -> List[ExtractionResult]:
        """Load all results for a specific PDF recursively from subdirectories"""
        pdf_name = Path(pdf_file).stem
        results = []
        
        for result_file in self.results_dir.rglob(f"*-*-{pdf_name}-*.json"):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(ExtractionResult(**data))
        
        return results
    
    def validate_and_score(self, data: Dict[str, Any]) -> float:
        """
        Rule-based scoring:
        1. Check for 'periodes' array with at least one item (25 pts)
        2. Check if exactly one item in 'periodes' (15 pts)
        3. Check for 4 required fields in the first item (10 pts each = 40 pts)
        4. Check types of required fields (5 pts each = 20 pts)
        
        Total = 100 points
        """
        score = 0.0
        
        if not isinstance(data, dict):
            return 0.0
            
        periodes = data.get('periodes')
        
        # 1. Check for 'periodes' array with at least one item
        if isinstance(periodes, list) and len(periodes) >= 1:
            score += 25.0
            
            # 2. Check if exactly one item
            if len(periodes) == 1:
                score += 15.0
            
            first_item = periodes[0]
            if isinstance(first_item, dict):
                required_fields = ['supplier', 'start_date', 'end_date', 'kwh_quantity']
                
                # 3. Check for presence of required fields
                for field in required_fields:
                    if field in first_item:
                        score += 10.0
                        
                        # 4. Check types
                        val = first_item[field]
                        if field == 'kwh_quantity':
                            if val is None or isinstance(val, (int, float)):
                                score += 5.0
                        else:
                            if val is None or isinstance(val, str):
                                score += 5.0
        
        return score

    def compare_results(
        self, 
        results: List[ExtractionResult],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """Compare multiple extraction results using rule-based scoring"""
        if not results:
            raise ValueError("No results to compare")
        
        pdf_file = results[0].pdf_file
        successful = [r for r in results if r.success and r.extracted_data]
        
        if not successful:
            return ComparisonResult(
                pdf_file=pdf_file,
                total_extractions=len(results),
                successful_extractions=0,
                field_scores=[],
            )
        
        # Calculate scores for each successful extraction using rule-based logic
        best_result = None
        best_score = float('-inf')
        
        for result in successful:
            score = self.validate_and_score(result.extracted_data)
            
            print(f"[Scoring] {result.extractor_name} + {result.llm_provider}/{result.llm_model}: rule-based score = {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_result = result

        # We still return field_scores for compatibility, but we populate them from the best result
        field_scores = []
        if best_result and best_result.extracted_data:
            field_values = defaultdict(list)
            self._flatten_dict(best_result.extracted_data, "", field_values)
            for field_name, values in field_values.items():
                field_scores.append(FieldScore(
                    field_name=field_name,
                    value=values[0],
                    agreement_count=1,
                    total_models=1,
                    confidence=1.0
                ))

        return ComparisonResult(
            pdf_file=pdf_file,
            total_extractions=len(results),
            successful_extractions=len(successful),
            field_scores=field_scores,
            best_extraction=best_result,
            consensus_data=best_result.extracted_data if best_result else {},
        )
    
    def calculate_similarity(self, extracted: Any, reference: Any) -> float:
        """Calculate a similarity score between 0 and 1.
        Based on row-level correctness for lists and field-level for dicts.
        """
        if extracted == reference:
            return 1.0
        
        # Numeric comparison
        if isinstance(extracted, (int, float)) and isinstance(reference, (int, float)):
            if reference == 0:
                return 1.0 if extracted == 0 else 0.0
            return max(0, 1.0 - abs(extracted - reference) / abs(reference)) if reference != 0 else 0.0

        # String comparison
        if isinstance(extracted, str) and isinstance(reference, str):
            e_clean = extracted.strip().lower()
            r_clean = reference.strip().lower()
            if e_clean == r_clean:
                return 1.0
            return 0.0
            
        # Dictionary comparison
        if isinstance(extracted, dict) and isinstance(reference, dict):
            if not reference:
                return 1.0 if not extracted else 0.5
            
            total_score = 0.0
            relevant_fields = list(reference.keys())
            if not relevant_fields:
                return 1.0
                
            for k in relevant_fields:
                if k in extracted:
                    total_score += self.calculate_similarity(extracted[k], reference[k])
            
            return total_score / len(relevant_fields)
            
        # List comparison (Row-based scoring)
        if isinstance(extracted, list) and isinstance(reference, list):
            if not reference:
                return 1.0 if not extracted else 0.0
            
            # Match each reference item to the best extracted item
            total_score = 0.0
            matched_indices = set()
            
            for ref_item in reference:
                best_item_score = 0.0
                best_idx = -1
                
                for i, ext_item in enumerate(extracted):
                    if i in matched_indices:
                        continue
                    item_score = self.calculate_similarity(ext_item, ref_item)
                    if item_score > best_item_score:
                        best_item_score = item_score
                        best_idx = i
                    if best_item_score == 1.0:
                        break
                
                # A row is "correct" if it has high similarity
                total_score += best_item_score
                if best_idx != -1 and best_item_score > 0.8: 
                    matched_indices.add(best_idx)
            
            # Penalize for extra items in extracted? 
            # If reference has 10 rows and model returns 20, score should reflect that 10 were right but 10 were extra.
            penalty = max(0, len(extracted) - len(reference)) * 0.1 # Small penalty for each extra row
            
            return max(0, (total_score / len(reference)) - (penalty / len(reference)))
            
        return 0.0

    def _calculate_result_score(
        self, 
        extracted_data: Dict, 
        field_scores: List[FieldScore],
        all_results: Optional[List[Dict]] = None,
        is_ground_truth: bool = False
    ) -> float:
        """Legacy support for the API/UI - calculates a score based on field_scores"""
        # Build a temporary reference dict from field_scores
        reference = {}
        for fs in field_scores:
            self._set_nested_value(reference, fs.field_name, fs.value)
            
        similarity = self.calculate_similarity(extracted_data, reference)
        return similarity * 100.0

    def _flatten_dict(self, d: Dict, prefix: str, result: Dict[str, List]):
        """Flatten nested dictionary for comparison"""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_dict(value, full_key, result)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._flatten_dict(item, f"{full_key}[{i}]", result)
                    else:
                        result[f"{full_key}[{i}]"].append(item)
            else:
                result[full_key].append(value)
    
    def _set_nested_value(self, d: Dict, key_path: str, value: Any):
        """Set value in nested dictionary using dot notation"""
        keys = key_path.split('.')
        current = d
        
        for key in keys[:-1]:
            if '[' in key:
                # Handle list indices
                key_name, rest = key.split('[')
                idx = int(rest.rstrip(']'))
                
                if key_name not in current:
                    current[key_name] = []
                while len(current[key_name]) <= idx:
                    current[key_name].append({})
                current = current[key_name][idx]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
        
        last_key = keys[-1]
        if '[' in last_key:
             key_name, rest = last_key.split('[')
             idx = int(rest.rstrip(']'))
             if key_name not in current:
                 current[key_name] = []
             while len(current[key_name]) <= idx:
                 current[key_name].append(None)
             current[key_name][idx] = value
        else:
            current[last_key] = value
