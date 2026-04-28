"""Scoring and comparison system for extraction results"""
from typing import Dict, List, Any, Optional
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
    extraction_time: float  # Total wall-clock time
    
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
    
    # Sustainability Metrics
    energy_kwh: Optional[float] = None
    co2_kg: Optional[float] = None
    cpu_energy_kwh: Optional[float] = None
    gpu_energy_kwh: Optional[float] = None
    ram_energy_kwh: Optional[float] = None
    energy_source: Optional[str] = None  # 'codecarbon' or 'ecologits'


class ComparisonResult(BaseModel):
    """Comparison of multiple extraction results"""
    pdf_file: str
    total_extractions: int
    successful_extractions: int
    field_scores: List[FieldScore]
    best_extraction: Optional[ExtractionResult] = None
    consensus_data: Optional[Dict[str, Any]] = None
    avg_extraction_time: float


class ResultScorer:
    """Score and compare extraction results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, result: ExtractionResult) -> Path:
        """Save extraction result to file"""
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
        else:
            model_short = model_lower.replace('/', '_').replace('\\', '_').replace(':', '_')
            
        provider_type = 'local' if result.llm_provider.lower() == 'local' else 'cloud'
        filename = f"{provider_type}-{model_short}-{pdf_name}-{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_results_for_pdf(self, pdf_file: str) -> List[ExtractionResult]:
        """Load all results for a specific PDF"""
        pdf_name = Path(pdf_file).stem
        results = []
        
        for result_file in self.results_dir.glob(f"*-*-{pdf_name}-*.json"):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(ExtractionResult(**data))
        
        return results
    
    def compare_results(
        self, 
        results: List[ExtractionResult],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """Compare multiple extraction results against each other or ground truth"""
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
                avg_extraction_time=sum(r.extraction_time for r in results) / len(results)
            )
        
        field_scores = []
        consensus_data = {}
        gt_data = None

        if ground_truth:
            # Use Ground Truth as the baseline for comparison
            gt_data = ground_truth.get("data", ground_truth)
            gt_fields = defaultdict(list)
            self._flatten_dict(gt_data, "", gt_fields)
            
            for field_name, values in gt_fields.items():
                val = values[0] if values else None
                field_scores.append(FieldScore(
                    field_name=field_name,
                    value=val,
                    agreement_count=1,
                    total_models=1,
                    confidence=1.0 
                ))
            consensus_data = gt_data
        else:
            # Analyze field agreement (consensus among models)
            field_values = defaultdict(list)
            for result in successful:
                self._flatten_dict(result.extracted_data, "", field_values)
            
            for field_name, values in field_values.items():
                value_counts = defaultdict(int)
                for value in values:
                    value_str = json.dumps(value, sort_keys=True)
                    value_counts[value_str] += 1
                
                most_common_str = max(value_counts.items(), key=lambda x: x[1])[0]
                most_common_value = json.loads(most_common_str)
                agreement_count = value_counts[most_common_str]
                
                field_scores.append(FieldScore(
                    field_name=field_name,
                    value=most_common_value,
                    agreement_count=agreement_count,
                    total_models=len(values),
                    confidence=agreement_count / len(values)
                ))
            
            # Build consensus data
            for score in field_scores:
                if score.confidence >= 0.5:
                    self._set_nested_value(consensus_data, score.field_name, score.value)
        
        # Find best extraction using similarity scoring
        best_result = None
        best_score = float('-inf')
        
        reference_data = gt_data if gt_data else consensus_data
        
        for result in successful:
            if not result.extracted_data:
                continue
            
            # Calculate similarity score (0 to 100)
            similarity = self.calculate_similarity(result.extracted_data, reference_data)
            score = similarity * 100.0
            
            print(f"[Scoring] {result.extractor_name} + {result.llm_provider}/{result.llm_model}: similarity = {similarity:.4f}, score = {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return ComparisonResult(
            pdf_file=pdf_file,
            total_extractions=len(results),
            successful_extractions=len(successful),
            field_scores=field_scores,
            best_extraction=best_result,
            consensus_data=consensus_data,
            avg_extraction_time=sum(r.extraction_time for r in successful) / len(successful)
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
