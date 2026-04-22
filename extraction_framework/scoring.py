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
    """Single extraction result"""
    pdf_file: str
    extractor_name: str
    llm_provider: str
    llm_model: str
    extraction_time: float
    success: bool
    error: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    timestamp: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    page_filter_stats: Optional[Dict[str, Any]] = None
    
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
    
    # Fields to ignore when calculating best extraction score
    # These fields are typically metadata or less important for comparison
    IGNORED_FIELDS = {
        'timestamp',
        'indirizzo',  # Often estimated/approximate
        # Add more fields here as needed
    }
    
    def __init__(self, results_dir: Path, ignored_fields: Optional[set] = None, unique_identifiers: Optional[List[str]] = None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Allow custom ignored fields, but default to class-level set
        self.ignored_fields = ignored_fields if ignored_fields is not None else self.IGNORED_FIELDS.copy()
        # Allow custom unique identifiers (default set in _get_unique_identifiers)
        self.custom_unique_identifiers = unique_identifiers
    
    def save_result(self, result: ExtractionResult) -> Path:
        """Save extraction result to file
        
        Args:
            result: Extraction result to save
            
        Returns:
            Path to saved file
        """
        # Create filename from parameters including model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = Path(result.pdf_file).stem
        # Sanitize model name for filename (remove slashes and special chars)
        model_safe = result.llm_model.replace('/', '_').replace('\\', '_').replace(':', '_') if result.llm_model else 'unknown'
        filename = f"{pdf_name}_{result.extractor_name}_{result.llm_provider}_{model_safe}_{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_results_for_pdf(self, pdf_file: str) -> List[ExtractionResult]:
        """Load all results for a specific PDF
        
        Args:
            pdf_file: Path to PDF file
            
        Returns:
            List of extraction results
        """
        pdf_name = Path(pdf_file).stem
        results = []
        
        for result_file in self.results_dir.glob(f"{pdf_name}_*.json"):
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(ExtractionResult(**data))
        
        return results
    
    def compare_results(
        self, 
        results: List[ExtractionResult],
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """Compare multiple extraction results against each other or ground truth
        
        Args:
            results: List of extraction results to compare
            ground_truth: Optional ground truth data for validation
            
        Returns:
            Comparison result with scores
        """
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
                    confidence=1.0 # Ground truth is 100% confident
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
        
        # Find best extraction using new scoring system
        best_result = None
        best_score = float('-inf')
        
        all_extracted_data = [r.extracted_data for r in successful if r.extracted_data]
        
        for result in successful:
            if not result.extracted_data:
                continue
            
            score = self._calculate_result_score(
                result.extracted_data, 
                field_scores,
                all_extracted_data,
                is_ground_truth=bool(ground_truth)
            )
            
            print(f"[Scoring] {result.extractor_name} + {result.llm_provider}/{result.llm_model}: score = {score:.2f}")
            
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
    
    def _calculate_result_score(
        self, 
        extracted_data: Dict, 
        field_scores: List[FieldScore],
        all_results: Optional[List[Dict]] = None,
        is_ground_truth: bool = False
    ) -> float:
        """Calculate score for an extraction result
        
        If is_ground_truth is True, scores match against the ground truth.
        """
        # Flatten this result's data
        this_fields = defaultdict(list)
        self._flatten_dict(extracted_data, "", this_fields)
        
        score = 0.0
        
        # Build baseline map for quick lookup
        baseline_map = {fs.field_name: fs for fs in field_scores}
        
        # Get unique identifier field names
        unique_identifiers = self._get_unique_identifiers(extracted_data)
        
        # Group fields by record
        record_groups = self._group_by_record(this_fields, unique_identifiers)
        baseline_groups = self._group_by_record_from_consensus(field_scores, unique_identifiers)
        
        for record_key, record_fields in record_groups.items():
            baseline_for_record = baseline_groups.get(record_key, {})
            
            for field_name, field_value in record_fields.items():
                if any(ignored in field_name for ignored in self.ignored_fields):
                    continue
                if any(uid in field_name for uid in unique_identifiers):
                    continue
                
                baseline_field = baseline_for_record.get(field_name) or baseline_map.get(field_name)
                
                if baseline_field is None:
                    # Extra field extracted
                    score += 0.5 if not is_ground_truth else -1.0 # Penalty if GT doesn't have it
                    continue
                
                this_value_str = json.dumps(field_value, sort_keys=True) if not isinstance(field_value, (str, float, int)) else field_value
                baseline_value_str = json.dumps(baseline_field.value, sort_keys=True) if not isinstance(baseline_field.value, (str, float, int)) else baseline_field.value
                
                if str(this_value_str).lower() == str(baseline_value_str).lower():
                    # Match!
                    if is_ground_truth:
                        score += 10.0 # High reward for matching Ground Truth
                    else:
                        score += 1.0 if baseline_field.confidence < 1.0 else 0.1
                else:
                    # Mismatch
                    score -= 5.0 if is_ground_truth else 1.0
        
        return score
    
    def _flatten_dict(self, d: Dict, prefix: str, result: Dict[str, List]):
        """Flatten nested dictionary for comparison"""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            # Skip ignored fields
            if key in self.ignored_fields or full_key in self.ignored_fields:
                continue
            
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
                key_name, idx = key.split('[')
                idx = int(idx.rstrip(']'))
                
                if key_name not in current:
                    current[key_name] = []
                while len(current[key_name]) <= idx:
                    current[key_name].append({})
                current = current[key_name][idx]
            else:
                if key not in current:
                    current[key] = {}
                current = current[key]
        
        current[keys[-1]] = value
    
    def _get_unique_identifiers(self, extracted_data: Dict) -> List[str]:
        """Get unique identifier field names"""
        if self.custom_unique_identifiers:
            return self.custom_unique_identifiers
        
        # Default fallback
        if 'consumi' in extracted_data and extracted_data['consumi']:
            return ['codice', 'giorno_inizio', 'giorno_fine']
        
        return []
    
    def _group_by_record(self, fields: Dict[str, List], unique_identifiers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Group flattened fields by record using unique identifiers"""
        if not unique_identifiers:
            return {"all": {k: v[0] if v else None for k, v in fields.items()}}
        
        records = defaultdict(dict)
        
        for field_name, values in fields.items():
            if '[' in field_name:
                parts = field_name.split('[')
                if len(parts) > 1:
                    idx_str = parts[1].split(']')[0]
                    record_idx = int(idx_str)
                    
                    # Try to find list name (e.g. 'consumi' or 'fatture' or 'rifiuti')
                    list_name = parts[0]
                    
                    record_key_parts = []
                    for uid in unique_identifiers:
                        uid_field = f"{list_name}[{record_idx}].{uid}"
                        if uid_field in fields and fields[uid_field]:
                            record_key_parts.append(str(fields[uid_field][0]))
                    
                    record_key = "|".join(record_key_parts) if record_key_parts else f"record_{record_idx}"
                    records[record_key][field_name] = values[0] if values else None
            else:
                records["all"][field_name] = values[0] if values else None
        
        return records
    
    def _group_by_record_from_consensus(self, field_scores: List[FieldScore], unique_identifiers: List[str]) -> Dict[str, Dict[str, FieldScore]]:
        """Group baseline field scores by record"""
        if not unique_identifiers:
            return {"all": {fs.field_name: fs for fs in field_scores}}
        
        records = defaultdict(dict)
        
        for field_score in field_scores:
            field_name = field_score.field_name
            
            if '[' in field_name:
                parts = field_name.split('[')
                if len(parts) > 1:
                    idx_str = parts[1].split(']')[0]
                    record_idx = int(idx_str)
                    list_name = parts[0]
                    
                    record_key_parts = []
                    for uid in unique_identifiers:
                        uid_field = f"{list_name}[{record_idx}].{uid}"
                        uid_score = next((fs for fs in field_scores if fs.field_name == uid_field), None)
                        if uid_score:
                            record_key_parts.append(str(uid_score.value))
                    
                    record_key = "|".join(record_key_parts) if record_key_parts else f"record_{record_idx}"
                    records[record_key][field_name] = field_score
            else:
                records["all"][field_name] = field_score
        
        return records
