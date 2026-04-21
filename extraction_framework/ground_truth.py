"""Ground truth management system"""
from pathlib import Path
from typing import Optional, Dict, Any
import json
from pydantic import BaseModel
from datetime import datetime


class GroundTruthManager:
    """Manage ground truth data for validation"""
    
    def __init__(self, ground_truth_dir: Path):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    def save_ground_truth(
        self, 
        pdf_file: str, 
        data: Dict[str, Any],
        validated_by: str = "user",
        notes: str = "",
        save_to_test_folder: bool = True
    ) -> Path:
        """Save ground truth data
        
        Args:
            pdf_file: Path to PDF file
            data: Ground truth data
            validated_by: Who validated this data
            notes: Optional notes
            save_to_test_folder: If True, also save JSON to Test folder alongside PDF
            
        Returns:
            Path to saved ground truth file
        """
        pdf_path = Path(pdf_file)
        pdf_name = pdf_path.stem
        
        ground_truth = {
            "pdf_file": str(pdf_file),
            "data": data,
            "validated_by": validated_by,
            "validated_at": datetime.now().isoformat(),
            "notes": notes
        }
        
        # Save to ground_truth directory
        filepath = self.ground_truth_dir / f"{pdf_name}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        
        # Also save just the data to Test folder alongside PDF
        if save_to_test_folder and pdf_path.exists():
            test_json_path = pdf_path.parent / f"{pdf_name}.json"
            with open(test_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved validated data to: {test_json_path}")
        
        return filepath
    
    def load_ground_truth(self, pdf_file: str) -> Optional[Dict[str, Any]]:
        """Load ground truth data from the same directory as the PDF

        Args:
            pdf_file: Path to PDF file

        Returns:
            Ground truth data or None if not found
        """
        pdf_path = Path(pdf_file)
        # Look for .json file with the same name next to the PDF
        json_path = pdf_path.with_suffix('.json')

        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # If the JSON is just the data (user defined), wrap it in the expected format
                if "data" not in data and isinstance(data, dict):
                    return {
                        "pdf_file": str(pdf_file),
                        "data": data,
                        "validated_by": "manual",
                        "validated_at": datetime.now().isoformat()
                    }
                return data
        except Exception as e:
            print(f"Error loading ground truth {json_path}: {e}")
            return None

    def has_ground_truth(self, pdf_file: str) -> bool:
        """Check if ground truth exists next to the PDF"""
        pdf_path = Path(pdf_file)
        return pdf_path.with_suffix('.json').exists()
    def validate_extraction(
        self, 
        extracted_data: Dict[str, Any],
        pdf_file: str
    ) -> Dict[str, Any]:
        """Validate extraction against ground truth
        
        Args:
            extracted_data: Extracted data to validate
            pdf_file: Path to PDF file
            
        Returns:
            Validation report
        """
        ground_truth = self.load_ground_truth(pdf_file)
        
        if not ground_truth:
            return {
                "has_ground_truth": False,
                "message": "No ground truth available"
            }
        
        gt_data = ground_truth["data"]
        
        # Compare data
        report = {
            "has_ground_truth": True,
            "matches": [],
            "mismatches": [],
            "missing": [],
            "extra": []
        }
        
        # Flatten both dicts for comparison
        extracted_flat = self._flatten_dict(extracted_data)
        gt_flat = self._flatten_dict(gt_data)
        
        # Compare fields
        all_keys = set(extracted_flat.keys()) | set(gt_flat.keys())
        
        for key in all_keys:
            if key in extracted_flat and key in gt_flat:
                if extracted_flat[key] == gt_flat[key]:
                    report["matches"].append({
                        "field": key,
                        "value": extracted_flat[key]
                    })
                else:
                    report["mismatches"].append({
                        "field": key,
                        "extracted": extracted_flat[key],
                        "ground_truth": gt_flat[key]
                    })
            elif key in gt_flat:
                report["missing"].append({
                    "field": key,
                    "expected": gt_flat[key]
                })
            else:
                report["extra"].append({
                    "field": key,
                    "value": extracted_flat[key]
                })
        
        # Calculate accuracy
        total = len(all_keys)
        correct = len(report["matches"])
        report["accuracy"] = correct / total if total > 0 else 0
        
        return report
    
    def _flatten_dict(
        self, 
        d: Dict, 
        parent_key: str = '', 
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(item, f"{new_key}[{i}]", sep=sep).items()
                        )
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def list_ground_truths(self) -> list[Dict[str, Any]]:
        """List all available ground truths
        
        Returns:
            List of ground truth metadata
        """
        ground_truths = []
        
        for gt_file in self.ground_truth_dir.glob("*.json"):
            with open(gt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ground_truths.append({
                    "pdf_file": data["pdf_file"],
                    "validated_by": data.get("validated_by", "unknown"),
                    "validated_at": data.get("validated_at", "unknown"),
                    "notes": data.get("notes", "")
                })
        
        return ground_truths
