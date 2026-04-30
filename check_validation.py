from typing import List, Optional
from pydantic import BaseModel
import json

class Periode(BaseModel):
    supplier: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    kwh_quantity: Optional[float] = None

class BachelorProefModel(BaseModel):
    """Docstring here"""
    periodes: List[Periode] = []

# Simulate LLM returning flat data instead of nested in 'periodes'
llm_data = {
    "supplier": "Dolomiti",
    "start_date": "2025-12-01",
    "end_date": "2025-12-31",
    "kwh_quantity": 628.0
}

try:
    model = BachelorProefModel.model_validate(llm_data)
    print(f"Model validated: {model}")
    print(f"Periodes: {model.periodes}")
except Exception as e:
    print(f"Validation failed: {e}")
