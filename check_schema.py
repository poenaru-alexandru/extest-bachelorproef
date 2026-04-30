from typing import List, Optional
from pydantic import BaseModel, Field
import json

class Periode(BaseModel):
    supplier: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    kwh_quantity: Optional[float] = None

class BachelorProefModel(BaseModel):
    """Docstring here"""
    periodes: List[Periode] = Field(..., min_length=1)

print(json.dumps(BachelorProefModel.model_json_schema(), indent=2))
