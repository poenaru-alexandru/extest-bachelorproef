from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict

class ElectricityConsumption(BaseModel):
    codice: Optional[str] = Field(None, description="POD or identification code of the supply")
    consumo: Optional[float] = Field(None, description="Total consumption in kWh")
    indirizzo: Optional[str] = Field(None, description="Supply address")
    consumo_f1: Optional[float] = Field(None, description="Consumption in F1 rate")
    consumo_f2: Optional[float] = Field(None, description="Consumption in F2 rate")
    consumo_f3: Optional[float] = Field(None, description="Consumption in F3 rate")
    giorno_inizio: Optional[str] = Field(None, description="Period start date (YYYY-MM-DD)")
    giorno_fine: Optional[str] = Field(None, description="Period end date (YYYY-MM-DD)")
    costo_periodo: Optional[float] = Field(None, description="Period cost in Euro")

class ElectricityData(BaseModel):
    """Schema for extracting data from electricity bills"""
    consumi: List[ElectricityConsumption]
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Page with consumption data and POD",
            "patterns": [r"POD", r"IT\d{3}E", r"kWh", r"F1"]
        }
    ]
