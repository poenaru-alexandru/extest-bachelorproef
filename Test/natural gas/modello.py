from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict

class GasConsumption(BaseModel):
    codice: Optional[str] = Field(None, description="PDR or identification code of the supply")
    consumo: Optional[float] = Field(None, description="Consumption in Smc")
    indirizzo: Optional[str] = Field(None, description="Supply address")
    giorno_inizio: Optional[str] = Field(None, description="Period start date (YYYY-MM-DD)")
    giorno_fine: Optional[str] = Field(None, description="Period end date (YYYY-MM-DD)")
    costo_periodo: Optional[float] = Field(None, description="Period cost in Euro")

class NaturalGasData(BaseModel):
    """Schema for extracting data from gas bills"""
    consumi: List[GasConsumption]
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Page with gas consumption data and PDR",
            "patterns": [r"PDR", r"Smc", r"Metano"]
        }
    ]
