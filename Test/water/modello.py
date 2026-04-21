from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict

class WaterConsumption(BaseModel):
    codice: Optional[str] = Field(None, description="Service code or meter serial number")
    consumo: Optional[float] = Field(None, description="Consumption in cubic meters (mc)")
    indirizzo: Optional[str] = Field(None, description="Supply address")
    giorno_inizio: Optional[str] = Field(None, description="Period start date (YYYY-MM-DD)")
    giorno_fine: Optional[str] = Field(None, description="Period end date (YYYY-MM-DD)")
    consumo_medio: Optional[float] = Field(None, description="Average daily consumption")
    costo_periodo: Optional[float] = Field(None, description="Period cost in Euro")

class WaterData(BaseModel):
    """Schema for extracting data from water bills"""
    consumi: List[WaterConsumption]
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Water bill",
            "patterns": [r"Servizio Idrico", r"contatore", r"mc"]
        }
    ]
