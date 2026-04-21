from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict

class FuelInvoice(BaseModel):
    codice: Optional[str] = Field(None, description="Invoice or transaction identification code")
    tipologia: Optional[str] = Field(None, description="Fuel type (e.g., DIESEL, UNLEADED 95)")
    quantita: Optional[float] = Field(None, description="Quantity delivered")
    um: Optional[str] = Field(None, description="Unit of measurement (e.g., L)")
    prezzo: Optional[float] = Field(None, description="Total price in Euro")
    giorno_inizio: Optional[str] = Field(None, description="Date (YYYY-MM-DD)")
    energia_unitaria: Optional[float] = None
    energia_fonte: Optional[str] = None
    carbonfootprint_unitaria: Optional[float] = None
    carbonfootprint_fonte: Optional[str] = None

class FuelData(BaseModel):
    """Schema for extracting data from fuel invoices"""
    fatture: List[FuelInvoice]
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Fuel invoice",
            "patterns": [r"GASOLIO", r"EURO 95", r"LITRI"]
        }
    ]
