from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict

class WasteItem(BaseModel):
    anno: Optional[int] = Field(None, description="Reference year")
    codice_cer: Optional[str] = Field(None, description="CER code of the waste")
    quantita: Optional[float] = Field(None, description="Quantity in kg or tons")
    tipo: Optional[str] = Field(None, description="Waste description")
    codice_smaltimento: Optional[str] = Field(None, description="Recovery/disposal operation code (e.g., R13, D15)")

class WasteData(BaseModel):
    """Schema for extracting data from waste records or forms"""
    rifiuti: List[WasteItem]
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Waste Record or Form",
            "patterns": [r"CER", r"Rifiuto", r"Smaltimento"]
        }
    ]
