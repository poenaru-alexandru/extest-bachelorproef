from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict


class WasteItem(BaseModel):
    """Singola riga di un registro di carico/scarico rifiuti o formulario"""
    anno: Optional[int] = Field(
        None,
        title="Anno di Riferimento",
        description="L'anno a cui si riferisce l'operazione."
    )
    tipo: Optional[str] = Field(
        None,
        title="Descrizione Rifiuto",
        description="Descrizione testuale della tipologia di rifiuto."
    )
    quantita: float = Field(
        ...,
        title="Quantità (kg/t)",
        description="Peso del rifiuto."
    )
    codice_cer: str = Field(
        ...,
        title="Codice CER",
        description="Codice del Catalogo Europeo dei Rifiuti (6 cifre)."
    )
    codice_smaltimento: Optional[str] = Field(
        None,
        title="Codice Recupero/Smaltimento",
        description="Codice dell'operazione (es. R13, D15)."
    )


class DatiRifiuti(BaseModel):
    """
    ESTRAI TUTTI I MOVIMENTI DI RIFIUTI DAL DOCUMENTO.
    
    Analizza il documento ed estrai OGNI RIGA di registro o formulario trovata.
    """
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Pagina con dati CER e movimenti rifiuti",
            "patterns": [
                r"(?i)(CER|CODICE\s+RIFIUTO|RIFIUTI)",
                r"(?i)(KG|TON|QUANTITÀ|PESO)"
            ]
        }
    ]
    
    rifiuti: List[WasteItem] = Field(
        ...,
        description="Lista di tutte le righe di registro o movimenti di rifiuti estratti."
    )
