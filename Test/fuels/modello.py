from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, ClassVar
from datetime import datetime

class FuelInvoice(BaseModel):
    """
    Rappresenta una singola operazione di rifornimento o una riga di una fattura carburante.
    """
    um: str = Field(
        "L",
        title="Unità di Misura",
        description="Unità di misura della quantità (es. L, KG)."
    )
    codice: str = Field(
        ...,
        title="Identificativo/Targa",
        description="Numero fattura, ID transazione o targa del veicolo."
    )
    prezzo: float = Field(
        ...,
        title="Importo Totale (€)",
        description="Costo totale della transazione comprensivo di IVA."
    )
    quantita: float = Field(
        ...,
        title="Quantità",
        description="Volume o peso erogato."
    )
    tipologia: str = Field(
        ...,
        title="Tipo Carburante",
        description="Tipo di prodotto (es. GASOLIO, BENZINA)."
    )
    energia_fonte: Optional[str] = None
    giorno_inizio: str = Field(
        ...,
        title="Data Transazione",
        description="Data del rifornimento nel formato ISO yyyy-mm-dd."
    )
    energia_unitaria: Optional[float] = None
    carbonfootprint_fonte: Optional[str] = None
    carbonfootprint_unitaria: Optional[float] = None

    @field_validator('giorno_inizio')
    @classmethod
    def validate_date(cls, v: str) -> str:
        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
            try:
                return datetime.strptime(v, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return v

class DatiCarburante(BaseModel):
    """
    ESTRAI TUTTE LE TRANSAZIONI DI CARBURANTE DAL DOCUMENTO.
    
    Analizza il documento ed estrai OGNI SINGOLA RIGA di transazione valida trovata nelle tabelle.
    """
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Pagina con dettagli rifornimenti",
            "patterns": [
                r"(?i)(GASOLIO|BENZINA|CARBURANTE|RIFORNIMENTO)",
                r"(?i)(LITRI|QUANTITÀ|PREZZO)"
            ]
        }
    ]
    
    fatture: List[FuelInvoice] = Field(
        ...,
        description="Elenco completo di tutte le transazioni di carburante individuate."
    )
