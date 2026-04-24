from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, ClassVar
from datetime import datetime

class FuelInvoice(BaseModel):
    """
    Rappresenta una singola operazione di rifornimento o una riga di una fattura carburante.
    """
    codice: str = Field(
        ...,
        title="Identificativo/Targa",
        description="Numero fattura, ID transazione o targa del veicolo."
    )
    consumo: float = Field(
        ...,
        title="Quantità Carburante",
        description="Quantità di carburante rifornita. FORMATO: Il documento usa la virgola per i decimali e il punto per le migliaia. Rimuovi il punto delle migliaia e usa il punto decimale (es. 1.234,56 -> 1234.56)."
    )
    tipologia: str = Field(
        ...,
        title="Tipo Carburante",
        description="Tipo di prodotto (es. GASOLIO, BENZINA)."
    )
    giorno_inizio: str = Field(
        ...,
        title="Data Transazione",
        description="Data del rifornimento nel formato ISO yyyy-mm-dd."
    )

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
    """Contenitore per l'estrazione di tutte le transazioni di carburante trovate nel documento."""
    
    fatture: List[FuelInvoice] = Field(
        ...,
        description=(
            "Elenco completo di tutte le transazioni di carburante individuate. "
            "Analizza il documento ed estrai OGNI SINGOLA RIGA di transazione valida trovata nelle tabelle. "
            "CRITICO: Devi estrarre TUTTI gli elementi presenti nel documento. Devi restituire un array con MOLTEPLICI oggetti. "
            "L'array non deve MAI contenere un solo elemento se sono presenti più voci (come tabelle o liste transazioni) nel testo."
        )
    )
