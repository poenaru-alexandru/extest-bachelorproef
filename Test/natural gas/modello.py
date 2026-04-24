from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List, ClassVar
from datetime import datetime


class PeriodoConsumoGas(BaseModel):
    """Singolo periodo di consumo per un PDR estratto dalla bolletta del gas."""
    codice: str = Field(
        ...,
        title="Codice PDR",
        description="Codice identificativo del Punto Di Riconsegna (PDR). Solitamente di 14 cifre."
    )
    consumo: float = Field(
        ...,
        title="Consumo (Smc)",
        description="Consumo totale in Smc. FORMATO: Il documento usa la virgola per i decimali e il punto per le migliaia. Rimuovi il punto delle migliaia e usa il punto decimale."
    )
    indirizzo: str = Field(
        ...,
        title="Indirizzo PDR",
        description="Indirizzo fisico della fornitura gas."
    )
    giorno_fine: str = Field(
        ...,
        title="Data Fine Periodo",
        description="Data fine in formato ISO yyyy-mm-dd"
    )
    giorno_inizio: str = Field(
        ...,
        title="Data Inizio Periodo",
        description="Data inizio in formato ISO yyyy-mm-dd"
    )
    costo_periodo: Optional[float] = Field(
        None,
        title="Costo Periodo (€)",
        description="Importo totale fatturato per questo periodo. FORMATO: Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )

    @field_validator('giorno_inizio', 'giorno_fine')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not v: return v
        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
            try:
                dt = datetime.strptime(v, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return v


class DatiBollettaGas(BaseModel):
    """Contenitore per l'estrazione di tutti i periodi di consumo gas (correnti e storici)."""
    
    consumi: List[PeriodoConsumoGas] = Field(
        ...,
        description=(
            "Elenco di tutti i periodi di consumo gas estratti (correnti e storici). "
            "Cerca ogni riga con consumo Smc e date, inclusa la tabella dello storico (solitamente 12-13 righe). "
            "CRITICO: Devi estrarre TUTTI gli elementi presenti nel documento. Devi restituire un array con MOLTEPLICI oggetti. "
            "L'array non deve MAI contenere un solo elemento se sono presenti più voci (come tabelle storiche o liste) nel testo."
        )
    )
