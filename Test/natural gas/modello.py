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
        description="Consumo totale espresso in Standard Metri Cubi (Smc)."
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
    consumo_annuale: Optional[float] = Field(
        None,
        title="Consumo Annuale (Smc)",
        description="Riepilogo ultimi 12 mesi. Valorizzare SOLO per il periodo principale della bolletta corrente."
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
    """
    ESTRAI TUTTI I PERIODI DI CONSUMO GAS DALLA BOLLETTA FORNITA.
    
    In una bolletta del gas è fondamentale trovare sia il periodo correntemente fatturato che la tabella dello storico dei consumi (solitamente 12-13 righe con Smc e date).
    
    ISTRUZIONI:
    1. Identifica il PDR (Punto di Riconsegna).
    2. Cerca nel documento ogni riga che riporta un consumo in Smc e un intervallo di date.
    3. Estrai OGNI occorrenza trovata nel campo 'consumi'.
    4. consumo_annuale: inseriscilo SOLO per il record della bolletta corrente. Per lo storico deve essere null.
    5. Se mancano date o consumo, scarta la riga.
    """
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Pagina con PDR e consumi gas",
            "patterns": [r"(?i)(PDR|GAS|SMC|METANO|CONSUMO)"]
        }
    ]
    
    consumi: List[PeriodoConsumoGas] = Field(
        ...,
        description="Elenco di tutti i periodi di consumo gas estratti (correnti e storici)."
    )
