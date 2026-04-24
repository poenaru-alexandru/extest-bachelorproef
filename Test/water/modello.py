from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, ClassVar
from datetime import datetime


class PeriodoConsumoAcqua(BaseModel):
    codice: str = Field(
        ...,
        title="Codice Contatore/Contratto",
        description="Numero di matricola del contatore dell'acqua, o codice contratto se non disponibile"
    )
    consumo: float = Field(
        ...,
        title="Consumo Totale (m³)",
        description="Consumo totale in m³. FORMATO: Il documento usa la virgola per i decimali e il punto per le migliaia. Rimuovi il punto delle migliaia e usa il punto decimale."
    )
    indirizzo: str = Field(
        ...,
        title="Indirizzo Punto di Prelievo",
        description="Indirizzo associato al punto di prelievo dell'acqua"
    )
    giorno_fine: str = Field(
        ...,
        title="Data Fine Periodo",
        description="Data di fine del periodo di riferimento, nel formato ISO yyyy-mm-dd"
    )
    consumo_medio: float = Field(
        ...,
        title="Consumo Medio Giornaliero (m³/giorno)",
        description="Consumo medio giornaliero in m³/giorno. FORMATO: Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )
    costo_periodo: Optional[float] = Field(
        None,
        title="Costo Periodo (€)",
        description="Costo del periodo in Euro. FORMATO: Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )
    giorno_inizio: str = Field(
        ...,
        title="Data Inizio Periodo",
        description="Data di inizio del periodo di riferimento, nel formato ISO yyyy-mm-dd"
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


class DatiBollettaAcqua(BaseModel):
    """Contenitore per l'estrazione di tutti i periodi di consumo idrico (correnti e storici)."""
    
    consumi: List[PeriodoConsumoAcqua] = Field(
        ...,
        title="Elenco Periodi di Consumo",
        description=(
            "Lista dei periodi di consumo idrico presenti nella bolletta (correnti e storici). "
            "Cerca di estrarre TUTTI i periodi identificabili nel documento. "
            "CRITICO: Devi estrarre TUTTI gli elementi presenti nel documento. Devi restituire un array con MOLTEPLICI oggetti. "
            "L'array non deve MAI contenere un solo elemento se sono presenti più voci (come tabelle storiche o liste) nel testo."
        )
    )
