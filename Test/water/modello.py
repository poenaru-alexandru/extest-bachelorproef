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
        description="Consumo totale di acqua durante il periodo, espresso in metri cubi"
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
        description="Consumo medio giornaliero nel periodo, in metri cubi per giorno"
    )
    costo_periodo: Optional[float] = Field(
        None,
        title="Costo Periodo (€)",
        description="Costo del periodo in Euro"
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
    """
    Estrai informazioni su ciascun periodo di consumo dalla bolletta dell'acqua in formato testuale.
    Cerca di estrarre TUTTI i periodi, incluso lo storico dei consumi se presente.
    
    REGOLE DI ESTRAZIONE:
    - Estrai solo dati di cui sei sicuro.
    - Per campi obbligatori (date, codice, indirizzo, consumo, consumo_medio): se mancano, ometti l'intero periodo.
    - Codice: usa numero di matricola del contatore; se non disponibile, usa codice contratto.
    - Assicurati che le date siano in formato ISO yyyy-mm-dd.
    """
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Pagina con dati consumo e contatore",
            "patterns": [
                r"(?:CONSUMO|LETTURA|CONTATORE|MATRICOLA)",
                r"(?:m³|m3|mc|METRI\s+CUBI)",
            ]
        }
    ]
    
    consumi: List[PeriodoConsumoAcqua] = Field(
        ...,
        title="Elenco Periodi di Consumo",
        description="Lista dei periodi di consumo idrico presenti nella bolletta (correnti e storici)."
    )
