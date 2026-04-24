from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional, List, ClassVar
from datetime import datetime


class PeriodoConsumoEE(BaseModel):
    """
    Rappresenta un singolo periodo di consumo elettrico estratto dalla bolletta o dallo storico consumi.
    """
    codice: str = Field(
        ...,
        title="Codice POD",
        description="Codice identificativo del punto di prelievo (POD). Copiare esattamente dal documento (es. IT001E...)."
    )
    consumo: float = Field(
        ...,
        title="Consumo Totale (kWh)",
        description="Consumo totale in kWh. FORMATO: Il documento usa la virgola per i decimali e il punto per le migliaia. Rimuovi il punto delle migliaia e sostituisci la virgola con il punto per restituire un numero float valido (es. 1.234,56 -> 1234.56)."
    )
    indirizzo: str = Field(
        ...,
        title="Indirizzo Punto di Prelievo",
        description="Indirizzo completo associato al POD. Riportare fedelmente come nel documento."
    )
    consumo_f1: Optional[float] = Field(
        None,
        title="Consumo F1 (kWh)",
        description="Consumo in fascia F1. FORMATO: Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )
    consumo_f2: Optional[float] = Field(
        None,
        title="Consumo F2 (kWh)",
        description="Consumo in fascia F2. FORMATO: Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )
    consumo_f3: Optional[float] = Field(
        None,
        title="Consumo F3 (kWh)",
        description="Consumo in fascia F3. FORMATO: Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )
    giorno_fine: str = Field(
        ...,
        title="Data Fine Periodo",
        description="Data di fine del periodo di riferimento, nel formato ISO yyyy-mm-dd."
    )
    costo_periodo: Optional[float] = Field(
        None,
        title="Costo Periodo (€)",
        description="Importo totale fatturato per questo periodo. FORMATO: Il documento usa la virgola per i decimali e il punto per le migliaia. Rimuovi il punto delle migliaia e usa il punto per i decimali."
    )
    giorno_inizio: str = Field(
        ...,
        title="Data Inizio Periodo",
        description="Data di inizio del periodo di riferimento, nel formato ISO yyyy-mm-dd."
    )

    @field_validator('giorno_inizio', 'giorno_fine')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Forza il formato ISO yyyy-mm-dd"""
        if not v: return v
        for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
            try:
                dt = datetime.strptime(v, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        return v


class DatiBollettaEE(BaseModel):
    """Contenitore per l'estrazione di tutti i periodi di consumo elettrico (correnti e storici)."""
    
    consumi: List[PeriodoConsumoEE] = Field(
        ...,
        description=(
            "Lista completa di tutti i periodi di consumo (correnti e storici) trovati nel documento. "
            "Estrai OGNI SINGOLO periodo, inclusa la tabella dello 'Storico Consumi' (solitamente 12-13 mensilità). "
            "CRITICO: Devi estrarre TUTTI gli elementi presenti nel documento. Devi restituire un array con MOLTEPLICI oggetti. "
            "L'array non deve MAI contenere un solo elemento se sono presenti più voci (come tabelle storiche o liste) nel testo."
        )
    )
