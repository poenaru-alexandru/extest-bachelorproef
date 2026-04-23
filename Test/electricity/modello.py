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
        description="Consumo totale di energia elettrica in kWh per il periodo indicato."
    )
    indirizzo: str = Field(
        ...,
        title="Indirizzo Punto di Prelievo",
        description="Indirizzo completo associato al POD. Riportare fedelmente come nel documento."
    )
    consumo_f1: Optional[float] = Field(
        None,
        title="Consumo F1 (kWh)",
        description="Consumo in fascia F1. Usare null se non specificato."
    )
    consumo_f2: Optional[float] = Field(
        None,
        title="Consumo F2 (kWh)",
        description="Consumo in fascia F2. Usare null se non specificato."
    )
    consumo_f3: Optional[float] = Field(
        None,
        title="Consumo F3 (kWh)",
        description="Consumo in fascia F3. Usare null se non specificato."
    )
    giorno_fine: str = Field(
        ...,
        title="Data Fine Periodo",
        description="Data di fine del periodo di riferimento, nel formato ISO yyyy-mm-dd."
    )
    costo_periodo: Optional[float] = Field(
        None,
        title="Costo Periodo (€)",
        description="Importo totale fatturato per questo specifico periodo, se presente."
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
    """
    SEI UN AGENTE SPECIALIZZATO NELL'ESTRAZIONE DATI DA BOLLETTE ELETTRICHE.
    
    Il tuo obiettivo è estrarre OGNI SINGOLO periodo di consumo presente nel documento.
    Generalmente, una bolletta contiene:
    1. Un periodo principale (quello fatturato correntemente).
    2. Una tabella dello "Storico Consumi" o "Consumi ultimi 12 mesi" con molti altri periodi mensili.
    
    DEVI ESTRARRE TUTTI I PERIODI TROVATI, inclusi quelli nella tabella dello storico.
    Dovresti ottenere circa 12-13 oggetti nel campo 'consumi'.
    
    REGOLE MANDATORIE:
    - Estrai TUTTI i periodi di consumo (POD, date, kWh).
    - Se mancano date o consumo, scarta quel periodo.
    - NON inventare dati. Se un valore non è presente, usa null.
    """
    
    consumi: List[PeriodoConsumoEE] = Field(
        ...,
        description="Lista completa di tutti i periodi di consumo (correnti e storici) trovati nel documento."
    )
