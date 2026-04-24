from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, ClassVar, Dict


class RigaRegistroRifiuto(BaseModel):
    """Singola riga di un registro di carico/scarico rifiuti o di un formulario FIR."""
    anno: int = Field(
        None,
        title="Anno",
        description="Anno di riferimento del movimento (es. 2023)."
    )
    codice_cer: str = Field(
        ...,
        title="Codice CER",
        description="Codice del Catalogo Europeo dei Rifiuti (6 cifre, es. 150101). Rimuovere eventuali punti."
    )
    quantita: float = Field(
        ...,
        title="Quantità (kg/t)",
        description="Peso o quantità del rifiuto. FORMATO: Il documento usa la virgola per i decimali e il punto per le migliaia. Rimuovi il punto delle migliaia e usa il punto decimale."
    )

    @field_validator('codice_cer')
    @classmethod
    def clean_cer(cls, v: str) -> str:
        """Pulisce il codice CER rimuovendo punti o spazi"""
        return v.replace('.', '').replace(' ', '')


class DatiRifiuti(BaseModel):
    """Contenitore per l'estrazione di tutti i movimenti di rifiuti da registri o formulari."""
    
    rifiuti: List[RigaRegistroRifiuto] = Field(
        ...,
        description=(
            "Lista completa dei movimenti di rifiuti estratti dal documento. "
            "Scorri le tabelle dei movimenti e identifica ogni riga con codice CER e quantità. "
            "CRITICO: Devi estrarre TUTTI gli elementi presenti nel documento. Devi restituire un array con MOLTEPLICI oggetti. "
            "L'array non deve MAI contenere un solo elemento se sono presenti più voci (come tabelle di carico/scarico o liste movimenti) nel testo."
        )
    )
