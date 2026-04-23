from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, ClassVar, Dict


class RigaRegistroRifiuto(BaseModel):
    """Singola riga di un registro di carico/scarico rifiuti o di un formulario FIR."""
    anno: Optional[int] = Field(
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
        description="Peso o quantità del rifiuto. Estrarre il valore numerico."
    )
    tipo: str = Field(
        ...,
        title="Descrizione Rifiuto",
        description="Descrizione testuale del rifiuto come riportato nel documento."
    )
    codice_smaltimento: Optional[str] = Field(
        None,
        title="Codice R/D",
        description="Codice operazione di recupero o smaltimento (es. R13, D15)."
    )

    @field_validator('codice_cer')
    @classmethod
    def clean_cer(cls, v: str) -> str:
        """Pulisce il codice CER rimuovendo punti o spazi"""
        return v.replace('.', '').replace(' ', '')


class DatiRifiuti(BaseModel):
    """
    ESTRAI TUTTI I MOVIMENTI DI RIFIUTI DAL REGISTRO O FORMULARIO.
    
    Questi documenti contengono elenchi di rifiuti prodotti, trasportati o smaltiti.
    Il tuo compito è estrarre OGNI riga della tabella che rappresenta un movimento di rifiuto.
    
    ISTRUZIONI:
    1. Scorri le tabelle dei movimenti (carico/scarico o riepiloghi MUD).
    2. Per ogni riga, identifica il codice CER (6 cifre) e la relativa quantità.
    3. Estrai TUTTE le righe trovate e inseriscile nel campo 'rifiuti'.
    4. Se mancano il codice CER o la quantità, quella riga va ignorata.
    5. Non inventare informazioni; se un campo come l'anno o il codice smaltimento non è chiaro, usa null.
    """
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Pagina con dati CER e movimenti rifiuti",
            "patterns": [
                r"(?i)(CER|CODICE|RIFIUTO|RIFIUTI)",
                r"(?i)(KG|TON|QUANTITÀ|PESO)",
                r"(?i)(CARICO|SCARICO|FORMULARIO|MUD)"
            ]
        }
    ]
    
    rifiuti: List[RigaRegistroRifiuto] = Field(
        ...,
        description="Lista completa dei movimenti di rifiuti estratti dal documento."
    )
