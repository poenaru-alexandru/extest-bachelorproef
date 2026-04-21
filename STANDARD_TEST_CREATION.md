# 📚 Standard per la Creazione di Nuovi Test

## 🎯 Overview

Il sistema è progettato per essere **completamente automatico**: basta creare una nuova cartella nella directory `Test/` seguendo una struttura standard, e il sistema rileverà automaticamente tutto.

## 🏗️ Struttura Standard di una Cartella Test

```
Test/
  └── nome_documento/          ← Nome descrittivo del tipo di documento
      ├── modello.py           ← OBBLIGATORIO: Definizione Pydantic model + regole validazione
      ├── documento1.pdf       ← PDF di test
      ├── documento2.pdf
      └── documento3.pdf
```

## 📋 Template modello.py

```python
"""Modello Pydantic per [NOME DOCUMENTO]"""
from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict
from datetime import date


# ========== MODELLO DATI ==========

class RecordConsumo(BaseModel):
    """Singolo record di consumo"""
    periodo_inizio: Optional[date] = None
    periodo_fine: Optional[date] = None
    quantita: Optional[float] = None
    unita_misura: Optional[str] = None
    costo: Optional[float] = None


class DatiDocumento(BaseModel):
    """Modello principale per [NOME DOCUMENTO]
    
    IMPORTANTE: 
    - Questo è il modello root che deve essere estratto dall'LLM
    - DEVE iniziare con "Dati" (es: DatiBolletta, DatiContratto, DatiFattura)
    - Il sistema seleziona automaticamente la prima classe che inizia con "Dati"
    - Deve contenere tutti i dati che vuoi estrarre dal documento
    """
    
    # ===== REGOLE DI VALIDAZIONE PAGINE =====
    # Queste regole vengono usate per filtrare le pagine NON valide
    # prima di passarle all'LLM, risparmiando token e migliorando accuracy
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Pagina deve contenere keyword principali del documento",
            "patterns": [
                r"fattura",           # Cerca questa parola (case insensitive)
                r"bolletta",          # E anche questa
                r"consumo"            # E anche questa
            ]
            # Se TUTTE le patterns vengono matchate → pagina VALIDA
        },
        {
            "description": "Pagina con dati periodo",
            "patterns": [
                r"periodo",
                r"\d{2}/\d{2}/\d{4}"  # Match date formato gg/mm/aaaa
            ]
        },
        {
            "description": "Pagina con codici identificativi",
            "patterns": [
                r"codice\s+cliente",
                r"numero\s+fattura"
            ]
        }
    ]
    # LOGICA: Se una pagina matcha ALMENO UNA regola → VALIDA
    # Le pagine invalide consecutive vengono rimosse da testa/coda del PDF
    # Le pagine invalide nel mezzo vengono mantenute (potrebbero essere utili per contesto)
    
    # ===== CAMPI DATI =====
    
    numero_documento: Optional[str] = Field(None, description="Numero identificativo del documento")
    data_emissione: Optional[date] = Field(None, description="Data di emissione")
    codice_cliente: Optional[str] = Field(None, description="Codice identificativo cliente")
    
    # Lista di consumi (può essere vuota, non required)
    consumi: List[RecordConsumo] = Field(default_factory=list, description="Lista dei consumi")
    
    importo_totale: Optional[float] = Field(None, description="Importo totale da pagare")


# ========== REGOLE SCORING (OPZIONALE) ==========
# Se vuoi validare automaticamente i risultati contro ground truth

UNIQUE_IDENTIFIERS = [
    "numero_documento",  # Campo che identifica univocamente il documento
    "codice_cliente"     # Altri campi identificativi
]

IGNORED_FIELDS = {
    "campo_poco_importante",  # Campi da ignorare nel calcolo dello score
}
```

## 🔧 Come Funziona il Sistema Automatico

### 1. **Discovery Automatico**

Quando avvii il server Flask (`python app.py`), il sistema:

1. Scansiona tutte le sottocartelle in `Test/`
2. Cerca file `modello.py` in ciascuna cartella
3. Carica dinamicamente il modello Pydantic
4. Conta i PDF presenti
5. Rileva i file JSON di ground truth (se presenti)

```python
# In extraction_framework/model_loader.py
def list_test_folders(self) -> List[Dict[str, Any]]:
    """Rileva automaticamente tutte le cartelle test"""
    test_folders = []
    
    for folder in self.test_dir.iterdir():
        if not folder.is_dir() or folder.name.startswith('.'):
            continue
        
        # Cerca modello.py
        model_files = list(folder.glob("modello*.py"))
        if not model_files:
            continue  # Ignora cartelle senza modello
        
        # Conta PDF
        pdf_files = list(folder.glob("*.pdf"))
        
        # Conta JSON validati
        json_files = list(folder.glob("*.json"))
        json_files = [j for j in json_files if not j.name.startswith('modello')]
        
        test_folders.append({
            "name": folder.name,
            "path": str(folder),
            "model_files": [m.name for m in model_files],
            "pdf_count": len(pdf_files),
            "validated_count": len(json_files)
        })
    
    return test_folders
```

### 2. **Caricamento Regole Validazione**

```python
# In extraction_framework/page_validator.py
def load_validation_rules_from_model(model_module) -> List[Dict]:
    """Carica regole da ClassVar nel modello Pydantic"""
    
    # Cerca nel modulo tutte le classi che hanno PAGE_VALIDATION_RULES
    for name in dir(model_module):
        obj = getattr(model_module, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            if hasattr(obj, 'PAGE_VALIDATION_RULES'):
                return obj.PAGE_VALIDATION_RULES
    
    return []  # Nessuna regola trovata
```

### 3. **Estrazione Automatica con Validazione**

```python
# In extraction_framework/test_runner.py
def run_extraction(self, pdf_path, extractor, llm_provider, llm_model, test_folder):
    """Esegue estrazione con validazione automatica"""
    
    # 1. Carica modello e regole
    model_module = self.model_loader.get_module_for_test(test_folder)
    validation_rules = load_validation_rules_from_model(model_module)
    
    # 2. Crea validator e iniettalo nell'extractor
    if validation_rules:
        validator = PageValidator(validation_rules)
        extractor.set_page_validator(validator)
    
    # 3. Estrazione con filtro automatico
    pages, filter_stats = extractor.extract_pages_filtered(pdf_path, verbose=True)
    
    # 4. Passa all'LLM solo le pagine valide
    extracted_data = llm_provider.extract(pages, model_class)
    
    return extracted_data, filter_stats
```

## 📊 Statistiche di Filtering

Il sistema traccia automaticamente:

- **total_pages**: Pagine totali nel PDF
- **validated_pages**: Pagine che hanno passato almeno una regola
- **filtered_pages**: Pagine effettivamente passate all'LLM
- **removed_from_head**: Pagine invalide rimosse dall'inizio
- **removed_from_tail**: Pagine invalide rimosse dalla fine
- **text_reduction_percent**: Percentuale di testo risparmiato

Esempio output:

```
=== PAGE VALIDATION STATS ===
Total pages: 8
Validated pages: 5 (62.5%)
Filtered pages: 6 (keeps invalid pages in middle)
Removed from head: 1
Removed from tail: 1
Text reduction: 23.4%
=============================
```

## 🎨 Pattern Regex Utili

### Date

```python
r"\d{2}/\d{2}/\d{4}"      # 25/12/2024
r"\d{2}-\d{2}-\d{4}"      # 25-12-2024
r"\d{4}-\d{2}-\d{2}"      # 2024-12-25 (ISO)
r"(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)"
```

### Numeri e Importi

```python
r"€\s*\d+[.,]\d+"         # €123,45 o € 123.45
r"\d+[.,]\d+\s*€"         # 123,45€
r"totale.*€\s*\d+"        # "Totale: € 123"
```

### Codici

```python
r"POD\s*[A-Z0-9]+"        # POD IT001E12345678
r"PDR\s*\d+"              # PDR 12345678
r"CF\s*[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]"  # Codice Fiscale
r"P\.IVA\s*\d{11}"        # Partita IVA
```

### Bollette Energia

```python
r"consumo.*kWh"           # Consumo energia
r"potenza.*kW"            # Potenza
r"lettura.*contatore"     # Lettura contatore
r"fasce.*F\d"             # Fasce orarie (F1, F2, F3)
```

### Bollette Acqua

```python
r"metri\s*cubi|m3|mc"     # Volume acqua
r"contatore.*acqua"       # Riferimenti contatore
r"lettura.*precedente"    # Letture
```

## ✅ Checklist Creazione Nuovo Test

- [ ] **Cartella creata** in `Test/nome_documento/`
- [ ] **modello.py creato** con:
  - [ ] Import di `BaseModel`, `ClassVar`, `List`, `Dict`
  - [ ] Classi Pydantic per struttura dati
  - [ ] `PAGE_VALIDATION_RULES` come `ClassVar[List[Dict]]` nella classe root
  - [ ] Almeno 2-3 regole di validazione con pattern regex
  - [ ] Docstring chiare su ogni classe
- [ ] **PDF di test** aggiunti (almeno 2-3 documenti diversi)
- [ ] **Test nel browser**:
  - [ ] Riavvia server Flask
  - [ ] Cartella appare nel dropdown "Test Folder"
  - [ ] PDF appaiono nel dropdown "File PDF"
  - [ ] Tab "Test Regex" mostra statistiche di filtering
- [ ] **(Opzionale)** Ground truth JSON creati per validazione automatica

## 🚀 Workflow Completo

1. **Crea cartella**: `Test/bolletta_gas/`
2. **Crea modello**: `Test/bolletta_gas/modello.py` (usa template sopra)
3. **Aggiungi PDF**: Copia i PDF di test nella cartella
4. **Riavvia server**: `python extraction_framework/web_ui/app.py`
5. **Apri browser**: http://localhost:5000
6. **Testa regex**: Tab "Test Regex" → seleziona cartella → "Test Tutti i PDF"
7. **Ottimizza regole**: Modifica `PAGE_VALIDATION_RULES` in `modello.py` finché ottieni 70-90% validazione
8. **Esegui estrazione**: Tab "Configurazione" → seleziona test folder, PDF, extractor, LLM → "Esegui Test"
9. **Confronta risultati**: Tab "Confronto" → vedi ranking stack tecnologici

## 🔍 Debug e Troubleshooting

### Cartella non appare nel dropdown

**Causa**: Manca `modello.py` o ha errori di sintassi

**Soluzione**:
```bash
# Test caricamento modello
cd extraction_framework/web_ui
python -c "from app import model_loader; print(model_loader.list_test_folders())"
```

### Extractors/Providers vuoti

**Causa**: Errore nel backend o nel caricamento config

**Soluzione**:
- Controlla debug log nel browser (tab "Risultati")
- Verifica `.env` con `LLM_PROVIDERS`
- Controlla terminale server per errori Python

### Regex non filtra pagine

**Causa**: Pattern regex troppo specifici o formato scorretto

**Soluzione**:
- Usa tab "Test Regex" per vedere quante pagine vengono matchate
- Obiettivo: 60-80% validazione
- Pattern troppo strict → nessuna pagina passa
- Pattern troppo loose → troppe pagine invalide passano

### LLM non estrae dati corretti

**Cause possibili**:
1. Troppe pagine invalide passate (regex troppo loose)
2. Modello Pydantic troppo complesso
3. PDF con formato inaspettato

**Soluzioni**:
1. Raffina `PAGE_VALIDATION_RULES`
2. Semplifica modello Pydantic
3. Aggiungi più esempi di PDF diversi
4. Prova diversi LLM tra i 3 NVIDIA e i 3 locali configurati

## 📝 Esempi Completi

Vedi cartelle esistenti come reference:

- `Test/bolletta_ee/` - Bollette elettricità con POD, fasce orarie
- `Test/bolletta_acqua/` - Bollette acqua con metri cubi, letture contatore

## 🎓 Best Practices

1. **Regole Regex**: Parti da pattern generali, poi specifica
2. **Validazione**: Obiettivo 70-80% delle pagine validate (non troppo, non troppo poco)
3. **Modello Pydantic**: Usa `Optional` per campi non sempre presenti
4. **Testing**: Testa con almeno 3 PDF diversi prima di considerare finito
5. **Iterazione**: Usa tab "Confronto" per identificare lo stack migliore
6. **Ground Truth**: Crea almeno 2 ground truth JSON per validazione automatica

## 🆘 Supporto

In caso di problemi:

1. Controlla debug log nel browser (tab "Risultati")
2. Controlla log del server Flask nel terminale
3. Verifica che tutti i file seguano la struttura standard
4. Confronta con cartelle esistenti funzionanti
