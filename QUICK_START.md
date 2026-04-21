# 🚀 Quick Start - Adding a New Test

## ⚡ 3 Quick Steps

### 1️⃣ Create the folder
```bash
cd Test
mkdir new_document_name
cd new_document_name
```

### 2️⃣ Copy the `modello.py` template

Use this base template and customize it:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict
from datetime import date

class YourData(BaseModel):
    field1: Optional[str] = None
    field2: Optional[float] = None

class DocumentData(BaseModel):
    """Document description
    
    IMPORTANT: The name MUST start with "Dati" to be automatically detected (e.g., DatiDocumento)
    """
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Important keywords",
            "patterns": [
                r"word1",
                r"word2"
            ]
        }
    ]
    
    number: Optional[str] = None
    date: Optional[date] = None
    data: List[YourData] = Field(default_factory=list)
```

### 3️⃣ Add PDFs and test

```bash
# Copy PDFs to the folder
cp /path/to/document.pdf .

# Restart the server (or let it restart automatically)
# The system automatically detects the new folder!
```

## ✅ Verification

Open http://localhost:5000:

- ✅ The new folder appears in the "Test Folder" dropdown
- ✅ The PDFs appear in the list
- ✅ You can run tests on them

## 📖 Complete Documentation

See [STANDARD_TEST_CREATION.md](./STANDARD_TEST_CREATION.md) for:
- Complete annotated template
- Useful regex patterns
- Best practices
- Troubleshooting

## 🎯 Real Example

```bash
cd Test
mkdir phone_bill
cd phone_bill

# Create modello.py with:
# - PAGE_VALIDATION_RULES with pattern "telefoni|chiamate|traffico"
# - Fields: contract_number, period, calls[], total_amount

# Copy phone bill PDFs
cp ~/Documents/bolletta_*.pdf .

# DONE! The system detects everything automatically
```

## 🔧 Useful Debug Commands

```bash
# List all detected folders
cd extraction_framework/web_ui
python -c "from app import model_loader; print([f['name'] for f in model_loader.list_test_folders()])"

# Verify model loading
python -c "from app import model_loader; model_loader.get_model_for_test('your_folder'); print('OK')"

# Test regex validation rules
python -c "from extraction_framework.page_validator import *; from app import model_loader; m = model_loader.get_module_for_test('your_folder'); print(len(load_validation_rules_from_model(m)), 'rules loaded')"
```

## ❓ Common Problems

**Folder does not appear** → `modello.py` is missing or has Python syntax errors

**Loading error** → Check Pydantic imports and ClassVar syntax

**Regex not filtering** → Patterns are too specific, loosen the regexes

---

💡 **Pro Tip**: Copy an existing folder (e.g., `bolletta_ee`) and modify it instead of starting from scratch!
