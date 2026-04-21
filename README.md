# 📄 Document Extraction Tester

Complete system for testing and comparing different data extraction strategies from PDFs using LLMs.

## 🎯 Main Features

- ✅ **100% Automatic**: Add a folder in `Test/` with `modello.py` and it works
- ✅ **Multi-Strategy**: Tests PyMuPDF-XML, pypdf-XML, PDF-to-Images
- ✅ **Multi-LLM**: Supports Gemini, Ollama (and others like OpenAI, Groq, OpenRouter, Azure)
- ✅ **Intelligent Filtering**: Regex validation of pages to reduce tokens and improve accuracy
- ✅ **Comparative Analysis**: Automatic ranking of technology stacks
- ✅ **Web UI**: Complete interface for testing and analysis
- ✅ **Ground Truth**: Automatic validation system

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd document-extraction-tester

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```bash
# LLM Providers Configuration (JSON format)
LLM_PROVIDERS={"gemini": {"api_key": "YOUR_GEMINI_API_KEY", "models": ["gemini-1.5-pro", "gemini-2.0-flash-exp"]}, "ollama": {"base_url": "http://localhost:11434/v1", "models": ["llama3", "mistral"]}}

# Optional: specific provider configuration
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### 3. Add your documents

```bash
cd Test
mkdir my_test_folder
cd my_test_folder

# Create modello.py (see QUICK_START.md for a template)
# Copy your PDFs into the folder

# DONE! The system automatically detects everything
```

### 4. Start the server

```bash
cd extraction_framework/web_ui
python app.py

# Open browser: http://localhost:5000
```

## 📚 Documentation

- **[QUICK_START.md](./QUICK_START.md)** - How to add a new test in 3 minutes
- **[STANDARD_TEST_CREATION.md](./STANDARD_TEST_CREATION.md)** - Complete guide with templates, regex, best practices

## 🏗️ Architecture

### Directory Structure

```
document-extraction-tester/
├── Test/                           # 👈 Add your folders here
│   ├── bolletta_ee/
│   │   ├── modello.py             # Pydantic model + PAGE_VALIDATION_RULES
│   │   ├── documento1.pdf
│   │   └── documento2.pdf
│   ├── bolletta_acqua/
│   │   ├── modello.py
│   │   └── *.pdf
│   └── [your_new_folder]/         # ✨ Auto-discovered!
│       ├── modello.py
│       └── *.pdf
│
├── extraction_framework/
│   ├── extractors/                # PDF → Text/XML/Images
│   │   ├── base_extractor.py     # With PageValidator embedded
L92- │   │   ├── pymupdf_extractor.py  # PyMuPDF-XML
L94- │   │   └── image_extractor.py    # PDF → Images (for vision APIs)
L95- │   │
L96- │   ├── llm_providers/             # Text → Structured Data
│   │   ├── openai_provider.py    # OpenAI-compatible (GPT, Groq, Ollama, etc.)
│   │   └── gemini_provider.py    # Google Gemini
│   │
│   ├── page_validator.py          # Regex validation rules
│   ├── model_loader.py            # Dynamic model loading
│   ├── test_runner.py             # Orchestration
│   └── web_ui/                    # Flask app
│       ├── app.py
│       └── templates/index.html
│
├── QUICK_START.md                 # 👈 Start here
├── STANDARD_TEST_CREATION.md      # 👈 Complete guide
└── README.md                      # 👈 You are here
```

### Workflow

```
1. 📂 AUTO-DISCOVERY
   Test/ → Scan folders → Load modello.py → Detect PDFs

2. 🔧 PRE-PROCESSING
   PDF → Extractor (PyMuPDF/pypdf/Images) → Pages

3. ✅ PAGE VALIDATION
   Pages → PageValidator (regex rules) → Filtered Pages
   ├─ Remove invalid HEAD pages
   ├─ Remove invalid TAIL pages
   └─ Keep invalid MIDDLE pages (context)

4. 🤖 LLM EXTRACTION
   Filtered Pages + Pydantic Model → LLM → Structured Data

5. 📊 ANALYSIS
   Multiple extractions → Scoring → Ranking → Best Stack
```

## 🎨 Web UI Tabs

### ⚙️ Configuration
- Select test folder (auto-discovered)
- Select PDF
- Select Extractors (multi-select)
- Select LLM providers and models
- Run single tests or complete suites

### 📊 Results
- Real-time debug log
- Statistics: time, tokens, success/failure
- Cards for each extraction

### 🔍 Comparison
- Global analytics: best extractors, LLMs, complete stacks
- Comparative table per document
- Extracted data visualization
- Ranking with ties

### ✅ Validation
- PDF viewer
- Comparison with ground truth
- Save ground truth

### 🔬 Regex Test
- Batch testing on all PDFs
- Filtering statistics: pages removed, text reduction
- Regex pattern validation

## 🧪 Complete Example

### 1. Create Test Folder

```bash
cd Test
mkdir bolletta_gas
```

### 2. Create `modello.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional, ClassVar, Dict
from datetime import date

class DatiBollettaGas(BaseModel):
    """Model for gas bills"""
    
    PAGE_VALIDATION_RULES: ClassVar[List[Dict]] = [
        {
            "description": "Page with gas and consumption",
            "patterns": [r"gas", r"consumo", r"smc"]
        },
        {
            "description": "Page with PDR",
            "patterns": [r"PDR", r"\d{14}"]
        }
    ]
    
    numero_bolletta: Optional[str] = None
    pdr: Optional[str] = None
    consumo_smc: Optional[float] = None
    importo_totale: Optional[float] = None
```

### 3. Add PDFs

```bash
cp ~/bollette/gas_*.pdf bolletta_gas/
```

### 4. Test

```bash
# Start server
cd ../../extraction_framework/web_ui
python app.py

# Browser: http://localhost:5000
# - "Configuration" Tab → Select "bolletta_gas"
# - "Regex Test" Tab → Automatic test on all PDFs
# - "Configuration" Tab → Run test with different stacks
# - "Comparison" Tab → See ranking of best combinations
```

## 📈 Metrics and Scoring

The system automatically tracks:

- **Extraction Time**: Total extraction time
- **Token Usage**: Input/Output/Total tokens
- **Page Filtering**: Pages removed, text reduction
- **Data Quality**: Number of extracted records, field completeness
- **Success Rate**: Percentage of successful extractions

### Ranking Algorithm

```python
score = (
    data_quality * 0.4 +      # Data completeness
    consistency * 0.3 +       # Value consistency
    efficiency * 0.2 +        # Optimal tokens/time
    robustness * 0.1          # Success on different PDFs
)
```

## 🔧 Advanced Configuration

### Custom LLM Provider

```json
{
  "custom_provider": {
    "api_key": "your-key",
    "base_url": "https://your-endpoint/v1",
    "models": ["model-1", "model-2"]
  }
}
```

### Custom Extractors

Create a new extractor in `extraction_framework/extractors/`:

```python
from .base_extractor import BaseExtractor

class MyExtractor(BaseExtractor):
    name = "My-Custom-Extractor"
    
    def extract_text(self, pdf_path):
        # Your logic
        return text
```

Register in `__init__.py`:

```python
from .my_extractor import MyExtractor

def get_all_extractors():
    return [
        # ... existing ...
        MyExtractor()
    ]
```

## 🐛 Troubleshooting

### Folder does not appear in frontend

```bash
# Verify detection
cd extraction_framework/web_ui
python -c "from app import model_loader; print(model_loader.list_test_folders())"
```

**Cause**: `modello.py` is missing or has syntax errors

### Empty Extractors/Providers

**Cause**: Error in `.env` or missing imports

```bash
# Test config
python -c "from app import *; print('Extractors:', [e.name for e in get_all_extractors()]); print('Providers:', get_available_providers())"
```

### Regex not filtering

**Goal**: 60-80% validation

- Too strict → 0% validation
- Too loose → 100% validation (useless)

Use the "Regex Test" tab to optimize.

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Commit with clear messages
5. Push and create PR

## 📄 License

MIT License - see LICENSE file

## 🙏 Credits

- PyMuPDF for PDF processing
- pypdf for metadata extraction
- Pydantic for data validation
- Flask for web UI
- OpenAI/Gemini/Anthropic for LLM APIs

---

💡 **Start now**: See [QUICK_START.md](./QUICK_START.md) to create your first test in 3 minutes!
