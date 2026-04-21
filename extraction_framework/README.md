# Document Extraction Tester Framework (Thesis Edition)

Lightweight framework for comparing Local vs. Cloud LLM strategies for PDF data extraction.

## 🚀 Research Scope

This framework is optimized for a bachelor thesis POC:
- **PDF-Only**: Focuses on Native and Scanned PDF documents.
- **Local vs Cloud**: Comparison between Ollama (Local) and NVIDIA NIM (Cloud).
- **KPIs**: Measures extraction time, token usage, and accuracy (Ground Truth).
- **Efficiency**: Intelligent Regex Page Filtering to reduce context size.

## 📁 Structure

```
extraction_framework/
├── extractors/           # PDF Extractors (PyMuPDF-XML, PDF-Images, PDF-Direct)
├── llm_providers/        # LLM Providers (OpenAI-compatible for NVIDIA & Ollama)
├── web_ui/              # Flask web interface
├── results/             # Test results (JSON)
├── ground_truth/        # Validated reference data for scoring
├── scoring.py           # Comparison and consensus system
├── ground_truth.py      # Ground truth management
├── page_validator.py    # Regex-based page filtering
└── test_runner.py       # Main orchestration engine
```

## 🔧 Installation

```powershell
cd extraction_framework
pip install -r requirements.txt
```

## 🤖 Configuration

Create a `.env` file with your providers:

```env
LLM_PROVIDERS={"nvidia": {"api_key": "nvapi-...", "base_url": "https://integrate.api.nvidia.com/v1", "models": ["nvidia/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct-v0.2"]}, "local": {"api_key": "not-needed", "base_url": "http://localhost:11434/v1", "models": ["llama3.1:8b", "mistral"]}}
```

## 🔍 Extractors

| Extractor | Use Case |
|-----------|----------|
| **PyMuPDF-XML** | Native PDFs with structured layout |
| **PDF-Images** | Scanned PDFs (converts pages to images for vision LLMs) |
| **PDF-Direct** | Modern LLMs with native PDF support |

## 📊 Scoring

The system automatically compares results against a **Ground Truth** JSON file if it exists in the test folder. It calculates:
- **Field Agreement**: Consistency across different models.
- **Accuracy**: F1-like score against validated data.
- **Efficiency**: Tokens saved by regex filtering.

## 🌐 Web UI

```powershell
cd extraction_framework/web_ui
python app.py
```
Open `http://localhost:5000` to run tests and visualize comparisons.
