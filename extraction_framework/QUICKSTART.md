# Quick Guide - Document Extraction Tester (Thesis PoC Edition)

## 🚀 Fast Setup

### 1. Install dependencies
```powershell
cd extraction_framework
pip install -r requirements.txt
```

### 2. Configure .env
Copy `.env.example` to `.env` and configure the providers (Local Ollama and NVIDIA NIM) in JSON format (single line):

```env
LLM_PROVIDERS={"nvidia": {"api_key": "nvapi-YOUR_NVIDIA_API_KEY", "base_url": "https://integrate.api.nvidia.com/v1", "models": ["nvidia/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct-v0.2", "qwen/qwen2.5-coder-7b-instruct"]}, "local": {"api_key": "not-needed", "base_url": "http://localhost:11434/v1", "models": ["llama3.1:8b", "mistral", "qwen2.5-coder:7b"]}}
```

### 3. Start Web UI
```powershell
cd web_ui
python app.py
```

Open: `http://localhost:5000`

---

## 📊 Available Extractors (PDF Only)

| Name | Description | When to use it |
|------|-------------|----------------|
| **PyMuPDF-XML** | Structured XML with font info | Native PDFs (Digital) |
| **PDF-Images** | Vision-based extraction | Scanned PDFs (Images) |
| **PDF-Direct** | Native LLM PDF handling | Direct PDF uploads |

---

## 🤖 Research Stack

This POC compares:
- **Local Inference**: Ollama (Measured via CodeCarbon)
- **Cloud Inference**: NVIDIA NIM (Estimated via EcoLogits)

---

## 📁 Test Structure

```
Test/
  bolletta_acqua/
    modello.py          # Pydantic model
    invoice.pdf         # Native or Scanned PDF
    invoice.json        # Ground Truth for Scoring
```

---

## ✅ Typical Workflow

1. **Configure .env** with local/cloud endpoints.
2. **Add PDF** to a subfolder in `Test/`.
3. **Define Pydantic model** in `modello.py`.
4. **Run Test** from Web UI to compare combinations.
5. **Score accuracy** against Ground Truth.
