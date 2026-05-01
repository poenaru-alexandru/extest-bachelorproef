# Document Extraction Tester — Bachelorproef PoC

Framework voor het vergelijken van lokale en cloud-gebaseerde LLM-strategieën voor het extraheren van gestructureerde data uit PDF-documenten (Italiaanse nutsrekeningen). Meet extractienauwkeurigheid, inferentiesnelheid, tokengebruik en energieverbruik (kWh/CO₂).

---

## Vereisten

- Python 3.11+
- NVIDIA GPU met CUDA
- `llama-server` binary voor lokale modellen (zie installatie hieronder)

---

## Installatie

### 1. Python-afhankelijkheden

```bash
pip install -r requirements.txt
```

### 2. llama-server (voor lokale modellen)

`llama-server` is een native C++-binary en wordt niet via pip geïnstalleerd. Bouwen vanuit broncode is de aanbevolen aanpak op beide platformen — zo wordt gecompileerd tegen de exacte CUDA-versie en GPU-architectuur van die machine.

Zoek eerst de compute capability van je GPU op via [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) (RTX 4060 = `89`). Pas de waarde van `CMAKE_CUDA_ARCHITECTURES` hieronder aan als de Ubuntu-server een andere GPU heeft.

**Ubuntu / Debian:**

Vereiste: CUDA Toolkit geïnstalleerd vóór het bouwen (`nvcc --version` moet werken). Op Ubuntu 22.04/24.04 via het NVIDIA-pakketarchief:
```bash
# Voorbeeld voor CUDA 12.x — pas de versie aan naar wat beschikbaar is voor jouw distro
sudo apt install cuda-toolkit-12-8
```

Daarna bouwen:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89   # pas aan naar de compute capability van de server-GPU
cmake --build build --config Release -j$(nproc)
# Binary staat in: build/bin/llama-server
# Optioneel: sudo cp build/bin/llama-server /usr/local/bin/
```

Bouwen vanuit broncode linkt automatisch tegen de aanwezige CUDA Toolkit — versie-mismatches zijn hier niet aan de orde.

**Windows:**

Bouwen vanuit broncode is de aanbevolen aanpak — zo wordt de binary gecompileerd tegen de exacte CUDA-versie die op jouw machine staat en vermijd je stille CPU-fallback bij een versie-mismatch.

Vereisten:
- Visual Studio 2022 met de workload "Desktop development with C++"
- CUDA Toolkit geïnstalleerd (zie opmerking over versies hieronder)

Open **"x64 Native Tools Command Prompt for VS 2022"** vanuit het Startmenu (niet een gewone PowerShell — `cl.exe` moet op PATH staan). Voer daarin uit:

```cmd
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

Binary staat in: `build\bin\Release\llama-server.exe`

> Als je Strawberry Perl geïnstalleerd hebt, zorg dan dat de cmake van Visual Studio eerder in PATH staat dan die van Strawberry (`C:\Strawberry\c\bin`). De x64 Native Tools Command Prompt regelt dit automatisch.

**Alternatief: pre-built binary van GitHub Releases**

Op [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) staan kant-en-klare Windows-binaries, bijvoorbeeld `llama-bXXXX-bin-win-cuda-12.4-x64.zip`. Let op: de CUDA-versie in de bestandsnaam (`cuda-12.4`, `cuda-12.8`, …) moet overeenkomen met de **major versie** van de CUDA Toolkit die op jouw systeem staat.

| Geïnstalleerde CUDA Toolkit | Gebruik binary met label |
|-----------------------------|--------------------------|
| 12.x                        | `cuda-12.x`              |
| 13.x                        | `cuda-13.x` (indien beschikbaar) of zelf bouwen |

Controleer je CUDA-versie via `nvcc --version` of in het NVIDIA Control Panel. Als de versies niet overeenkomen, laadt llama-server het model stil op de CPU in plaats van de GPU — zonder foutmelding.

Stel daarna `LLAMA_SERVER_PATH` in `.env` in op het volledige pad naar de binary (zie Configuratie hieronder).

### 3. Lokale modellen

Plaats GGUF-modelbestanden in de map `BP/local_models/`:

```
BP/
  local_models/
    llama-3.1-8b-instruct-q4_k_m.gguf
    qwen2.5-7b-instruct-q4_k_m.gguf
```

---

## Configuratie

Kopieer `.env.example` naar `.env` en vul in:

```env
# Providers: modelnamen per provider (JSON, één regel)
LLM_PROVIDERS={"huggingface": {"api_key": "hf_...", "base_url": "https://api-inference.huggingface.co/v1", "models": ["meta-llama/llama-3.1-8b-instruct"]}, "local": {"models": ["llama-3.1-8b-instruct-q4_k_m"]}}

# Pad naar llama-server binary (weglaten als het op PATH staat)
# Windows:  LLAMA_SERVER_PATH=C:/tools/llama.cpp/llama-server.exe
# Linux:    LLAMA_SERVER_PATH=/usr/local/bin/llama-server
LLAMA_SERVER_PATH=

# Serverpoort (standaard 8080)
LLAMA_SERVER_PORT=8080

# Hardware-parameters (geoptimaliseerd voor RTX 4060 8 GB)
LLAMA_SERVER_N_CTX=32768
LLAMA_SERVER_N_BATCH=2048
LLAMA_SERVER_N_UBATCH=2048
LLAMA_SERVER_FLASH_ATTN=on
LLAMA_SERVER_N_GPU_LAYERS=-1
LLAMA_SERVER_CACHE_TYPE_K=q4_0
LLAMA_SERVER_CACHE_TYPE_V=q4_0
LLAMA_SERVER_VERBOSE=false

# Modellaadstrategie voor lokale modellen (zie hieronder)
LLAMA_RELOAD_MODEL_PER_CALL=true
```

### Modellaadstrategie (`LLAMA_RELOAD_MODEL_PER_CALL`)

| Waarde | Gedrag | Wanneer gebruiken |
|--------|--------|-------------------|
| `true` | Laad model → inferentie → verwijder uit VRAM, voor elke aanroep | Geïsoleerde metingen, maximale reproduceerbaarheid |
| `false` | Laad model eenmalig → verwerk alle PDF's → verwijder → volgende model | Snellere batch-benchmarks |

---

## Starten

### Web UI (aanbevolen voor interactief testen)

```bash
cd web_ui
python app.py
# Opent op http://localhost:5000
```

### CLI-batchrun (alle PDF's × alle modellen)

```bash
python test_all.py
```

Resultaten worden per sessie opgeslagen in `results/results_YYYYMMDD_HHMMSS/` als JSON.

---

## Projectstructuur

```
extraction_framework/
├── extractors/               # PDF naar tekst (PyMuPDF4LLM → Markdown)
├── llm_providers/
│   ├── huggingface_provider.py   # Cloud: HuggingFace / NVIDIA NIM
│   ├── llama_cpp_provider.py     # Lokaal: HTTP-client naar llama-server
│   ├── local_server_manager.py   # Start/stop llama-server subprocess
│   └── base_provider.py          # Abstracte basisklasse + promptbuilder
├── web_ui/app.py             # Flask-interface (poort 5000)
├── test_runner.py            # Hoofdorchestrator
├── test_all.py               # CLI-batchrun
├── scoring.py                # ExtractionResult + validatiescoring
├── ground_truth.py           # Beheer van referentiedata
├── Test/
│   ├── modello.py            # Pydantic-extractieschema (BachelorProefModel)
│   └── *.pdf                 # Testdocumenten
└── results/                  # Uitvoer per sessie (JSON)
```

---

## Probleemoplossing

### Model draait op CPU in plaats van GPU

Symptomen: 0% GPU-gebruik in Taakbeheer, RAM loopt snel vol, inferentie is erg traag.

**Stap 1 — Zet verbose logging aan** in `.env`:
```env
LLAMA_SERVER_VERBOSE=true
```
Start opnieuw. De startup-output van llama-server verschijnt nu in de terminal. Zoek naar regels zoals:
- `ggml_cuda_init: found 1 CUDA devices` → GPU wordt herkend
- `ggml_cuda_init: CUDA not available` of geen CUDA-regels → GPU wordt niet gevonden

**Stap 2 — Controleer CUDA-versie** (alleen Windows pre-built binaries)

De DLL-namen bevatten de CUDA major-versie (`cudart64_12.dll` voor CUDA 12, `cudart64_13.dll` voor CUDA 13). Een pre-built binary gecompileerd voor CUDA 12.4 vindt `cudart64_13.dll` niet en valt stil terug op CPU.

```powershell
nvcc --version                          # toont geïnstalleerde CUDA Toolkit versie
```

Oplossingen (kies één):

- **Download een pre-built binary die overeenkomt met jouw CUDA-versie.** Controleer via `nvcc --version` welke versie geïnstalleerd is en download de bijpassende binary van GitHub Releases.
- **Kopieer de 3 benodigde DLLs naar de map van `llama-server.exe`.** Windows zoekt altijd eerst in de eigen map van de binary, dus er is geen PATH-wijziging nodig:
  ```powershell
  $src = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
  $dst = "C:\pad\naar\llama-server-map"
  Copy-Item "$src\cudart64_12.dll"   $dst
  Copy-Item "$src\cublas64_12.dll"   $dst
  Copy-Item "$src\cublasLt64_12.dll" $dst
  ```
  Pas `v12.4` aan naar de versie die overeenkomt met de binary. Let op: de CUDA Toolkit-installer voegt de nieuwe versie niet altijd automatisch toe aan PATH als er al een andere versie aanwezig is — deze aanpak omzeilt dat probleem.
- **Bouw llama-server zelf vanuit broncode** — dan wordt altijd de juiste versie gebruikt.

**Stap 3 — Controleer `LLAMA_SERVER_N_GPU_LAYERS`** in `.env`:
```env
LLAMA_SERVER_N_GPU_LAYERS=-1   # -1 = alle lagen op GPU
```

---

## Gemeten KPI's per inferentieaanroep

| Metriek | Bron |
|---------|------|
| Time-to-first-token (TTFT) | Streamingtelemetrie |
| Generatietijd | Streamingtelemetrie |
| Input- / outputtokens | `stream_options: {include_usage: true}` |
| Energieverbruik (kWh) | CodeCarbon (lokaal) / EcoLogits (cloud) |
| CO₂-uitstoot (kg) | CodeCarbon (lokaal) / EcoLogits (cloud) |
| CPU / GPU / RAM-energie | CodeCarbon hardware-telemetrie (alleen lokaal) |
