# volumen
An ancient papyrus reader and AI tool.
Volumen - An Ancient Papyrus Reader

An AI-assisted desktop application for reading and analysing ancient papyrus fragments.

## Features

| Step | Technique | Notes |
|------|-----------|-------|
| 1 | Load image | PNG, JPEG, TIFF, BMP, WebP |
| 2 | Grayscale conversion | 8-bit single-channel |
| 3 | Denoise | Non-Local Means (`fastNlMeansDenoising`) |
| 4 | Smooth | Bilateral Filter (edge-preserving) |
| 5 | Enhance contrast | CLAHE (Contrast Limited AHE) |
| 6 | AI analysis | Local Ollama multimodal model (default: `llava`) |

---

## Requirements

* Python 3.10 +
* [Ollama](https://ollama.com) running locally with a multimodal model pulled, e.g.:

```bash
ollama pull llava
```

---

## Quick Start

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate          # Windows PowerShell
# or
source .venv/bin/activate       # macOS / Linux

# 2. Run the app
python app.py
```

---

## Usage

1. Click **Open Image** and select your papyrus scan.
2. Click **▶ Process** – the pipeline runs automatically and shows the cleaned image.
3. Click **🔍 AI Analyze** – the processed image is sent to Ollama and the transcription/translation appears on the right.
4. Click **Save Analysis** to save the result to a `.txt` file.
5. Click **⚙ Settings** at any time to tweak processing parameters or AI prompts.

---

## Settings

All parameters are stored in `config.json` and exposed in the **Settings** dialog:

### Processing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| NLM h | 10 | Filter strength – higher = smoother, less detail |
| NLM Template Window | 7 | Patch size for similarity comparison |
| NLM Search Window | 21 | Area searched for similar patches |
| Bilateral d | 9 | Neighbourhood diameter |
| Bilateral Sigma Color | 75 | Color similarity range |
| Bilateral Sigma Space | 75 | Spatial influence range |
| CLAHE Clip Limit | 3.0 | Contrast enhancement ceiling |
| CLAHE Tile Grid | 8×8 | Sub-region size for local histogram |

### AI Settings

| Setting | Description |
|---------|-------------|
| Ollama Model | Model name (must be multimodal, e.g. `llava`, `llava:13b`, `minicpm-v`) |
| Ollama Host | URL of the Ollama API (default `http://localhost:11434`) |
| System Prompt | Scholarly role and instruction prompt sent to the model |
| User Prompt Template | Per-image request sent alongside the image |

---

## Output

When **Save Intermediate Steps** is enabled (default), the following files are written to the `output/` folder:

```
output/
  <filename>_1_raw.png
  <filename>_2_grayscale.png
  <filename>_3_denoised.png
  <filename>_4_smoothed.png
  <filename>_5_enhanced.png
```

---

## Project Structure

```
PHerc. 1667/
├── app.py            Main GUI application
├── processing.py     Image processing pipeline
├── ai_analysis.py    Ollama AI integration
├── config.json       All settings (editable via UI)
├── README.md
└── output/           Generated processed images
```
