# TRACE — Tactical Reconnaissance & Analysis of Coastal Environments

> AI-powered satellite intelligence platform for autonomous maritime vessel detection and environmental oil spill monitoring.

---

## Overview

TRACE transforms raw satellite imagery into actionable maritime intelligence using two specialized AI models trained from scratch:

- **YOLOv8-OBB** — Optical-band vessel detection with automatic orientation-aware classification, dimensional measurement, and GPS tagging
- **U-Net (ResNet34)** — SAR-band oil spill segmentation with area calculation and polygon mapping
- **Qwen 2.5-VL** — Vision-language model generating tactical intelligence reports via Hugging Face serverless API

---

## Project Structure

```
TRACEEE/
├── src/
│   ├── app.py          # FastAPI backend
│   └── index.html      # Cyberpunk dashboard frontend
├── models/
│   ├── best.pt         # YOLOv8-OBB weights (DOTA v1.5 trained)
│   └── best_unet_sos.pth  # U-Net weights (SOS dataset trained)
├── notebooks/
│   ├── yolo8n-dota.ipynb         # YOLO training notebook
│   └── oilspill-segmentation.ipynb  # U-Net training notebook
├── docs/
│   ├── TECHNICAL_DETAILS.md   # Architecture documentation
│   └── TRACE_PitchDeck.pdf    # Project presentation
├── requirements.txt
├── .env                # HF Token (gitignored)
├── .env.example        # Token template
└── .gitignore
```

---

## Technologies

| Component | Technology |
|---|---|
| Object Detection | YOLOv8-OBB (Ultralytics) — DOTA v1.5 |
| Oil Spill Segmentation | U-Net + ResNet34 encoder (smp) — SOS Dataset |
| Tactical Analysis | Qwen/Qwen2.5-VL-7B-Instruct (Hugging Face) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML5 / CSS3 / Vanilla JS — Cyberpunk UI |

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env — set your HF_TOKEN
```

**3. Start backend**
```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

**4. Open UI**

Navigate to → **http://localhost:8000**

---

## Analysis Modes

| Mode | YOLO | U-Net | Qwen |
|---|---|---|---|
| **Optical** | Active | Inactive | Active |
| **SAR (Radar)** | Experimental ⚠ | Active | Active |

- **Optical**: `SAR-segmentation is inactive for optical data`
- **SAR**: YOLO runs with domain-shift warning. U-Net produces oil spill masks.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Frontend dashboard |
| `GET` | `/health` | Backend + model status |
| `POST` | `/process` | Image analysis (`mode`: optical/sar) |
| `POST` | `/analyze` | Qwen 2.5-VL tactical report proxy |

---

## Model Training

The `notebooks/` folder contains the original training notebooks:

- **`yolo8n-dota.ipynb`** — YOLOv8s-OBB trained on DOTAv1.5 (16 classes, `imgsz=1024`, 100 epochs, AdamW)
- **`oilspill-segmentation.ipynb`** — U-Net (ResNet34 encoder) trained on the SOS SAR dataset (80 epochs, BCE+Dice loss)

---

## Target Users

- **Port authorities** — vessel traffic control and berth management
- **Environmental agencies** — pollution incident documentation
- **Coast guard** — ghost vessel (AIS-dark) detection
- **Insurance companies** — accident and damage assessment
- **Logistics operators** — port congestion analysis
