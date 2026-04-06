# TRACE v2.0 — Tactical Reconnaissance & Analysis of Coastal Environments

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-OBB-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

> AI-powered satellite intelligence platform for autonomous maritime vessel detection, environmental oil spill monitoring, and dynamic tactical analysis — fully self-hostable, open-source, and multi-modal.

---

## 🌊 What is TRACE?

TRACE transforms raw satellite imagery (or a set of GPS coordinates) into a complete maritime intelligence briefing in seconds — combining computer vision, SAR segmentation, real-time OSINT, and a vision-language model into a single unified pipeline.

Unlike enterprise solutions (Windward, Orbital Insight) that cost $100k+/year and require your data to leave your infrastructure, TRACE runs entirely on-premise — critical for coast guards, navies, and government agencies with data sovereignty requirements.

Input: GPS Coordinates or Satellite Image
  ↓
Output: Vessel detections + GPS tags + Oil spill polygon + Risk Score (0–100) + LLM Tactical Report

---

## 🤖 AI Models

| Model | Task | Architecture | Dataset |
|---|---|---|---|
| YOLOv8-OBB | Vessel detection & orientation | YOLOv8s Oriented Bounding Box | DOTAv1.5 (16 classes) |
| U-Net (ResNet34) | Oil spill segmentation | PyTorch segmentation_models_pytorch | SOS dataset |
| Qwen 2.5-VL-72B | Tactical intelligence report generation | Vision-Language Model | HuggingFace Serverless API |

### Model Performance

| Model | Metric | Score |
|---|---|---|
| YOLOv8-OBB | mAP50 (ship class, DOTAv1.5 test set) | ~0.73 |
| U-Net ResNet34 | IoU (SOS dataset) | ~0.68 |
| U-Net ResNet34 | Dice Score (SOS dataset) | ~0.71 |

### Why These Models?

- YOLOv8-OBB: The only YOLO variant with oriented bounding boxes — essential for oblique satellite viewing angles where axis-aligned boxes fail. Corrects for heading/orientation automatically.
- U-Net + ResNet34: SAR imagery is inherently noisy (backscatter). U-Net's encoder-decoder skip connections preserve spatial detail; ResNet34 encoder (ImageNet pretrained) provides robust feature extraction. Custom 0.5 × BCE + 0.5 × Dice loss handles severe class imbalance (spills < 5% of pixels).
- Qwen 2.5-VL-72B: Top open-weight vision-language model. No vendor lock-in, no per-token OpenAI pricing at scale. Hallucination risk is controlled via deterministic prompt injection — the model receives structured detector outputs and can only reason within the provided context.

---

## 🎯 Key Features

- Multi-Modal Fusion Pipeline — Parallel execution of optical vessel detection (YOLO) + SAR oil spill segmentation (U-Net) + weather/news OSINT, all synthesized into one risk score
- Zero-Prep Auto-Routing — Input GPS coordinates and TRACE automatically fetches and stitches satellite map tiles — no pre-downloaded imagery needed
- Risk Engine (0–100 Score) — Multi-factor scoring: vessel density + weather multiplier (wind > 30 km/h) + pollution penalty (+40 for any spill) + geopolitical news sentiment overlay
- On-Premise Deployable — Self-hosted with no external data dependency beyond open APIs (ESA Sentinel, Open-Meteo)
- Live Fleet Registry — Cross-reference detected vessels against registered friendly fleet database to flag unidentified tracks
- LLM Tactical Reports — Qwen synthesizes all signals into a military-grade written briefing with correlation analysis
- Interactive Dashboard — Leaflet.js map with bounding box overlays, segmentation polygons, and real-time alert feed

---

## 📂 Project Structure
[4/6/2026 7:24 PM] ML/DL: TRACEEE/
├── src/
│   ├── app.py             # Main FastAPI backend & inference orchestration
│   ├── index.html         # Cyberpunk-styled interactive dashboard frontend
│   ├── intelligence.py    # OSINT gathering (Weather, News NLP, Sentinel Hub)
│   ├── risk_engine.py     # Multi-factor risk calculation & Qwen context builder
│   ├── database.py        # SQLite history, alerts, and analysis state management
│   └── fleet.py           # Fleet registry & port management
├── models/
│   ├── best.pt            # YOLOv8s-OBB weights (DOTAv1.5, 16 classes)
│   └── best_unet_sos.pth  # U-Net weights (SOS dataset, BCE+Dice loss)
├── notebooks/
│   ├── yolo8n-dota.ipynb        # YOLO training notebook
│   └── oilspill-segmentation.py # U-Net training pipeline
├── docs/
│   ├── TECHNICAL_DETAILS.md     # Deep-dive architecture & AI documentation
│   └── TRACE_PitchDeck.pdf      # Executive pitch presentation
├── trace.db               # SQLite database (auto-generated on first run)
├── requirements.txt
├── .env.example           # Environment variable template
└── .env                   # Your local config (never commit this)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- ~4GB disk space for model weights
- A free HuggingFace account for Qwen reports

### 1. Install dependencies
pip install -r requirements.txt

### 2. Configure environment
cp .env.example .env

Edit .env with your keys:

# Required — enables Qwen 2.5-VL tactical report generation
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx          # → huggingface.co/settings/tokens (free)

# Optional — enables geopolitical news sentiment analysis
NEWS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx     # → newsapi.org (free tier: 100 req/day)

# No key needed — open APIs used automatically:
# Open-Meteo (weather), OSM tiles (map auto-download)

### 3. Place model weights
models/best.pt             # YOLOv8-OBB weights
models/best_unet_sos.pth   # U-Net weights

### 4. Start the backend
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

### 5. Open the dashboard
http://localhost:8000

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Interactive dashboard UI |
| GET | /health | System diagnostics & loaded model status |
| POST | /process | Core pipeline — accepts image file upload OR lat/lon params for auto-download |
| GET | /api/history | Past analysis sessions from SQLite |
| GET | /api/alerts | Active threat alerts (Risk Score > 75) |
| GET | /api/intel | Quick OSINT snapshot (weather + news) for a coordinate |
| GET | /api/fleet | Registered fleet database |
| GET | /api/route | Maritime route calculation (land-avoidance) |

### Example: Process by coordinates
curl -X POST "http://localhost:8000/process" \
  -F "lat=37.9" \
  -F "lon=23.7"

### Example: Process by image upload
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/satellite_image.jpg"

---

## 🗺 System Architecture
Client Request (/process)
        │
        ├──[Coordinates]──► Download & Stitch Map Tiles ─┐
        │                                                  ▼
        └──[Image Upload]─────────────────────► Image Preprocessing
                                                          │
                        ┌─────────────────────────────────┤
                        ▼                                 ▼
              YOLOv8-OBB Detection           U-Net SAR Segmentation
              (vessel GPS + heading)         (oil spill polygon)
                        │                                 │
                        └──────────────┬──────────────────┘
                                       ▼
                         OSINT Layer (parallel):
                         ├─ Open-Meteo Weather API
                         └─ News NLP Sentiment (TextBlob)
                                       │
                                       ▼
                              Risk Engine (0–100)
                                       │
                                       ▼
                     Qwen 2.5-VL-72B Tactical Report
                                       │
                                       ▼
                          SQLite (trace.db) → JSON Response

---

## 🗄 Data Persistence & Production Path

Current (MVP): SQLite — zero-config, file-based, suitable for local deployment and demos.

Production roadmap:
SQLite (MVP) → PostgreSQL + PostGIS (geospatial queries at scale) → TimescaleDB (time-series vessel tracking)

Database schema:
- analysis_history — full JSON blobs of each pipeline run (detections, OSINT, Qwen report)
- alerts — high-priority alerts triggered by Risk Engine (Risk > 75 → "Critical Incident")
- fleet / ports — known vessel registries for anomaly cross-referencing

---

## 🌍 Target Use Cases

| Sector | Application |
|---|---|
| Coast Guard / Navy | Autonomous patrol coverage, unidentified vessel flagging |
| Maritime Insurance | Real-time risk scoring for vessel underwriting |
| Environmental Agencies | Rapid oil spill detection and drift vector calculation |
| Port Authorities | Predictive threat assessment for incoming traffic |

---

## 🛣 Roadmap

- [x] YOLOv8-OBB vessel detection with GPS tagging
- [x] U-Net SAR oil spill segmentation
- [x] Multi-factor Risk Engine
- [x] Qwen 2.5-VL tactical report generation
- [x] Real-time weather + news OSINT
- [ ] Real-time Sentinel Hub streaming integration
- [ ] AIS transponder anomaly correlation
- [ ] Predictive drift vector modeling for oil spills
- [ ] PostgreSQL + PostGIS migration for production scale
- [ ] Multi-region deployment with load balancing

---

## 📚 Documentation

- [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md) — Deep-dive: model architectures, training hyperparameters, loss functions, OSINT pipeline
- [docs/TRACE_PitchDeck.pdf](docs/TRACE_PitchDeck.pdf) — Executive pitch presentation

---
