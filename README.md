# TRACE

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=24&pause=1000&color=2ED3A2&center=true&vCenter=true&width=900&lines=TRACE+%E2%80%94+Maritime+Intelligence+Platform;AIS+Correlation+%2B+Temporal+Forecasting+%2B+Risk+Scoring;FastAPI+%2B+CV+%2B+OSINT+%2B+Local%2FCloud+LLM" alt="TRACE animated banner" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white">
  <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white">
  <img alt="Prometheus" src="https://img.shields.io/badge/Prometheus-Enabled-E6522C?logo=prometheus&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-FCC624">
</p>

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXl5bGw5eXk2NzdmM2VvNTRld2V3NGE4M2YwbWQ5d3VrbjNwY2M1NyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7TKtnuHOHHUjR38Y/giphy.gif" width="780" alt="Satellite map animation" />
</p>

Production-grade maritime intelligence API for vessel detection, AIS identity correlation, oil-spill analysis, temporal forecasting, and explainable risk scoring.

---

## Why TRACE

- CV + OSINT + AIS fused into one operational decision stream
- Local-first architecture with optional cloud fallback
- Stateful tracking and trajectory prediction across frames
- Contract-driven ML runtime for safe model upgrades
- Scalable persistence path from SQLite to PostGIS + TimescaleDB

---

## Feature Highlights

- YOLO vessel detection and SAR spill segmentation
- MarineTraffic/AISHub/static AIS provider failover
- AIS-silent vessel flagging and fleet enrichment
- Kalman tracking and temporal movement projection
- Drift forecast for oil spills using wind/current signals
- Change/anomaly detection against AOI history
- Hybrid risk engine (rule-based + optional XGBoost)
- SSE streaming endpoint for real-time pipeline progress
- API key security, optional mTLS gate, audit logging
- Prometheus metrics and health/contract diagnostics

---

## Architecture

```mermaid
graph TD
    A[Client] --> B[FastAPI API]
    B --> C[Pipeline Orchestrator]

    C --> D[Tile Ingestion]
    C --> E[Vision Inference]
    C --> F[OSINT Enrichment]
    C --> G[AIS Provider Layer]
    C --> H[Tracking + Temporal]
    C --> I[Risk Engine + ML]
    C --> J[LLM Report]
    C --> K[Persistence]

    G --> G1[MarineTraffic]
    G --> G2[AISHub]
    G --> G3[Static Cache]

    K --> K1[SQLite]
    K --> K2[PostgreSQL + PostGIS + TimescaleDB]
```

---

## Project Layout

```text
TRACEEE/
├── docs/
│   ├── API_REFERENCE.md
│   ├── OPERATIONS.md
│   ├── TECHNICAL_DETAILS.md
│   ├── TRAINING_DATA_GUIDE.md
│   └── V3_UPGRADE.md
├── models/
│   ├── risk_features.json
│   ├── risk_model_meta.json
│   ├── best.pt
│   └── best_unet_sos.pth
├── src/
│   ├── app.py
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   └── http_middleware.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── pipeline_service.py
│   │   ├── tile_service.py
│   │   └── vision_service.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── artifact_validator.py
│   │   └── feature_contract.py
│   ├── ais.py
│   ├── intelligence.py
│   ├── tracker.py
│   ├── temporal.py
│   ├── change_detection.py
│   ├── risk_engine.py
│   ├── ml_risk.py
│   ├── database.py
│   ├── fleet.py
│   ├── segmentation.py
│   └── index.html
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Configure

```bash
cp .env.example .env
```

Set at minimum:

```env
TRACE_LLM_PROVIDER=auto
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

Optional:

```env
HF_TOKEN=
TRACE_API_KEYS=
TRACE_REQUIRE_MTLS=false
TRACE_POSTGRES_DSN=
MARINETRAFFIC_API_KEY=
MARINETRAFFIC_API_URL=
RISK_MODEL_PATH=models/risk_xgb.json
RISK_FEATURES_PATH=models/risk_features.json
RISK_MODEL_META_PATH=models/risk_model_meta.json
```

### 3) Run

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- API: `http://localhost:8000`
- Health: `http://localhost:8000/health`
- Metrics: `http://localhost:8000/metrics`

---

## API Snapshot

- `POST /process` full synchronous analysis
- `POST /process/stream` SSE stage-by-stage stream
- `GET /api/ml/contract` ML artifact contract status
- `GET /api/history` recent runs
- `GET /api/tracks` active temporal tracks
- `GET /api/intel` OSINT snapshot
- `GET /api/fleet`, `GET /api/ports` fleet registry

See full reference in `docs/API_REFERENCE.md`.

---

## ML Integration Contract

TRACE expects:
- `models/risk_xgb.json`
- `models/risk_features.json`
- `models/risk_model_meta.json`

Validation endpoints:
- `GET /api/ml/contract`
- `GET /health` (`ml_contract` section)

This guarantees feature ordering and safe runtime behavior before loading your trained weights.

---

## Documentation Index

- Technical architecture: `docs/TECHNICAL_DETAILS.md`
- API specification: `docs/API_REFERENCE.md`
- Deployment and operations: `docs/OPERATIONS.md`
- Training data and labeling: `docs/TRAINING_DATA_GUIDE.md`
- Upgrade notes: `docs/V3_UPGRADE.md`

---

## Deployment Modes

- Local demo: SQLite + local LLM
- Hybrid: SQLite + external AIS/OSINT + HF fallback
- Production: PostgreSQL/PostGIS/Timescale dual-write + secured API + monitoring

---

## License

MIT
