<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Space+Mono&size=38&duration=2500&pause=1000&color=38BDF8&center=true&vCenter=true&width=900&lines=TRACE;Tactical+Reconnaissance+and+Analysis;of+Coastal+Environments;AI-powered+Maritime+Intelligence" />
<br>

![Python](https://img.shields.io/badge/python-3.11+-3776ab?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/fastapi-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-ee4c2c?style=for-the-badge\&logo=pytorch\&logoColor=white)
![YOLOv8](https://img.shields.io/badge/yolov8-obb-6c63ff?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/xgboost-orange?style=for-the-badge)
<br><br>
### satellite imagery → ai analysis → maritime intelligence

> not palantir.
>
> probably because this repository is only a few gigabytes smaller.

<br>

<img width="100%" src="https://github.com/user-attachments/assets/4f0fdf7f-cb78-43eb-aa24-a1087f901f18" />

</div>

---

## what is trace?

trace is an ai-powered maritime intelligence platform that transforms satellite imagery and vessel telemetry into actionable insights.

it combines computer vision, geospatial analysis, anomaly detection and language models into a single workflow capable of monitoring maritime activity, detecting environmental incidents and generating intelligence reports.

instead of staring at raw satellite pixels, users receive information they can actually use.

---

## mission

oceans are huge.

humans are bad at watching millions of square kilometers of water.

trace helps turn satellite imagery into something understandable.

---

## capabilities

|                          |                            |
| ------------------------ | -------------------------- |
| 🛰️ **vessel detection** | 🌊 **oil spill detection** |
| optical imagery support  | u-net segmentation         |
| sar imagery support      | spill polygon extraction   |
| vessel localization      | area estimation            |
| geospatial positioning   | environmental monitoring   |
| automatic tile analysis  | map visualization          |

|                                |                             |
| ------------------------------ | --------------------------- |
| 🚨 **ais anomaly detection**   | 🤖 **intelligence reports** |
| impossible speeds              | vessel detections           |
| signal gaps                    | anomaly alerts              |
| abnormal turns                 | environmental context       |
| anchor inconsistencies         | weather conditions          |
| suspicious navigation patterns | risk assessments            |

---

## workflow

```text
satellite imagery
        ↓
 vessel detection
        ↓
 oil spill analysis
        ↓
 ais anomaly detection
        ↓
 risk assessment
        ↓
 intelligence report
```

---

## some numbers

| metric                | value  |
| --------------------- | ------ |
| ais messages analyzed | 7.1m+  |
| anomaly recall        | 99.95% |
| anomaly precision     | 98.13% |
| roc-auc               | 0.9999 |
| oil spill iou         | 79%    |
| generated reports     | lots   |

---

## tech stack

### ai

```text
yolov8-obb
u-net resnet34
xgboost
qwen 2.5 vl
```

### backend

```text
python
fastapi
sqlite
```

### frontend

```text
vanilla javascript
leaflet
html
css
```

### geospatial

```text
sentinel hub
esri imagery
openstreetmap
```

---

## architecture

```text
satellite imagery
        │
        ├─────────────► yolov8
        │
        ├─────────────► u-net
        │
        ▼

     risk engine
        ▲
        │

    xgboost
        ▲
        │

     ais data

        │
        ▼

 intelligence report
```

---

## quick start

clone repository

```bash
git clone https://github.com/yourusername/trace.git
cd trace
```

create virtual environment

```bash
python -m venv venv
```

activate environment

```bash
source venv/bin/activate
```

install dependencies

```bash
pip install -r requirements.txt
```

run application

```bash
uvicorn src.app:app --reload
```

open

```text
http://localhost:8000
```

---

## project structure

```text
trace/

├── src/
│   ├── app.py
│   ├── intelligence.py
│   ├── anomaly_scorer.py
│   ├── risk_engine.py
│   ├── database.py
│   └── index.html
│
├── models/
│   ├── best.pt
│   ├── best_unet.pth
│   └── anomaly_xgb.json
│
├── notebooks/
├── docs/
├── requirements.txt
└── README.md
```

---

## roadmap

### current

* vessel detection
* oil spill segmentation
* anomaly detection
* intelligence reports

### next

* vessel tracking
* route prediction
* multi-image analysis
* realtime monitoring

### future

* global monitoring
* sar-first workflows
* fleet intelligence dashboard
* autonomous intelligence pipelines

---

## disclaimer

> trace is a research and educational project.
>
> it is not intended for operational military use.
>
> probably.
---
</div>
