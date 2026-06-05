<div align="center">
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=38&duration=2500&pause=1000&color=00E676&center=true&vCenter=true&width=900&lines=TRACE;Tactical+Reconnaissance+and+Analysis;of+Coastal+Environments;AI-powered+Maritime+Intelligence" />
<br>
<br>

satellite imagery → ai analysis → maritime intelligence

not palantir.

probably because this repository is only a few gigabytes smaller.

<br>
<img src="docs/demo.gif" width="100%"/>
</div>

⸻

what is trace?

trace is an experimental maritime intelligence platform that combines satellite imagery, vessel detection, oil spill monitoring, ais anomaly detection and automated reporting.

the goal is simple:

turn thousands of square kilometers of ocean into something a human can actually understand.

instead of staring at raw satellite imagery, operators receive:

* detected vessels
* suspicious movement alerts
* potential oil spills
* intelligence summaries
* risk assessments

⸻

mission

oceans are huge.

humans are bad at watching millions of square kilometers of water.

trace helps convert satellite pixels and vessel signals into actionable intelligence.

⸻

what it can do

<table>
<tr>
<td width="50%">

vessel detection

* yolov8 oriented bounding boxes
* optical satellite imagery
* sar imagery support
* vessel localization
* automatic georeferencing

</td>
<td width="50%">

oil spill detection

* u-net segmentation
* spill polygon extraction
* area estimation
* map visualization
* environmental monitoring

</td>
</tr>
<tr>
<td width="50%">

ais anomaly detection

* impossible speed detection
* signal gap detection
* suspicious maneuver detection
* anchor movement anomalies
* xgboost scoring engine

</td>
<td width="50%">

intelligence reports

* ai generated summaries
* weather integration
* regional context
* risk scoring
* operational recommendations

</td>
</tr>
</table>

⸻

how it works

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

⸻

demo

<p align="center">
<img src="docs/dashboard.gif" width="100%">
</p>

⸻

some numbers

metric	value
ais messages analyzed	7.1m+
anomaly recall	99.95%
anomaly precision	98.13%
roc auc	0.9999
oil spill iou	79%
vessel detections	thousands

⸻

screenshots

command center

<img src="docs/screenshots/dashboard.png">

⸻

vessel detection

<img src="docs/screenshots/vessels.png">

⸻

oil spill segmentation

<img src="docs/screenshots/oilspill.png">

⸻

intelligence report

<img src="docs/screenshots/report.png">

⸻

tech stack

ai

* yolov8-obb
* u-net resnet34
* xgboost
* qwen 2.5 vl

backend

* python
* fastapi
* sqlite

frontend

* vanilla javascript
* leaflet
* html
* css

geospatial

* sentinel hub
* esri imagery
* openstreetmap

⸻

architecture

graph LR
A[Satellite Imagery]
--> B[YOLOv8]
A
--> C[U-Net]
B
--> D[Risk Engine]
C
--> D
E[AIS Data]
--> F[XGBoost]
F
--> D
D
--> G[AI Report]

⸻

project structure

trace/
├── src/
│   ├── app.py
│   ├── anomaly_scorer.py
│   ├── intelligence.py
│   ├── risk_engine.py
│   └── database.py
│
├── models/
│   ├── best.pt
│   ├── best_unet.pth
│   └── anomaly_xgb.json
│
├── docs/
│
├── notebooks/
│
└── readme.md

⸻

quick start

clone repository

git clone https://github.com/yourname/trace.git
cd trace

create virtual environment

python -m venv venv

activate environment

source venv/bin/activate

install dependencies

pip install -r requirements.txt

run server

uvicorn src.app:app --reload

open browser

http://localhost:8000

⸻

roadmap

current

* vessel detection
* oil spill segmentation
* anomaly detection
* intelligence reports

next

* vessel tracking
* route prediction
* multi-image analysis
* realtime monitoring
* fleet intelligence dashboard

future

* sar-first pipeline
* maritime threat assessment
* global monitoring
* autonomous intelligence workflows

⸻

disclaimer

trace is a research and educational project.

it is not intended for operational military use.

probably.

⸻

<div align="center">

built with coffee, satellite imagery and questionable sleep schedules

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=14&duration=4000&pause=1000&color=00E676&center=true&vCenter=true&width=600&lines=detect+vessels.;find+oil+spills.;catch+anomalies.;generate+reports.;repeat." />
</div>