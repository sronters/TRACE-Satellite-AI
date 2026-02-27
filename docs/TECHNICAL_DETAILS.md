# TRACE — Technical Documentation

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SATELLITE IMAGE INPUT                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
            ┌─────────▼──────────┐
            │   MODE SELECTION   │
            └──────┬──────┬──────┘
                   │      │
          Optical  │      │  SAR (Radar)
                   │      │
         ┌─────────▼─┐  ┌─▼──────────┐
         │  YOLOv8   │  │   U-Net    │
         │  OBB Det  │  │ Seg (oil)  │
         └─────────┬─┘  └─┬──────────┘
                   └──┬───┘
                      │
             ┌────────▼────────┐
             │  Qwen 2.5-VL   │
             │ Tactical Report │
             └─────────────────┘
```

---

## 2. YOLO — Vessel Detection

**Model:** `YOLOv8s-OBB` (Oriented Bounding Box)

**Dataset:** DOTAv1.5 — Large-scale aerial object detection

| Parameter | Value |
|---|---|
| Architecture | YOLOv8s-OBB |
| Input Size | 1024 × 1024 |
| Epochs | 100 |
| Optimizer | AdamW |
| Learning Rate | 0.001 (cosine decay) |
| Batch Size | 8 |
| Augmentation | mosaic, mixup, fliplr, flipud, rotation ±180° |
| Classes | 16 (ship, plane, vehicle, harbor, bridge, …) |

**Why OBB?** Satellite imagery contains rotated objects. Standard axis-aligned bounding boxes perform poorly on oblique-angle ships. OBB models learn rotation-invariant detection.

**Inference parameters:**
- `conf=0.1` — Low threshold tuned for satellite imagery
- `iou=0.45` — Suppresses overlapping detections
- `imgsz=1024` — Matches training resolution

### GSD-based Measurement

```
length_m = max(bbox_width, bbox_height) × GSD
width_m  = min(bbox_width, bbox_height) × GSD
area_m2  = length_m × width_m

Where GSD = 3.0 m/pixel (approximate for sub-meter satellites)
```

---

## 3. U-Net — Oil Spill Segmentation

**Model:** `segmentation_models_pytorch.Unet` with ResNet34 encoder

**Dataset:** Refined Deep SAR Oil Spill (SOS) Dataset

| Parameter | Value |
|---|---|
| Encoder | ResNet34 (pretrained on ImageNet) |
| Input Channels | 1 (grayscale SAR) |
| Output | Binary mask (1 channel) |
| Input Size | 512 × 512 |
| Epochs | 80 |
| Learning Rate | 2e-4 (Adam) |
| Loss Function | 0.5 × BCE + 0.5 × Dice |
| Batch Size | 8 |

### Preprocessing Pipeline

SAR imagery requires specific normalization (different from optical):

```python
# 1. Convert to grayscale
gray = image.convert("L")

# 2. Min-max normalization (NOT /255)
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

# 3. Resize to 512×512
# 4. Unsqueeze → [1, 1, 512, 512]
```

### Postprocessing

```python
mask = torch.sigmoid(output)      # logits → [0, 1]
binary = (mask > 0.5).float()     # threshold
area_m2 = pixel_count × (GSD²)   # real-world area
```

---

## 4. Mode Switching Logic

```
if mode == "optical":
    → run YOLOv8-OBB
    → U-Net: DISABLED
    → banner: "SAR-segmentation is inactive for optical data"

if mode == "sar":
    → run U-Net (primary)
    → run YOLOv8 with experimental flag (domain shift warning)
    → amber warning banner displayed on all detections
```

---

## 5. Qwen 2.5-VL Integration

**Model:** `Qwen/Qwen2.5-VL-7B-Instruct` via Hugging Face serverless API

**Authentication:** HF token with "Inference Providers" scope

**Prompt template:**
```
System TRACE detected {detections_context}.
Generate a tactical maritime intelligence report with:
1) Threat assessment
2) Vessel profiles and activity analysis
3) Environmental risk evaluation
4) Recommended immediate actions
```

**Fallback chain:**
1. `InferenceClient` auto-routing (huggingface_hub)
2. `router.huggingface.co/hf-inference/...` direct HTTP
3. Local structured report from detection data

---

## 6. Security Architecture

```
[Browser / User] ──── HTTP ──→ [FastAPI :8000]
                                      │
                              HF_TOKEN (env only)
                                      │
                                      └──→ [Hugging Face API]

Token never exposed to browser. All AI calls proxied through backend.
```

---

## 7. Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 inference |
| `segmentation-models-pytorch` | U-Net with ResNet34 |
| `torch` + `torchvision` | Deep learning runtime |
| `huggingface_hub` | Qwen API client |
| `fastapi` + `uvicorn` | REST API server |
| `python-dotenv` | Token management |
| `opencv-python-headless` | Image processing |
| `Pillow` | Image I/O |
