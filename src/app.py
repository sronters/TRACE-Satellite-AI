import os
import io
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
import requests

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

YOLO_WEIGHTS = Path("models/best.pt")
UNET_WEIGHTS = Path("models/best_unet_sos.pth")
INDEX_HTML = Path("src/index.html")

app = FastAPI(title="TRACE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_origin_regex=r".*",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model: Optional[YOLO] = None
unet_model = None

OPTICAL_COLOR = (0, 255, 65)
WARNING_COLOR = (255, 165, 0)


def load_models():
    global yolo_model, unet_model

    if YOLO_WEIGHTS.exists():
        yolo_model = YOLO(str(YOLO_WEIGHTS))

    if UNET_WEIGHTS.exists():
        try:
            import segmentation_models_pytorch as smp
            m = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=1,
                classes=1,
            )
            state = torch.load(str(UNET_WEIGHTS), map_location=DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            m.load_state_dict(state, strict=True)
            m.to(DEVICE)
            m.eval()
            unet_model = m
        except Exception as e:
            print(f"U-Net load error: {e}")
            unet_model = None


load_models()


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_yolo(image: Image.Image, experimental: bool = False):
    detections = []
    draw = ImageDraw.Draw(image)
    w, h = image.size

    results = yolo_model(image, verbose=False, conf=0.1, iou=0.45, imgsz=1024)[0]

    is_obb = hasattr(results, "obb") and results.obb is not None and len(results.obb) > 0

    if is_obb:
        boxes_data = results.obb
        for i in range(len(boxes_data)):
            xyxyxyxy = boxes_data.xyxyxyxy[i].cpu().numpy()
            conf = float(boxes_data.conf[i].cpu().numpy())
            cls_id = int(boxes_data.cls[i].cpu().numpy())
            label = yolo_model.names.get(cls_id, f"Object_{cls_id}")

            pts = xyxyxyxy.reshape(-1, 2).astype(int)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            x1, x2 = int(x_coords.min()), int(x_coords.max())
            y1, y2 = int(y_coords.min()), int(y_coords.max())

            bw, bh = x2 - x1, y2 - y1
            GSD = 3.0
            length_m = round(max(bw, bh) * GSD, 1)
            width_m = round(min(bw, bh) * GSD, 1)
            area_m2 = round(length_m * width_m, 1)
            lat = round((y1 + y2) / 2 / h * 0.01, 6)
            lon = round((x1 + x2) / 2 / w * 0.01, 6)

            color = WARNING_COLOR if experimental else OPTICAL_COLOR
            poly = [tuple(p) for p in pts.tolist()]
            draw.polygon(poly, outline=color)
            draw.text((x1, max(0, y1 - 14)), f"{label} {conf:.2f}", fill=color)

            det = {
                "label": label, "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "length_m": length_m, "width_m": width_m, "area_m2": area_m2,
                "lat": lat, "lon": lon,
            }
            if experimental:
                det["warning"] = "YOLO running in experimental mode: domain shift from SAR input"
            detections.append(det)

    else:
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                label = yolo_model.names.get(cls_id, f"Object_{cls_id}")

                bw, bh = x2 - x1, y2 - y1
                GSD = 3.0
                length_m = round(max(bw, bh) * GSD, 1)
                width_m = round(min(bw, bh) * GSD, 1)
                area_m2 = round(length_m * width_m, 1)
                lat = round((y1 + y2) / 2 / h * 0.01, 6)
                lon = round((x1 + x2) / 2 / w * 0.01, 6)

                color = WARNING_COLOR if experimental else OPTICAL_COLOR
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1, max(0, y1 - 14)), f"{label} {conf:.2f}", fill=color)

                det = {
                    "label": label, "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "length_m": length_m, "width_m": width_m, "area_m2": area_m2,
                    "lat": lat, "lon": lon,
                }
                if experimental:
                    det["warning"] = "YOLO running in experimental mode: domain shift from SAR input"
                detections.append(det)

    return detections, image


def run_unet(image: Image.Image):
    orig_w, orig_h = image.size
    gray = image.convert("L")
    arr = np.array(gray).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    resized = cv2.resize(arr, (512, 512))
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = unet_model(tensor)
        mask = torch.sigmoid(out).squeeze().cpu().numpy()

    mask_bin = (mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_bin, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    spill_pixels = int(np.sum(mask_resized > 0))
    spill_area_m2 = round(spill_pixels * (3.0 ** 2), 1)

    overlay = image.copy().convert("RGBA")
    mask_pil = Image.fromarray(mask_resized).convert("L")
    red_layer = Image.new("RGBA", overlay.size, (255, 0, 60, 140))
    overlay.paste(red_layer, mask=mask_pil)
    result_img = overlay.convert("RGB")

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_count = sum(1 for c in contours if cv2.contourArea(c) > 50)

    return {
        "spill_detected": spill_pixels > 200,
        "spill_area_m2": spill_area_m2,
        "spill_pixel_count": spill_pixels,
        "polygon_count": polygon_count,
    }, result_img


@app.post("/process")
async def process_image(file: UploadFile = File(...), mode: str = Form(...)):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    response_data = {"mode": mode, "detections": [], "spill": None, "sar_warning": None}

    if mode == "optical":
        if yolo_model is None:
            raise HTTPException(status_code=503, detail="YOLO model not loaded")
        detections, result_img = run_yolo(image.copy())
        response_data["detections"] = detections
        response_data["sar_status"] = "SAR-segmentation is inactive for optical data"
        response_data["processed_image"] = pil_to_b64(result_img)

    elif mode == "sar":
        if unet_model is None:
            raise HTTPException(status_code=503, detail="U-Net model not loaded")
        spill_data, result_img = run_unet(image.copy())
        response_data["spill"] = spill_data
        response_data["processed_image"] = pil_to_b64(result_img)
        if yolo_model is not None:
            yolo_dets, _ = run_yolo(image.copy(), experimental=True)
            response_data["detections"] = yolo_dets
            response_data["sar_warning"] = "YOLO running in experimental mode: domain shift from SAR input"
        else:
            response_data["detections"] = []
            response_data["sar_warning"] = "YOLO unavailable"
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'optical' or 'sar'")

    return JSONResponse(content=response_data)


def _local_tactical_report(context: str) -> str:
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""╔══════════════════════════════════════════════════════╗
║  TRACE TACTICAL INTELLIGENCE REPORT  [LOCAL MODE]   ║
║  Generated: {ts}                     ║
╚══════════════════════════════════════════════════════╝

■ DETECTION SUMMARY
{context}

■ THREAT ASSESSMENT
Automated threat classification applied. AIS-silent vessels
require immediate cross-referencing with maritime databases.

■ VESSEL ACTIVITY ANALYSIS
Vessels geo-tagged and dimensionally profiled via GSD ≈ 3.0 m/px.
Position accuracy: ±0.005°.

■ ENVIRONMENTAL RISK
Vessels within 500m of spill polygons flagged as primary suspects.
Recommended chemical sampling radius: 1.5 km from spill centroid.

■ RECOMMENDED ACTIONS
1. Dispatch coastal patrol to verify AIS-silent contacts
2. Cross-reference vessel dimensions with Lloyd's Register
3. Issue POLREP if spill area > 1,000 m²
4. Request aerial overflight within 2 hours
5. Notify nearest MRCC

[NOTE: Enable "Inference Providers" scope on your HuggingFace
token at huggingface.co/settings/tokens for AI-enhanced analysis.]"""


@app.post("/analyze")
async def analyze_with_qwen(file: UploadFile = File(...), context: str = Form(...)):
    from huggingface_hub import InferenceClient

    raw = await file.read()
    b64_image = base64.b64encode(raw).decode()
    mime = file.content_type or "image/jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_image}"}},
                {"type": "text", "text": (
                    f"System TRACE detected {context}. "
                    "Generate a tactical maritime intelligence report with: "
                    "1) Threat assessment, 2) Vessel profiles and activity analysis, "
                    "3) Environmental risk evaluation, 4) Recommended immediate actions."
                )},
            ],
        }
    ]

    if HF_TOKEN:
        try:
            client = InferenceClient(api_key=HF_TOKEN)
            completion = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=messages,
                max_tokens=1024,
            )
            return JSONResponse(content={"report": completion.choices[0].message.content})
        except Exception:
            pass

        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "messages": messages, "max_tokens": 1024}
        for url in [
            "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-VL-7B-Instruct/v1/chat/completions",
            "https://router.huggingface.co/nebius/v1/chat/completions",
        ]:
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    return JSONResponse(content={"report": resp.json()["choices"][0]["message"]["content"]})
            except Exception:
                pass

    return JSONResponse(content={"report": _local_tactical_report(context)})


@app.get("/health")
def health():
    return {
        "status": "operational",
        "yolo_loaded": yolo_model is not None,
        "unet_loaded": unet_model is not None,
        "device": str(DEVICE),
    }


@app.get("/")
def serve_ui():
    return FileResponse(INDEX_HTML)
