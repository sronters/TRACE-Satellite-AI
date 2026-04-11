"""
TRACE v2.0 — FastAPI Backend
Full upgrade: Intelligence Layer + Risk Engine + Database + enhanced Qwen
"""

import os
import sys
import io
import math
import json
import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import httpx

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

                                                                                
load_dotenv(Path(__file__).parent.parent / ".env")

                                                                                
sys.path.insert(0, str(Path(__file__).parent))
from intelligence import gather_intelligence                                                    
from risk_engine import calculate_risk, build_qwen_context
from database import init_db, save_analysis, get_history, get_analysis, get_alerts, get_stats, get_vessel_heatmap, acknowledge_alert
from fleet import init_fleet_tables, enrich_detections_with_fleet, get_nearby_port, get_fleet, get_ports, add_port, import_fleet_from_json, import_fleet_from_csv, delete_vessel, delete_port

                                                                                
HF_TOKEN = os.getenv("HF_TOKEN", "")

                                                                                
YOLO_WEIGHTS = Path("models/best.pt")
UNET_WEIGHTS = Path("models/best_unet_sos.pth")
INDEX_HTML   = Path("src/index.html")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = None
unet_model = None

OPTICAL_COLOR = (0, 255, 65)
WARNING_COLOR = (255, 165, 0)


def load_models():
    global yolo_model, unet_model

    if YOLO_WEIGHTS.exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(YOLO_WEIGHTS))
            print(f"[TRACE] YOLOv8-OBB loaded [OK] ({DEVICE})")
        except Exception as e:
            print(f"[TRACE] YOLO load error: {e}")

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
            print(f"[TRACE] U-Net loaded [OK] ({DEVICE})")
        except Exception as e:
            print(f"[TRACE] U-Net load error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_fleet_tables()
    load_models()
    print("[TRACE] v2.0 ready [OK]")
    yield


app = FastAPI(title="TRACE v2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


def _lat_lon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to WMTS tile xy."""
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def _tile_to_lat_lon(tx: int, ty: int, zoom: int) -> Tuple[float, float]:
    """Convert tile xy to lat/lon of the top-left corner."""
    n = 2 ** zoom
    lon = tx / n * 360 - 180
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    lat = math.degrees(lat_r)
    return lat, lon


async def download_area_image(
    lat: float,
    lon: float,
    zoom: int = 15,
    radius: int = 2,
) -> Tuple[Optional[bytes], Optional[Tuple[float, float, float, float]]]:
    """
    Download ESRI satellite tiles around given coordinates, stitch into one image.
    Returns (jpeg_bytes, (lat_min, lat_max, lon_min, lon_max)) or (None, None).
    zoom=15: GSD ≈ 4.4m/px per tile → 5x5 grid ~5.6km², 2000px resolution (ideal for vessel detection)
    zoom=14: GSD ≈ 8.8m/px per tile → 5x5 grid ~22km², good for wide area
    """
    cx, cy = _lat_lon_to_tile(lat, lon, zoom)
    tile_size = 256
    grid = radius * 2 + 1

    url_template = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )

    coords = []
    urls = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            tx, ty = cx + dx, cy + dy
            coords.append((dx + radius, dy + radius, tx, ty))
            urls.append(url_template.format(z=zoom, y=ty, x=tx))

    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            responses = await asyncio.gather(
                *[client.get(u) for u in urls],
                return_exceptions=True,
            )
    except Exception as e:
        print(f"[TRACE] Tile download error: {e}")
        return None, None

    canvas = Image.new("RGB", (tile_size * grid, tile_size * grid), color=(5, 10, 20))
    for (dx, dy, tx, ty), resp in zip(coords, responses):
        if isinstance(resp, Exception) or resp.status_code != 200:
            continue
        try:
            tile_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            canvas.paste(tile_img, (dx * tile_size, dy * tile_size))
        except Exception:
            continue

                                                
    top_lat, left_lon = _tile_to_lat_lon(cx - radius, cy - radius, zoom)
    bot_lat, right_lon = _tile_to_lat_lon(cx + radius + 1, cy + radius + 1, zoom)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    print(f"[TRACE] Tiles stitched: {grid}x{grid} @ zoom{zoom} → {canvas.size} px | bbox {bot_lat:.4f},{top_lat:.4f},{left_lon:.4f},{right_lon:.4f}")
    return buf.getvalue(), (bot_lat, top_lat, left_lon, right_lon)


@app.get("/", response_class=HTMLResponse)
async def root():
    if INDEX_HTML.exists():
        return INDEX_HTML.read_text(encoding="utf-8")
    return "<h1>TRACE v2.0</h1><p>Place index.html in src/</p>"


@app.get("/health")
async def health():
    return {
        "status": "online",
        "yolo": yolo_model is not None,
        "unet": unet_model is not None,
        "device": str(DEVICE),
        "version": "2.0",
        "db": True,
    }


@app.post("/process")
async def process(
    file: Optional[UploadFile] = File(None),
    mode: str = Form("optical"),
    lat: float = Form(0.0),
    lon: float = Form(0.0),
    use_sentinel: bool = Form(False),
    zoom: int = Form(15),
):
    """
    Core analysis endpoint.
    Accepts image upload OR coordinates (auto-downloads satellite tiles).
    Returns detections + risk + intel + qwen report in one response.
    """
                        
    img_bytes = None
    bbox = None                                        

    if file and file.filename:
        img_bytes = await file.read()
                                                
        if lat or lon:
            delta = 0.05
            bbox = (lat - delta, lat + delta, lon - delta, lon + delta)

    elif lat != 0.0 or lon != 0.0:
                                                                 
        print(f"[TRACE] Downloading tiles for {lat:.4f},{lon:.4f} zoom={zoom}")
        img_bytes, bbox = await download_area_image(lat, lon, zoom=zoom, radius=2)
        if not img_bytes:
            raise HTTPException(503, "Failed to download satellite tiles. Check internet connection.")
        use_sentinel = True
    else:
        raise HTTPException(400, "Provide an image file or lat/lon coordinates.")

                                                             
    has_coords = (lat != 0.0 or lon != 0.0)
    intel_task = asyncio.create_task(
        gather_intelligence(lat, lon, mode) if has_coords else _empty_intel(lat, lon)
    )

                                
    detections = {"vessels": [], "oil_spill_area_m2": 0, "oil_polygons": []}
    processed_img_b64 = None

    if img_bytes:
        detections, processed_img_b64 = await asyncio.to_thread(
            _run_models, img_bytes, mode, lat, lon, bbox
        )

                    
    intel = await intel_task

                       
    risk = calculate_risk(
        vessels=detections.get("vessels", []),
        oil_spill_area_m2=detections.get("oil_spill_area_m2"),
        weather=intel.get("weather", {}),
        mode=mode,
        news=intel.get("news", []),
    )

                                            
    if detections.get("vessels"):
        detections["vessels"] = enrich_detections_with_fleet(detections["vessels"])
    nearby_port = get_nearby_port(lat, lon) if (lat or lon) else None

                                   
    qwen_context = build_qwen_context(intel, risk, detections)

                        
    qwen_report = await _call_qwen(qwen_context, processed_img_b64)

                   
    analysis_id = await asyncio.to_thread(
        save_analysis, lat, lon, mode, detections, risk, intel, qwen_report
    )

    return JSONResponse({
        "analysis_id": analysis_id,
        "detections": detections,
        "risk": risk.to_dict(),
        "intel": intel,
        "nearby_port": nearby_port,
        "qwen_report": qwen_report,
        "processed_image": processed_img_b64,
        "image_bbox": list(bbox) if bbox else None,                                        
        "timestamp": datetime.utcnow().isoformat(),
    })


def _create_water_mask(img_np: np.ndarray, mode: str = "optical") -> np.ndarray:
    """
    Improved water mask using spectral ratios.
    Optical: NDWI-like (blue dominance over red), + large-blob filter to reject fields.
    SAR:     adaptive threshold on inverted backscatter.
    """
    if mode == "sar":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                                                        
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
                                                                          
        thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, water = cv2.threshold(blur, thresh * 0.8, 255, cv2.THRESH_BINARY_INV)
    else:
        r = img_np[:, :, 0].astype(np.float32)
        g = img_np[:, :, 1].astype(np.float32)
        b = img_np[:, :, 2].astype(np.float32)
        eps = 1.0

                                                  
        bdr = (b - r) / (b + r + eps)                              
                                                          
        brightness = (r + g + b) / 3.0

                                                         
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h_ch = hsv[:, :, 0]

                                                         
        is_blue_dominant = bdr > 0.05                               
        is_not_green_land = (g < b * 1.3)                                 
        is_dark_enough = brightness < 180                                   
        is_water_hue = (h_ch >= 80) & (h_ch <= 140)                         

                                                                    
        is_deep_ocean = (brightness < 40) & ((b - r) > -5)

        water = ((is_blue_dominant & is_not_green_land & is_dark_enough) |
                 is_water_hue | is_deep_ocean).astype(np.uint8) * 255

                                                                              
    kernel_close = np.ones((31, 31), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_close)
                                                                    
    kernel_open = np.ones((12, 12), np.uint8)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, kernel_open)
                                                            
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(water, connectivity=8)
    min_area = water.size * 0.005
    filtered = np.zeros_like(water)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered


@app.get("/api/fleet")
async def api_fleet():
    return get_fleet()

@app.post("/api/fleet/import/json")
async def api_fleet_import_json(file: UploadFile = File(...)):
    """Upload JSON file: [{name, mmsi, length_m, vessel_type, flag, home_port}, ...]"""
    content = await file.read()
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            data = [data]
        count = import_fleet_from_json(data)
        return {"imported": count}
    except Exception as e:
        raise HTTPException(400, f"JSON parse error: {e}")

@app.post("/api/fleet/import/csv")
async def api_fleet_import_csv(file: UploadFile = File(...)):
    """Upload CSV. Required column: name. Optional: mmsi, length_m, vessel_type, flag, home_port."""
    content = await file.read()
    try:
        count = import_fleet_from_csv(content.decode("utf-8", errors="replace"))
        return {"imported": count}
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")

@app.delete("/api/fleet/{vessel_id}")
async def api_fleet_delete(vessel_id: int):
    delete_vessel(vessel_id)
    return {"deleted": vessel_id}

@app.get("/api/ports")
async def api_ports():
    return get_ports()

@app.post("/api/ports")
async def api_add_port(
    name: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    country: str = Form(""),
    radius_km: float = Form(5.0),
    is_home: bool = Form(False),
    notes: str = Form(""),
):
    port_id = add_port(name, lat, lon, country, radius_km, is_home, notes)
    return {"id": port_id, "name": name}

@app.delete("/api/ports/{port_id}")
async def api_port_delete(port_id: int):
    delete_port(port_id)
    return {"deleted": port_id}

@app.get("/api/route")
async def api_route(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float):
    """Calculate maritime route between two points avoiding land using searoute."""
    try:
        import searoute as sr
        origin = [origin_lon, origin_lat]
        dest = [dest_lon, dest_lat]
        route = sr.searoute(origin, dest)
        return route
    except ImportError:
        return {
            "type": "Feature",
            "properties": {"length": 0},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [origin_lon, origin_lat],
                    [dest_lon, dest_lat]
                ]
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Routing error: {e}")


@app.post("/api/anomaly/score")
async def api_anomaly_score(data: dict):
    """
    Binary anomaly scoring for a single AIS ping.

    Required body fields:  lat, lon, sog
    Optional current ping: mmsi, heading, status, cog, vessel_type, length, width, draft, cargo
    Optional previous ping: prev_lat, prev_lon, prev_heading, prev_dt_min (minutes ago)
    """
    try:
        from anomaly_scorer import score_ais_record                                   
        from datetime import datetime, timedelta

        current: dict = {
            "MMSI":       data.get("mmsi"),
            "LAT":        data.get("lat"),
            "LON":        data.get("lon"),
            "SOG":        data.get("sog"),
            "COG":        data.get("cog"),
            "Heading":    data.get("heading"),
            "Status":     data.get("status", 0),
            "VesselType": data.get("vessel_type"),
            "Length":     data.get("length"),
            "Width":      data.get("width"),
            "Draft":      data.get("draft"),
            "Cargo":      data.get("cargo"),
        }

        prev = None
        if data.get("prev_lat") is not None:
            prev = {
                "LAT":     data.get("prev_lat"),
                "LON":     data.get("prev_lon"),
                "Heading": data.get("prev_heading"),
            }
                                                                                
            dt_min = data.get("prev_dt_min")
            if dt_min is not None:
                now = datetime.utcnow()
                current["BaseDateTime"] = now.isoformat()
                prev["BaseDateTime"] = (now - timedelta(minutes=float(dt_min))).isoformat()

        result = await asyncio.to_thread(score_ais_record, current, prev)

        return {
            "is_anomaly":  result["is_anomaly"],
            "probability": round(result["probability"], 4),
            "label":       "ANOMALY" if result["is_anomaly"] else "NORMAL",
        }

    except ImportError as exc:
        raise HTTPException(503, f"Anomaly scorer unavailable (install xgboost + scikit-learn): {exc}")
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model file not found: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Scoring error: {exc}")


def _run_models(
    img_bytes: bytes,
    mode: str,
    lat: float,
    lon: float,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> tuple:
    """
    Run YOLO + UNet. Returns (detections_dict, base64_annotated_image).
    bbox = (lat_min, lat_max, lon_min, lon_max) for accurate GPS from pixels.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    h, w = img.height, img.width
    GSD = 3.0                            

                                                  
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        R = 6_371_000
        dlat = math.radians(lat_max - lat_min)
        dlon = math.radians(lon_max - lon_min)
        lat_c = math.radians((lat_min + lat_max) / 2)
        height_m = R * dlat
        width_m  = R * dlon * math.cos(lat_c)
        GSD_y = height_m / h
        GSD_x = width_m  / w
        GSD = (GSD_x + GSD_y) / 2
        print(f"[TRACE] Image {w}x{h}px | GSD {GSD:.1f}m/px | area {width_m/1000:.1f}x{height_m/1000:.1f}km")

    def pixel_to_gps(px, py):
        """Convert pixel (px, py) to GPS using bbox."""
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            vessel_lat = lat_max - (py / h) * (lat_max - lat_min)
            vessel_lon = lon_min + (px / w) * (lon_max - lon_min)
            return round(vessel_lat, 6), round(vessel_lon, 6)
        elif lat or lon:
            vessel_lat = lat + (0.5 - py / h) * 0.05
            vessel_lon = lon + (px / w - 0.5) * 0.05
            return round(vessel_lat, 6), round(vessel_lon, 6)
        return None, None

    detections = {"vessels": [], "oil_spill_area_m2": 0, "oil_polygons": []}
    img_np = np.array(img)
    annotated = img_np.copy()

                                                                                  
    water_mask = _create_water_mask(img_np, mode)
    water_coverage = np.sum(water_mask > 0) / water_mask.size
    print(f"[TRACE] Water coverage: {water_coverage:.1%}")
    water_tint = annotated.copy()
    water_tint[water_mask > 0] = (water_tint[water_mask > 0] * 0.88 + np.array([0, 20, 40]) * 0.12).astype(np.uint8)
    annotated = water_tint

                                                                                
    if yolo_model and mode in ("optical", "sar", "dual"):
        experimental = (mode == "sar")
        results = yolo_model(img, verbose=False, conf=0.1, iou=0.45, imgsz=1024)
        result = results[0]

        def is_on_water(px, py, margin: int = 15) -> bool:
            y1 = max(0, py - margin); y2 = min(h, py + margin)
            x1 = max(0, px - margin); x2 = min(w, px + margin)
            roi = water_mask[y1:y2, x1:x2]
            return float(np.mean(roi)) > 30

        is_obb = hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0

        if is_obb:
            for i in range(len(result.obb)):
                xyxyxyxy = result.obb.xyxyxyxy[i].cpu().numpy()
                conf = float(result.obb.conf[i].cpu().numpy())
                cls_id = int(result.obb.cls[i].cpu().numpy())
                label = yolo_model.names.get(cls_id, "ship")

                pts = xyxyxyxy.reshape(-1, 2).astype(int)
                xc = float(pts[:, 0].mean())
                yc = float(pts[:, 1].mean())
                bw = float(pts[:, 0].max() - pts[:, 0].min())
                bh = float(pts[:, 1].max() - pts[:, 1].min())
                length_m = max(bw, bh) * GSD
                width_m  = min(bw, bh) * GSD

                if not is_on_water(int(xc), int(yc)):
                    continue

                vessel_lat, vessel_lon = pixel_to_gps(xc, yc)

                color = WARNING_COLOR if experimental else OPTICAL_COLOR
                cv2.polylines(annotated, [pts], True, color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}",
                            (int(pts[:, 0].min()), max(0, int(pts[:, 1].min()) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                detections["vessels"].append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "length_m": round(length_m, 1),
                    "width_m": round(width_m, 1),
                    "area_m2": round(length_m * width_m, 1),
                    "pixel": {"x": round(xc), "y": round(yc)},
                    "gps": {"lat": vessel_lat, "lon": vessel_lon},
                })
        else:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    label = yolo_model.names.get(cls_id, "ship")
                    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                    bw, bh = x2 - x1, y2 - y1
                    length_m = max(bw, bh) * GSD
                    width_m  = min(bw, bh) * GSD

                    if not is_on_water(int(xc), int(yc)):
                        continue

                    vessel_lat, vessel_lon = pixel_to_gps(xc, yc)

                    color = WARNING_COLOR if experimental else OPTICAL_COLOR
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"{label} {conf:.2f}",
                                (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    detections["vessels"].append({
                        "class": label,
                        "confidence": round(conf, 3),
                        "length_m": round(length_m, 1),
                        "width_m": round(width_m, 1),
                        "area_m2": round(length_m * width_m, 1),
                        "pixel": {"x": round(xc), "y": round(yc)},
                        "gps": {"lat": vessel_lat, "lon": vessel_lon},
                    })

                                                                                
    if unet_model and mode in ("sar", "dual"):
        gray = Image.fromarray(img_np).convert("L")
        arr = np.array(gray.resize((512, 512)), dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = unet_model(tensor)
            mask = torch.sigmoid(out).squeeze().cpu().numpy()

        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        mask_full = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        pixel_count = int(np.sum(mask_full > 0))
        area_m2 = pixel_count * (GSD ** 2)
        detections["oil_spill_area_m2"] = round(area_m2, 1)

        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                polygons.append(cnt.reshape(-1, 2).tolist())
        detections["oil_polygons"] = polygons

        overlay = annotated.copy()
        overlay[mask_full > 0] = [255, 50, 50]
        annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

                                                                                
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return detections, b64


async def _call_qwen(context: str, img_b64: Optional[str] = None) -> str:
    """Call Qwen2.5-VL-72B-Instruct via HuggingFace Inference API (HF PRO required)."""
    if not HF_TOKEN:
        return _fallback_report(context, "HF_TOKEN not set in .env")

    MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

                                                                            
    safe_img_b64 = None
    if img_b64:
        try:
            raw = base64.b64decode(img_b64)
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            max_side = 1024
            if max(pil.size) > max_side:
                ratio = max_side / max(pil.size)
                new_size = (int(pil.width * ratio), int(pil.height * ratio))
                pil = pil.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=82)
            safe_img_b64 = base64.b64encode(buf.getvalue()).decode()
            size_kb = len(safe_img_b64) * 3 / 4 / 1024
            print(f"[TRACE] Image for Qwen: {pil.size[0]}x{pil.size[1]}px | ~{size_kb:.0f}KB")
        except Exception as e:
            print(f"[TRACE] Image prep failed, sending text only: {e}")
            safe_img_b64 = None

                                                                             
    user_content = []
    if safe_img_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{safe_img_b64}"
            },
        })
    user_content.append({
        "type": "text",
        "text": context,
    })

    messages = [
        {
            "role": "system",
            "content": (
                "You are TRACE — a professional maritime intelligence analyst AI. "
                "You are given a satellite image and structured detection data. "
                "Cross-reference what you visually observe in the image with the "
                "provided detection metrics. Write a structured tactical report: "
                "threat assessment, vessel profiles, environmental risk, and recommended actions. "
                "Be concise, specific, and professional. Always cite real numbers from the context."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

                                                                             
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.3,
                },
            )

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]

            error_msg = f"HTTP {r.status_code}: {r.text[:300]}"
            print(f"[TRACE] Qwen72B-VL failed: {error_msg}")
            return _fallback_report(context, error_msg)

    except httpx.TimeoutException:
        msg = f"{MODEL} → timeout after 90s (model may be cold, retry in 30s)"
        print(f"[TRACE] {msg}")
        return _fallback_report(context, msg)
    except Exception as e:
        msg = f"{MODEL} → {e}"
        print(f"[TRACE] {msg}")
        return _fallback_report(context, msg)


def _fallback_report(context: str, error_msg: str = "") -> str:
    import datetime as dt
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    err_section = f"\n> **Qwen API Error:** `{error_msg}`\n" if error_msg else ""
    return f"""### TRACE Tactical Intelligence Report [LOCAL MODE]
Generated: {ts}{err_section}

**1. THREAT ASSESSMENT**
Automated detection complete. Review the risk factors for primary threats identified by the AI models.

**2. ENVIRONMENTAL RISK**
Oil spill detection executed via U-Net SAR segmentation. Wind drift analysis requires weather feed.

**3. VESSEL PROFILES**
Vessels geo-tagged and dimensionally profiled via GSD ≈ 3.0 m/px. Position accuracy: ±0.005°.

**4. RECOMMENDED ACTIONS**
1. Dispatch coastal patrol to verify AIS-silent contacts
2. Cross-reference vessel dimensions with Lloyd's Register
3. Issue POLREP if spill area > 1,000 m²

**5. STRATEGIC OUTLOOK**
Waiting for HuggingFace API connection to generate AI-enhanced tactical analysis. Ensure HF_TOKEN is valid and model is awake.

*Context snippet: {context[:300]}...*"""


async def _empty_intel(lat: float = 0, lon: float = 0) -> dict:
    return {
        "weather": {},
        "news": [],
        "sentinel": {},
        "timestamp": datetime.utcnow().isoformat(),
        "coordinates": {"lat": lat, "lon": lon},
    }


@app.get("/api/history")
async def api_history(limit: int = 20):
    return get_history(limit)

@app.get("/api/analysis/{analysis_id}")
async def api_analysis(analysis_id: int):
    a = get_analysis(analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")
    return a

@app.get("/api/alerts")
async def api_alerts(unacknowledged_only: bool = False):
    return get_alerts(unacknowledged_only)

@app.post("/api/alerts/{alert_id}/ack")
async def api_ack_alert(alert_id: int):
    acknowledge_alert(alert_id)
    return {"acknowledged": True}

@app.get("/api/stats")
async def api_stats():
    return get_stats()

@app.get("/api/vessels/heatmap")
async def api_heatmap(days: int = 30):
    return get_vessel_heatmap(days)

@app.get("/api/intel")
async def api_intel(lat: float = 0.0, lon: float = 0.0, mode: str = "optical"):
    """Get intelligence data without running models — for map preview."""
    intel = await gather_intelligence(lat, lon, mode)
    return intel


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
