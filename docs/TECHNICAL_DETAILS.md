# TRACE v2.0 — Техническая документация

> **Версия:** 2.0  
> **Язык:** Python 3.11+  
> **Фреймворк:** FastAPI  
> **Дата:** 2024

---

## Содержание

1. [Общая архитектура](#1-общая-архитектура)
2. [Backend — FastAPI](#2-backend--fastapi)
3. [ML-конвейер](#3-ml-конвейер)
   - [YOLOv8-OBB — детекция судов](#31-yolov8-obb--детекция-судов)
   - [U-Net — сегментация разливов](#32-u-net--сегментация-разливов)
   - [XGBoost — AIS-аномалии](#33-xgboost--ais-аномалии)
4. [Разведывательный слой](#4-разведывательный-слой)
5. [Движок риска](#5-движок-риска)
6. [База данных](#6-база-данных)
7. [Реестр флота и портов](#7-реестр-флота-и-портов)
8. [Фронтенд](#8-фронтенд)
9. [API Reference](#9-api-reference)
10. [Конфигурация](#10-конфигурация)
11. [Производительность](#11-производительность)

---

## 1. Общая архитектура

TRACE состоит из трёх логических слоёв:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ФРОНТЕНД (SPA)                               │
│  Leaflet.js карта · Leaflet WMS · Vanilla JS · SVG/CSS UI      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP / REST JSON
┌───────────────────────────▼─────────────────────────────────────┐
│                    FASTAPI BACKEND                               │
│                                                                  │
│  /process ──► Tile Download ──► ML Pipeline ──► Risk Engine    │
│  /api/anomaly/score ──────────────────────► XGBoost Scorer     │
│  /api/fleet · /api/ports · /api/history ──► SQLite             │
│                                                                  │
│  intelligence.py ──► Weather · News · Sentinel Hub             │
└──────────┬──────────────────────┬────────────────────────────────┘
           │                      │
┌──────────▼──────┐   ┌───────────▼──────────────────────────────┐
│  SQLite DB      │   │         ML MODELS                         │
│  trace.db       │   │  best.pt     → YOLOv8-OBB                │
│  - analyses     │   │  best_unet   → U-Net ResNet-34            │
│  - alerts       │   │  anomaly_xgb → XGBoost Classifier         │
│  - fleet        │   │  Qwen 72B    → HuggingFace API            │
│  - ports        │   └──────────────────────────────────────────┘
└─────────────────┘
```

**Принцип работы (основной сценарий):**

1. Пользователь кликает на карту → получает координаты
2. Нажимает **INITIATE SCAN**
3. Backend скачивает 5×5 тайлов ESRI (zoom=15) → сшивает в один JPEG
4. Параллельно: запрашивает погоду + новости
5. YOLO детектирует суда → координаты вычисляются через геотрансформ
6. U-Net сегментирует разливы (только SAR/DUAL режим)
7. Risk Engine вычисляет риск 0–100
8. Контекст передаётся в Qwen 2.5-VL-72B → тактический отчёт
9. Результат сохраняется в SQLite
10. Всё возвращается фронтенду одним JSON-ответом

---

## 2. Backend — FastAPI

### Файл: `src/app.py`

Основной модуль FastAPI-приложения. Жизненный цикл управляется через `lifespan`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()           # создание таблиц SQLite
    init_fleet_tables() # таблицы флота/портов
    load_models()       # загрузка YOLO + U-Net в память
    yield
```

### Скачивание спутниковых тайлов

```python
async def download_area_image(lat, lon, zoom=15, radius=2)
```

- Источник: `ESRI World Imagery` (бесплатно, без ключа)
- Размер сетки: `(radius*2+1)² = 5×5 = 25 тайлов`
- Разрешение тайла: 256×256 px
- Итоговый снимок: **1280×1280 px**
- GSD при zoom=15: ≈ **4.4 м/пиксель**
- Охватываемая площадь: ≈ **5.6 км²**

**Вычисление bounding box:**

```python
top_lat, left_lon = _tile_to_lat_lon(cx - radius, cy - radius, zoom)
bot_lat, right_lon = _tile_to_lat_lon(cx + radius + 1, cy + radius + 1, zoom)
```

### Водяная маска (`_create_water_mask`)

Фильтрует суда, обнаруженные YOLO на суше:

**Оптический режим:**
```
NDWI-подобный индекс:  BDR = (B - R) / (B + R + ε)
Условия воды:
  - BDR > 0.05           (синяя доминанта)
  - G < B × 1.3          (не растительность)
  - brightness < 180     (вода не слишком яркая)
  - HSV hue ∈ [80, 140]  (сине-голубой диапазон)
  - ИЛИ: очень тёмное (brightness < 40)
```

**SAR-режим:** adaptive threshold + Otsu

После бинаризации: `MORPH_CLOSE(31×31)` → `MORPH_OPEN(12×12)` → удаление компонент < 0.5% площади.

---

## 3. ML-конвейер

### 3.1 YOLOv8-OBB — детекция судов

**Модель:** `YOLOv8n-OBB` (Oriented Bounding Box)  
**Файл весов:** `models/best.pt`  
**Обучение:** на датасете DOTA (Detection in Optical Remote Sensing Images)

**Инференс:**
```python
results = yolo_model(img, verbose=False, conf=0.1, iou=0.45, imgsz=1024)
```

**Параметры:**
| Параметр | Значение | Описание |
|---|---|---|
| `conf` | 0.10 | Минимальная уверенность |
| `iou` | 0.45 | NMS IoU-порог |
| `imgsz` | 1024 | Размер входа |

**Вычисление GPS из пикселей:**

```python
vessel_lat = lat_max - (py / h) * (lat_max - lat_min)
vessel_lon = lon_min + (px / w) * (lon_max - lon_min)
```

**Вычисление размеров:**
```python
length_m = max(bw, bh) * GSD  # GSD = метр/пиксель
width_m  = min(bw, bh) * GSD
```

---

### 3.2 U-Net — сегментация разливов

**Архитектура:** `smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)`  
**Файл весов:** `models/best_unet_sos.pth`  
**Обучение:** на SAR Oil Spill Dataset (SOS)

**Пайплайн инференса:**
```python
# 1. Конвертация в grayscale (1 канал)
gray = Image.fromarray(img_np).convert("L")

# 2. Ресайз до 512×512 + нормализация [0, 1]
arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

# 3. Инференс
with torch.no_grad():
    out = unet_model(tensor)
    mask = torch.sigmoid(out).squeeze().cpu().numpy()

# 4. Бинаризация (порог 0.5)
mask_bin = (mask > 0.5).astype(np.uint8) * 255

# 5. Расчёт площади
area_m2 = pixel_count * (GSD ** 2)
```

---

### 3.3 XGBoost — AIS-аномалии

**Файл:** `src/anomaly_scorer.py`  
**Веса:** `models/anomaly_xgb.json`

#### Датасет обучения

| Параметр | Значение |
|---|---|
| Источник | AIS_2022_03_31.csv (Kaggle) |
| Всего записей | 7 167 046 |
| Аномалий | 114 382 (≈1.6%) |
| `scale_pos_weight` | 61.2 (балансировка классов) |

#### Гиперпараметры модели

```python
XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",        # GPU-ускорение
    scale_pos_weight=61.2,     # балансировка
)
```

#### Пайплайн (ноутбук)

```
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   XGBClassifier(...))
])
```

> **Важно:** При инференсе через `AnomalyScorer` sklearn-трансформеры (imputer + scaler) дообучаются на первом батче данных. На продакшене рекомендуется сохранить fitted трансформеры отдельно.

#### Инжиниринг признаков

```python
# Функция Haversine (морские мили)
R = 3440.065
dist_nm = 2 * R * arcsin(sqrt(sin(dlat/2)² + cos(lat1)*cos(lat2)*sin(dlon/2)²))

# Расчётная скорость
implied_speed_kn = dist_nm / (dt_min / 60)

# Скорость поворота
turn_rate = delta_heading / (dt_min + 1e-6)

# Признак скорости
is_fast = int(SOG > 20)
```

#### Использование из Python

```python
from src.anomaly_scorer import score_ais_record, score_vessel_history

# Один пинг
result = score_ais_record(
    record={"LAT": 29.78, "LON": -95.08, "SOG": 65.0, "Heading": 226},
    prev={"LAT": 29.77, "LON": -95.07, "Heading": 220,
          "BaseDateTime": "2022-03-31T00:00:01"},
)
# → {"is_anomaly": True, "probability": 0.987, "features": {...}}

# История судна
result = score_vessel_history(sorted_pings_list)
# → {"mmsi": 367702220, "anomaly_count": 3, "vessel_is_anomalous": True, ...}
```

---

## 4. Разведывательный слой

**Файл:** `src/intelligence.py`

```python
async def gather_intelligence(lat, lon, mode) -> dict:
    # Параллельные запросы
    weather = await _get_weather(lat, lon)
    news    = await _get_news(lat, lon)
    sentinel = _get_sentinel_info(lat, lon, mode)
    return {weather, news, sentinel, timestamp, coordinates}
```

### Источники данных

| Источник | Данные | API |
|---|---|---|
| OpenWeatherMap | Температура, ветер (м/с), видимость, облачность, шкала Бофора | `api.openweathermap.org` |
| GNews API | Новости по региону (фильтр по ключевым словам) | `gnews.io` |
| Sentinel Hub WMS | Метаданные Sentinel-2/Sentinel-1 снимков | `services.sentinel-hub.com` |

### Погодные данные (структура)

```json
{
  "condition": "Partly cloudy",
  "temp_c": 24.5,
  "wind_ms": 7.2,
  "wind_dir": "NW",
  "wind_beaufort": 4,
  "visibility_km": 9.0,
  "clouds_pct": 35
}
```

---

## 5. Движок риска

**Файл:** `src/risk_engine.py`

Вычисляет интегральный риск `0–100` на основе детекций и разведданных.

### Формула риска

```
total = 0

# AIS-dark суда (максимальный вес)
total += min(dark_count × 15, 40)

# Плотность трафика
total += 10 (если > 50 судов)
total += 5  (если > 20 судов)

# Крупные суда > 80 м
total += min(count × 5, 15)

# Разлив нефти
total += 35 (> 1 км²)
total += 25 (> 100 000 м²)
total += 15 (> 10 000 м²)
total += 8  (< 10 000 м²)

# Ветер при разливе: +5 или +10

# Погода: шкала Бофора
total += 10 (≥ 7)
total += 5  (≥ 5)

# Видимость
total += 12 (< 1 км)
total += 6  (< 3 км)

# Новости: угрозы пиратства/терроризма
total += 20

total = min(total, 100)  # зажим
```

### Уровни риска

| Диапазон | Уровень | Цвет |
|---|---|---|
| 75–100 | CRITICAL | `#E24B4A` |
| 50–74 | HIGH | `#D85A30` |
| 25–49 | MEDIUM | `#EF9F27` |
| 0–24 | LOW | `#1D9E75` |

### Определение AIS-dark судов

```python
def _estimate_dark_vessels(vessels, weather) -> int:
    # Heuristic: крупные суда с низкой уверенностью YOLO
    if length > 60 and confidence < 0.45:
        dark += 1
    elif length > 100 and visibility_km < 5:
        dark += 1
```

---

## 6. База данных

**Файл:** `src/database.py`  
**СУБД:** SQLite (`trace.db`)

### Схема таблиц

```sql
-- История анализов
CREATE TABLE analyses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT,
    lat         REAL,
    lon         REAL,
    mode        TEXT,           -- 'optical'|'sar'|'dual'
    detections  TEXT,           -- JSON: vessels + oil
    risk        TEXT,           -- JSON: RiskReport
    intel       TEXT,           -- JSON: погода + новости
    qwen_report TEXT,           -- текст тактического отчёта
    risk_total  INTEGER,
    risk_level  TEXT            -- 'LOW'|'MEDIUM'|'HIGH'|'CRITICAL'
);

-- Алерты
CREATE TABLE alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER REFERENCES analyses(id),
    threat_type     TEXT,
    severity        TEXT,       -- 'high'|'medium'|'low'
    description     TEXT,
    timestamp       TEXT,
    acknowledged    INTEGER DEFAULT 0
);

-- Флот
CREATE TABLE fleet (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    mmsi        TEXT,
    length_m    REAL,
    vessel_type TEXT,
    flag        TEXT,
    home_port   TEXT,
    ...
);

-- Порты
CREATE TABLE ports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    lat         REAL,
    lon         REAL,
    country     TEXT,
    radius_km   REAL DEFAULT 5.0,
    is_home     INTEGER DEFAULT 0,
    notes       TEXT
);
```

---

## 7. Реестр флота и портов

**Файл:** `src/fleet.py`

### Обогащение детекций

После YOLO-детекции каждое обнаруженное судно сравнивается с флотом:

```python
def enrich_detections_with_fleet(vessels) -> list:
    for vessel in vessels:
        # Поиск ближайшего судна флота по GPS (радиус 10 км)
        match = _find_nearest_fleet(vessel["gps"]["lat"], vessel["gps"]["lon"])
        if match:
            vessel["fleet_match"] = {
                "name": match.name,
                "vessel_type": match.vessel_type,
                "origin": match.home_port,
                "cargo": match.cargo,
                "has_protection": match.has_protection,
            }
    return vessels
```

### Импорт флота

**CSV-формат:**
```csv
name,mmsi,length_m,vessel_type,flag,home_port
ATLAS,367702220,180.5,Tanker,US,Houston
```

**JSON-формат:**
```json
[{"name": "ATLAS", "mmsi": "367702220", "length_m": 180.5, "vessel_type": "Tanker"}]
```

---

## 8. Фронтенд

**Файл:** `src/index.html` — Single Page Application

### Технологии

| Технология | Версия | Назначение |
|---|---|---|
| Leaflet.js | 1.9.4 | Интерактивная карта |
| Leaflet WMS | — | Sentinel-2/Sentinel-1 слои |
| Google Fonts | Inter + JetBrains Mono | Типографика |
| Vanilla JS | ES2022 | Логика без фреймворка |
| CSS Custom Properties | — | Тёмная тема |

### Компоненты UI

```
Topbar
  ├── Логотип TRACE
  ├── Табы: MAP · HISTORY · ANALYTICS · EXPORT
  └── Статус системы + часы UTC

Left Panel (284px)
  ├── Target Coordinates (lat/lon, mode, upload)
  ├── Fleet Integration (CSV upload)
  ├── AIS Anomaly Check  ← NEW
  ├── Intelligence Layer (4 карточки)
  └── Active Alerts (прокручиваемый список)

Map Center (1fr)
  ├── Leaflet карта (ESRI/Sentinel-2/SAR/OSM)
  ├── Layer toggle pills
  ├── Drop zone
  └── Progress overlay

Right Panel (308px)
  ├── Risk Assessment (0-100 + gauge + факторы)
  ├── Tactical Intel (отчёт Qwen)
  ├── Object Inspector (детали судна + добавить во флот)
  ├── Analysis History
  └── News Ticker
```

### Жизненный цикл сканирования (JS)

```javascript
async function initiateScan() {
  // 1. Формирование FormData (файл или координаты)
  // 2. POST /process
  // 3. Анимация прогресс-бара (фиктивные этапы)
  // 4. Получение ответа
  // 5. Обновление карты: addImageOverlay + маркеры судов
  // 6. Обновление панели риска
  // 7. Отображение тактического отчёта
  // 8. Обновление истории
}
```

---

## 9. API Reference

### POST `/process`

**Параметры формы (multipart):**

| Поле | Тип | Описание |
|---|---|---|
| `file` | файл | Спутниковый снимок (опционально) |
| `mode` | string | `optical` / `sar` / `dual` |
| `lat` | float | Широта целевой точки |
| `lon` | float | Долгота целевой точки |
| `use_sentinel` | bool | Использовать Sentinel-метаданные |
| `zoom` | int | Уровень зума тайлов (14–17) |

**Ответ (JSON):**
```json
{
  "analysis_id": 42,
  "detections": {
    "vessels": [
      {
        "class": "ship",
        "confidence": 0.87,
        "length_m": 185.3,
        "width_m": 28.1,
        "area_m2": 5207.0,
        "gps": {"lat": 37.9452, "lon": 23.6431},
        "fleet_match": null
      }
    ],
    "oil_spill_area_m2": 0,
    "oil_polygons": []
  },
  "risk": {
    "total": 68,
    "level": "HIGH",
    "level_color": "#D85A30",
    "factors": [...],
    "recommended_actions": [...],
    "summary": "Risk 68/100 (HIGH)..."
  },
  "intel": {
    "weather": {...},
    "news": [...],
    "sentinel": {...}
  },
  "qwen_report": "### TRACE Tactical Intelligence Report...",
  "processed_image": "base64-jpeg...",
  "image_bbox": [37.901, 37.990, 23.600, 23.689],
  "timestamp": "2024-04-11T08:42:00"
}
```

### POST `/api/anomaly/score`

Подробнее: [README.md → API-эндпоинты](#-api-эндпоинты)

---

## 10. Конфигурация

### `.env` переменные

```env
HF_TOKEN=hf_...    # HuggingFace PRO токен (Qwen 72B)
```

> Погода и новости настраиваются в `src/intelligence.py` через константы `OWM_API_KEY` и `GNEWS_KEY` (или переменные окружения).

### Настройка разрешения тайлов

В `app.py` → `download_area_image(zoom=15, radius=2)`:

| zoom | GSD (м/пкс) | Охват (5×5 тайлов) |
|---|---|---|
| 14 | ~8.8 | ~22 км² |
| 15 | ~4.4 | ~5.6 км² ← **рекомендуется** |
| 16 | ~2.2 | ~1.4 км² |
| 17 | ~1.1 | ~0.35 км² |

---

## 11. Производительность

### Типичное время обработки

| Этап | Время | Условия |
|---|---|---|
| Скачивание тайлов | 2–5 с | Сеть 50 Мбит/с, 25 тайлов |
| Погода + новости | 1–3 с | Параллельно |
| YOLO (GPU) | 0.3–1 с | RTX 3060, imgsz=1024 |
| YOLO (CPU) | 3–8 с | — |
| U-Net (GPU) | 0.2–0.5 с | |
| XGBoost (`/api/anomaly/score`) | < 10 мс | |
| Qwen 2.5-VL-72B | 15–45 с | HF Inference API |
| **Итого (с Qwen)** | ~25–55 с | GPU сервер |
| **Итого (без Qwen / fallback)** | ~5–12 с | |

### Рекомендации для продакшена

1. **GPU обязателен** для разумного времени ответа YOLO + U-Net
2. Настройте **кэш разведданных** (погода обновляется каждые 30 мин)
3. Используйте **Gunicorn + Uvicorn workers** для параллельных запросов:
   ```bash
   gunicorn src.app:app -k uvicorn.workers.UvicornWorker --workers 4
   ```
4. Для AIS-аномалий при высокой нагрузке — сохраните fitted sklearn-трансформеры (`joblib.dump`) и загружайте при старте
