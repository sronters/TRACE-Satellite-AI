"""
TRACE Intelligence Layer
Provides real-time context: weather, news, AIS data, Sentinel imagery
"""

import os
import httpx
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import base64
from io import BytesIO

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
COPERNICUS_USER = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASS = os.getenv("COPERNICUS_PASS", "")


# ── WEATHER ──────────────────────────────────────────────────────────────────

async def get_weather(lat: float, lon: float) -> dict:
    """Fetch current weather for coordinates."""
    if not WEATHER_API_KEY:
        return _mock_weather(lat, lon)
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, params={
                "lat": lat, "lon": lon,
                "appid": WEATHER_API_KEY, "units": "metric"
            })
            d = r.json()
            wind_ms = d.get("wind", {}).get("speed", 0)
            beaufort = _ms_to_beaufort(wind_ms)
            return {
                "condition": d.get("weather", [{}])[0].get("description", "unknown").capitalize(),
                "temp_c": round(d.get("main", {}).get("temp", 0), 1),
                "wind_ms": round(wind_ms, 1),
                "wind_beaufort": beaufort,
                "wind_dir": _deg_to_compass(d.get("wind", {}).get("deg", 0)),
                "visibility_km": round(d.get("visibility", 10000) / 1000, 1),
                "humidity": d.get("main", {}).get("humidity", 0),
                "clouds_pct": d.get("clouds", {}).get("all", 0),
                "sar_suitable": d.get("clouds", {}).get("all", 0) > 70,
                "optical_suitable": d.get("clouds", {}).get("all", 0) < 30,
                "location_name": d.get("name", f"{lat:.2f}°N {lon:.2f}°E"),
            }
    except Exception as e:
        print(f"[Intel] Weather fetch failed: {e}")
        return _mock_weather(lat, lon)


def _mock_weather(lat, lon):
    return {
        "condition": "Partly cloudy",
        "temp_c": 18.0,
        "wind_ms": 7.2,
        "wind_beaufort": 4,
        "wind_dir": "NW",
        "visibility_km": 12.0,
        "humidity": 65,
        "clouds_pct": 40,
        "sar_suitable": False,
        "optical_suitable": True,
        "location_name": f"{lat:.2f}°N {lon:.2f}°E",
    }


def _ms_to_beaufort(ms: float) -> int:
    thresholds = [0.3, 1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6]
    for i, t in enumerate(thresholds):
        if ms < t:
            return i
    return 12


def _deg_to_compass(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[round(deg / 22.5) % 16]


# ── NEWS ──────────────────────────────────────────────────────────────────────

async def get_maritime_news(location_name: str = "") -> list[dict]:
    """Fetch recent maritime/environmental/security news."""
    if not NEWS_API_KEY:
        return _mock_news(location_name)
    try:
        # Include keywords for piracy, terrorism, and security incidents
        query = f"(maritime OR vessel OR shipping) AND (piracy OR attack OR security OR spill) {location_name}".strip()
        url = "https://newsapi.org/v2/everything"
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url, params={
                "q": query,
                "apiKey": NEWS_API_KEY,
                "language": "en",
                "sortBy": "relevancy",  # Sort by relevancy instead of just date to get important security news
                "pageSize": 6,
                "from": (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d"), # Look back a week for major incidents
            })
            articles = r.json().get("articles", [])
            return [
                {
                    "title": a.get("title", "")[:90],
                    "source": a.get("source", {}).get("name", ""),
                    "published": a.get("publishedAt", "")[:10],
                    "url": a.get("url", ""),
                }
                for a in articles[:5]
            ]
    except Exception as e:
        print(f"[Intel] News fetch failed: {e}")
        return _mock_news(location_name)


def _mock_news(location_name: str = ""):
    loc = location_name or "the region"
    return [
        {"title": f"High Risk Warning: Suspected piracy activity reported near {loc}", "source": "Maritime Security Network", "published": datetime.utcnow().strftime("%Y-%m-%d"), "url": "#"},
        {"title": "Coast Guard intercepts AIS-dark vessel in suspected smuggling op", "source": "Maritime News", "published": "2026-03-22", "url": "#"},
        {"title": "Satellite AI detects 47 illegal oil spills in Q1 2026", "source": "Reuters", "published": "2026-03-20", "url": "#"},
    ]


# ── SENTINEL IMAGERY ──────────────────────────────────────────────────────────

async def get_sentinel_image(lat: float, lon: float, mode: str = "optical") -> Optional[dict]:
    """
    Fetch latest Sentinel satellite image metadata for coordinates.
    Returns metadata dict, or mock if credentials unavailable.
    """
    if not COPERNICUS_USER or not COPERNICUS_PASS:
        return _mock_sentinel_meta(lat, lon, mode)

    try:
        collection = "SENTINEL-2" if mode == "optical" else "SENTINEL-1"
        token = await _get_copernicus_token()
        if not token:
            return _mock_sentinel_meta(lat, lon, mode)

        search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        params = {
            "$filter": (
                f"Collection/Name eq '{collection}' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
                f"{lon-0.3} {lat-0.3},{lon+0.3} {lat-0.3},"
                f"{lon+0.3} {lat+0.3},{lon-0.3} {lat+0.3},"
                f"{lon-0.3} {lat-0.3}))') and "
                f"ContentDate/Start gt {(datetime.utcnow()-timedelta(days=10)).strftime('%Y-%m-%dT00:00:00.000Z')}"
            ),
            "$orderby": "ContentDate/Start desc",
            "$top": 1,
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(search_url, params=params,
                                 headers={"Authorization": f"Bearer {token}"})
            products = r.json().get("value", [])
            if not products:
                return _mock_sentinel_meta(lat, lon, mode)

            product = products[0]
            return {
                "product_id": product.get("Id", ""),
                "name": product.get("Name", ""),
                "date": product.get("ContentDate", {}).get("Start", "")[:10],
                "collection": collection,
                "cloud_cover": product.get("Attributes", {}).get("cloudCover", "N/A"),
                "resolution": "10m" if mode == "optical" else "20m",
                "source": "ESA Copernicus",
                "mock": False,
            }
    except Exception as e:
        print(f"[Intel] Sentinel fetch failed: {e}")
        return _mock_sentinel_meta(lat, lon, mode)


async def _get_copernicus_token() -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                data={
                    "grant_type": "password",
                    "client_id": "cdse-public",
                    "username": COPERNICUS_USER,
                    "password": COPERNICUS_PASS,
                }
            )
            return r.json().get("access_token")
    except Exception:
        return None


def _mock_sentinel_meta(lat, lon, mode):
    collection = "Sentinel-2 L1C" if mode == "optical" else "Sentinel-1 GRD"
    return {
        "product_id": "mock-product-001",
        "name": f"{collection} · {lat:.2f}°N {lon:.2f}°E",
        "date": (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%d"),
        "collection": collection,
        "cloud_cover": "8%" if mode == "optical" else "N/A",
        "resolution": "10m" if mode == "optical" else "20m",
        "source": "ESA Copernicus (mock)",
        "mock": True,
    }


# ── GATHER ALL INTEL ──────────────────────────────────────────────────────────

async def gather_intelligence(lat: float, lon: float, mode: str = "optical") -> dict:
    """Run all intel fetches in parallel."""
    region = weather_location_from_coords(lat, lon)
    weather, news, sentinel = await asyncio.gather(
        get_weather(lat, lon),
        get_maritime_news(region),
        get_sentinel_image(lat, lon, mode),
    )
    return {
        "weather": weather,
        "news": news,
        "sentinel": sentinel,
        "timestamp": datetime.utcnow().isoformat(),
        "coordinates": {"lat": lat, "lon": lon},
    }


def weather_location_from_coords(lat: float, lon: float) -> str:
    """Rough region name for news queries."""
    regions = [
        ((30, 47, 25, 45), "Aegean Mediterranean Sea"),
        ((50, 60, -5, 10), "North Sea English Channel"),
        ((10, 30, 40, 60), "Red Sea Gulf of Aden"),
        ((20, 30, 115, 125), "South China Sea"),
        ((-10, 10, -80, -60), "Caribbean Sea"),
        ((35, 50, -10, 40), "Mediterranean Black Sea"),
    ]
    for (lat_min, lat_max, lon_min, lon_max), name in regions:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "maritime shipping"


def build_qwen_context_stub() -> str:
    """Stub context when no intel/risk available."""
    return (
        "You are TRACE — a maritime intelligence AI. "
        "Generate a professional tactical maritime report based on the satellite image provided. "
        "Focus on: vessel identification, threat assessment, environmental risk, and recommended actions."
    )
