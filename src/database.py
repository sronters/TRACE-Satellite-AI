"""
TRACE Database Layer
SQLite (no install needed) with full analysis history, vessel tracking, alerts.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.getenv("TRACE_DB", "trace.db")


def init_db():
    """Create all tables on startup."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.executescript("""
    CREATE TABLE IF NOT EXISTS analyses (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL,
        lat         REAL NOT NULL,
        lon         REAL NOT NULL,
        mode        TEXT NOT NULL,
        n_vessels   INTEGER DEFAULT 0,
        oil_area_m2 REAL DEFAULT 0,
        risk_score  INTEGER DEFAULT 0,
        risk_level  TEXT DEFAULT 'LOW',
        weather_json     TEXT,
        sentinel_json    TEXT,
        detections_json  TEXT,
        risk_json        TEXT,
        qwen_report      TEXT
    );

    CREATE TABLE IF NOT EXISTS vessels (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id INTEGER REFERENCES analyses(id),
        detected_at TEXT NOT NULL,
        lat         REAL,
        lon         REAL,
        length_m    REAL,
        width_m     REAL,
        area_m2     REAL,
        confidence  REAL,
        class_name  TEXT,
        is_dark_ais INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS alerts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL,
        analysis_id INTEGER REFERENCES analyses(id),
        level       TEXT NOT NULL,
        title       TEXT NOT NULL,
        detail      TEXT,
        lat         REAL,
        lon         REAL,
        acknowledged INTEGER DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_vessels_analysis ON vessels(analysis_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialized: {DB_PATH}")


def save_analysis(
    lat: float, lon: float, mode: str,
    detections: dict, risk_report, intel: dict, qwen_report: str
) -> int:
    """Save a full analysis run. Returns analysis ID."""
    vessels = detections.get("vessels", [])
    oil_area = detections.get("oil_spill_area_m2", 0) or 0

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    now = datetime.utcnow().isoformat()
    c.execute("""
        INSERT INTO analyses
        (created_at, lat, lon, mode, n_vessels, oil_area_m2, risk_score, risk_level,
         weather_json, sentinel_json, detections_json, risk_json, qwen_report)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        now, lat, lon, mode,
        len(vessels), oil_area,
        risk_report.total, risk_report.level,
        json.dumps(intel.get("weather", {})),
        json.dumps(intel.get("sentinel", {})),
        json.dumps(detections),
        json.dumps(risk_report.to_dict()),
        qwen_report,
    ))
    analysis_id = c.lastrowid

                             
    for v in vessels:
        gps = v.get("gps", {}) or {}
        is_dark = 1 if (v.get("length_m", 0) > 60 and v.get("confidence", 1) < 0.45) else 0
        c.execute("""
            INSERT INTO vessels
            (analysis_id, detected_at, lat, lon, length_m, width_m, area_m2, confidence, class_name, is_dark_ais)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            analysis_id, now,
            gps.get("lat"), gps.get("lon"),
            v.get("length_m"), v.get("width_m"),
            v.get("area_m2"), v.get("confidence"),
            v.get("class", "ship"),
            is_dark,
        ))

                                       
    for factor in risk_report.factors:
        if factor.severity in ("high", "medium"):
            c.execute("""
                INSERT INTO alerts
                (created_at, analysis_id, level, title, detail, lat, lon)
                VALUES (?,?,?,?,?,?,?)
            """, (now, analysis_id, factor.severity.upper(),
                  factor.name, factor.detail, lat, lon))

    conn.commit()
    conn.close()
    return analysis_id


def get_history(limit: int = 20) -> list[dict]:
    """Return recent analyses for history panel."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, created_at, lat, lon, mode, n_vessels, oil_area_m2,
               risk_score, risk_level, qwen_report
        FROM analyses
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis(analysis_id: int) -> Optional[dict]:
    """Get full analysis by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    row = c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for key in ("weather_json", "sentinel_json", "detections_json", "risk_json"):
        if d.get(key):
            try:
                d[key.replace("_json", "")] = json.loads(d[key])
            except Exception:
                pass
    return d


def get_alerts(unacknowledged_only: bool = False, limit: int = 50) -> list[dict]:
    """Return recent alerts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    q = "SELECT * FROM alerts"
    if unacknowledged_only:
        q += " WHERE acknowledged=0"
    q += " ORDER BY created_at DESC LIMIT ?"
    rows = c.execute(q, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_vessel_heatmap(days: int = 30) -> list[dict]:
    """Return all vessel positions for map heatmap."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    rows = c.execute("""
        SELECT lat, lon, length_m, is_dark_ais, confidence
        FROM vessels
        WHERE detected_at > datetime('now', ?)
          AND lat IS NOT NULL AND lon IS NOT NULL
        LIMIT 1000
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def acknowledge_alert(alert_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Dashboard statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    stats = {
        "total_analyses": c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0],
        "total_vessels": c.execute("SELECT COUNT(*) FROM vessels").fetchone()[0],
        "dark_vessels": c.execute("SELECT COUNT(*) FROM vessels WHERE is_dark_ais=1").fetchone()[0],
        "total_spill_m2": c.execute("SELECT COALESCE(SUM(oil_area_m2),0) FROM analyses").fetchone()[0],
        "active_alerts": c.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged=0").fetchone()[0],
        "avg_risk": c.execute("SELECT ROUND(AVG(risk_score),1) FROM analyses").fetchone()[0] or 0,
    }
    conn.close()
    return stats
