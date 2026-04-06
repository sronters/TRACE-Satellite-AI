"""
TRACE Fleet Registry
Allows importing/managing known vessels and ports for cross-reference during analysis.
"""
import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.getenv("TRACE_DB", "trace.db")


def init_fleet_tables():
    """Create fleet + port tables."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS fleet (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        added_at    TEXT NOT NULL,
        name        TEXT NOT NULL,
        mmsi        TEXT,
        imo         TEXT,
        vessel_type TEXT,
        flag        TEXT,
        home_port   TEXT,
        length_m    REAL,
        width_m     REAL,
        owner       TEXT,
        notes       TEXT,
        is_friendly INTEGER DEFAULT 1,
        origin      TEXT,
        destination TEXT,
        cargo       TEXT,
        weight_tons REAL,
        has_protection INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS ports (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        country     TEXT,
        lat         REAL,
        lon         REAL,
        radius_km   REAL DEFAULT 5.0,
        is_home     INTEGER DEFAULT 0,
        notes       TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_fleet_mmsi ON fleet(mmsi);
    CREATE INDEX IF NOT EXISTS idx_fleet_name ON fleet(name);
    """)

    # Check and add new columns if upgrading from older version
    c = conn.cursor()
    c.execute("PRAGMA table_info(fleet)")
    columns = [row[1] for row in c.fetchall()]
    new_cols = {
        "origin": "TEXT",
        "destination": "TEXT",
        "cargo": "TEXT",
        "weight_tons": "REAL",
        "has_protection": "INTEGER DEFAULT 0"
    }
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            c.execute(f"ALTER TABLE fleet ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def import_fleet_from_json(data: list[dict]) -> int:
    """Import vessel list from JSON. Returns count of imported vessels."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = 0
    now = datetime.utcnow().isoformat()
    for v in data:
        c.execute("""
            INSERT OR REPLACE INTO fleet
            (added_at, name, mmsi, imo, vessel_type, flag, home_port,
             length_m, width_m, owner, notes, is_friendly,
             origin, destination, cargo, weight_tons, has_protection)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            now,
            v.get("name", "Unknown"),
            v.get("mmsi"),
            v.get("imo"),
            v.get("type") or v.get("vessel_type"),
            v.get("flag") or v.get("country"),
            v.get("port") or v.get("home_port"),
            v.get("length_m") or v.get("length"),
            v.get("width_m") or v.get("width"),
            v.get("owner"),
            v.get("notes"),
            int(v.get("friendly", True)),
            v.get("origin"),
            v.get("destination"),
            v.get("cargo"),
            v.get("weight_tons"),
            int(v.get("has_protection", False)),
        ))
        count += 1
    conn.commit()
    conn.close()
    return count


def import_fleet_from_csv(csv_text: str) -> int:
    """Import vessels from CSV text."""
    import csv, io
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = [dict(r) for r in reader]
    return import_fleet_from_json(rows)


def add_port(name: str, lat: float, lon: float, country: str = "",
             radius_km: float = 5.0, is_home: bool = False, notes: str = "") -> int:
    """Add a port/AOI to the registry."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO ports (name, country, lat, lon, radius_km, is_home, notes)
        VALUES (?,?,?,?,?,?,?)
    """, (name, country, lat, lon, radius_km, int(is_home), notes))
    port_id = c.lastrowid
    conn.commit()
    conn.close()
    return port_id


def get_fleet(is_friendly: Optional[bool] = None) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    q = "SELECT * FROM fleet"
    if is_friendly is not None:
        q += f" WHERE is_friendly={int(is_friendly)}"
    q += " ORDER BY name"
    rows = c.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ports() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM ports ORDER BY is_home DESC, name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_vessel(vessel_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM fleet WHERE id=?", (vessel_id,))
    conn.commit()
    conn.close()


def delete_port(port_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM ports WHERE id=?", (port_id,))
    conn.commit()
    conn.close()


def cross_reference_vessel(length_m: float, width_m: float, gps_lat: Optional[float] = None,
                           gps_lon: Optional[float] = None) -> Optional[dict]:
    """
    Try to match a detected vessel to a known fleet entry by size and proximity to home port.
    Returns best match or None.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Find vessels within 20% size tolerance
    lo, hi = length_m * 0.8, length_m * 1.2
    rows = conn.execute(
        "SELECT * FROM fleet WHERE length_m BETWEEN ? AND ? ORDER BY ABS(length_m - ?) LIMIT 3",
        (lo, hi, length_m)
    ).fetchall()
    conn.close()
    if not rows:
        return None
    best = dict(rows[0])
    best["match_confidence"] = max(0, 1 - abs(best.get("length_m", length_m) - length_m) / max(length_m, 1))
    return best


def enrich_detections_with_fleet(vessels: list[dict]) -> list[dict]:
    """
    Cross-reference detected vessels with known fleet.
    Adds 'fleet_match' key to each vessel dict.
    """
    enriched = []
    for v in vessels:
        match = cross_reference_vessel(
            length_m=v.get("length_m", 0),
            width_m=v.get("width_m", 0),
            gps_lat=v.get("gps", {}).get("lat"),
            gps_lon=v.get("gps", {}).get("lon"),
        )
        v = dict(v)
        v["fleet_match"] = match
        enriched.append(v)
    return enriched


def get_nearby_port(lat: float, lon: float) -> Optional[dict]:
    """Return the nearest port within its radius."""
    import math
    ports = get_ports()
    for port in ports:
        if not port.get("lat"):
            continue
        dlat = math.radians(lat - port["lat"])
        dlon = math.radians(lon - port["lon"])
        a = (math.sin(dlat/2)**2 +
             math.cos(math.radians(lat)) * math.cos(math.radians(port["lat"])) *
             math.sin(dlon/2)**2)
        dist_km = 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        if dist_km <= port.get("radius_km", 5):
            port = dict(port)
            port["distance_km"] = round(dist_km, 2)
            return port
    return None
