"""
TRACE Risk Engine
Calculates a 0-100 risk score for each scan based on detections + intelligence.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskFactor:
    name: str
    score: int                                          
    severity: str                                
    detail: str


@dataclass
class RiskReport:
    total: int                                 
    level: str                                                                  
    level_color: str                                
    factors: list[RiskFactor] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self):
        return {
            "total": self.total,
            "level": self.level,
            "level_color": self.level_color,
            "factors": [
                {"name": f.name, "score": f.score, "severity": f.severity, "detail": f.detail}
                for f in self.factors
            ],
            "recommended_actions": self.recommended_actions,
            "summary": self.summary,
        }


def calculate_risk(
    vessels: list[dict],
    oil_spill_area_m2: Optional[float],
    weather: dict,
    mode: str = "optical",
    news: list[dict] = None,
) -> RiskReport:
    """
    Main risk calculation.
    vessels: list of YOLO detections with keys: class, confidence, length_m, width_m, gps
    oil_spill_area_m2: from U-Net, None if not detected
    weather: from intelligence layer
    news: latest region news to check for security threats
    """
    if news is None:
        news = []
    
    factors: list[RiskFactor] = []
    total = 0

                                                                               
    n_vessels = len(vessels)
    large_vessels = [v for v in vessels if v.get("length_m", 0) > 80]
    dark_candidates = _estimate_dark_vessels(vessels, weather)

                                     
    if dark_candidates > 0:
        pts = min(dark_candidates * 15, 40)
        total += pts
        factors.append(RiskFactor(
            name="AIS-dark vessels",
            score=pts,
            severity="high" if dark_candidates >= 2 else "medium",
            detail=f"{dark_candidates} vessel(s) with no AIS transponder signal detected",
        ))

                          
    if n_vessels > 50:
        pts = 10
        total += pts
        factors.append(RiskFactor(
            name="High vessel density",
            score=pts,
            severity="medium",
            detail=f"{n_vessels} vessels in area of interest — collision risk elevated",
        ))
    elif n_vessels > 20:
        pts = 5
        total += pts
        factors.append(RiskFactor(
            name="Moderate vessel density",
            score=pts,
            severity="low",
            detail=f"{n_vessels} vessels detected",
        ))

                                    
    if large_vessels:
        pts = min(len(large_vessels) * 5, 15)
        total += pts
        factors.append(RiskFactor(
            name="Large unidentified vessels",
            score=pts,
            severity="medium",
            detail=f"{len(large_vessels)} vessel(s) >80m without confirmed identity",
        ))

                                                                               
    if oil_spill_area_m2 and oil_spill_area_m2 > 0:
        if oil_spill_area_m2 > 1_000_000:
            pts, sev = 35, "high"
            detail = f"Major spill: {oil_spill_area_m2/1e6:.2f} km² — environmental emergency"
        elif oil_spill_area_m2 > 100_000:
            pts, sev = 25, "high"
            detail = f"Large spill: {oil_spill_area_m2/1000:.0f}k m² — immediate response needed"
        elif oil_spill_area_m2 > 10_000:
            pts, sev = 15, "medium"
            detail = f"Medium spill: {oil_spill_area_m2:.0f} m² — monitoring required"
        else:
            pts, sev = 8, "low"
            detail = f"Small spill: {oil_spill_area_m2:.0f} m² — possible bilge discharge"
        total += pts
        factors.append(RiskFactor(name="Oil spill detected", score=pts, severity=sev, detail=detail))

                               
        wind_ms = weather.get("wind_ms", 0)
        if wind_ms > 10:
            pts = 10
            total += pts
            factors.append(RiskFactor(
                name="High wind drift risk",
                score=pts,
                severity="high",
                detail=f"Wind {wind_ms}m/s will accelerate spill spread — response window <2h",
            ))
        elif wind_ms > 5:
            pts = 5
            total += pts
            factors.append(RiskFactor(
                name="Moderate wind drift",
                score=pts,
                severity="medium",
                detail=f"Wind {wind_ms}m/s · estimated drift 1.2km/h",
            ))

                                                                                
    beaufort = weather.get("wind_beaufort", 0)
    if beaufort >= 7:
        pts = 10
        total += pts
        factors.append(RiskFactor(
            name="Severe weather conditions",
            score=pts,
            severity="high",
            detail=f"Beaufort {beaufort} — vessel stability risk, reduced coast guard response",
        ))
    elif beaufort >= 5:
        pts = 5
        total += pts
        factors.append(RiskFactor(
            name="Adverse weather",
            score=pts,
            severity="medium",
            detail=f"Beaufort {beaufort} — elevated operational risk",
        ))

                     
    vis = weather.get("visibility_km", 10)
    if vis < 1:
        pts = 12
        total += pts
        factors.append(RiskFactor(
            name="Critically low visibility",
            score=pts,
            severity="high",
            detail=f"Visibility {vis}km — collision risk very high",
        ))
    elif vis < 3:
        pts = 6
        total += pts
        factors.append(RiskFactor(
            name="Low visibility",
            score=pts,
            severity="medium",
            detail=f"Visibility {vis}km — monitoring required",
        ))

                                                                                
    if mode == "sar" and weather.get("clouds_pct", 0) > 70:
        pts = 5
        total += pts
        factors.append(RiskFactor(
            name="SAR advantage active",
            score=pts,
            severity="low",
            detail="Cloud cover >70% — SAR provides coverage impossible for optical sensors",
        ))

                                                                                
    if news:
        security_keywords = ['piracy', 'attack', 'hijack', 'terror', 'drone', 'missile', 'armed']
        has_threats = False
        threat_titles = []
        for n in news:
            title_lower = n.get("title", "").lower()
            if any(k in title_lower for k in security_keywords):
                has_threats = True
                threat_titles.append(n.get("title"))
                
        if has_threats:
            pts = 20
            total += pts
                                                    
            threat_str = " | ".join(threat_titles[:2])
            factors.append(RiskFactor(
                name="Security Incident Reported in Region",
                score=pts,
                severity="high",
                detail=f"Recent news indicates elevated security risk: {threat_str}",
            ))

                                                                               
    total = min(total, 100)

    if total >= 75:
        level, color = "CRITICAL", "#E24B4A"
        actions = [
            "Alert coast guard — immediate deployment required",
            "Notify port authority and maritime coordination center",
            "Activate environmental emergency response team",
            "Maintain continuous SAR coverage every 15 minutes",
        ]
    elif total >= 50:
        level, color = "HIGH", "#D85A30"
        actions = [
            "Alert coast guard for investigation",
            "Increase satellite monitoring frequency",
            "Notify relevant maritime authority",
        ]
    elif total >= 25:
        level, color = "MEDIUM", "#EF9F27"
        actions = [
            "Schedule follow-up scan within 2 hours",
            "Log incident for pattern analysis",
            "Monitor AIS feeds for vessel identification",
        ]
    else:
        level, color = "LOW", "#1D9E75"
        actions = ["Continue routine monitoring"]

                                      
    factors.sort(key=lambda f: f.score, reverse=True)

                            
    parts = []
    if dark_candidates > 0:
        parts.append(f"{dark_candidates} AIS-dark vessel(s)")
    if oil_spill_area_m2 and oil_spill_area_m2 > 0:
        parts.append(f"oil spill {oil_spill_area_m2:.0f}m²")
    if beaufort >= 5:
        parts.append(f"Beaufort {beaufort} conditions")
    if any(f.score > 0 and 'Security Incident' in f.name for f in factors):
        parts.append("Elevated Security Risk")

    summary = f"Risk {total}/100 ({level}). " + (
        "Detected: " + ", ".join(parts) + "." if parts else "No critical threats detected."
    )

    return RiskReport(
        total=total,
        level=level,
        level_color=color,
        factors=factors,
        recommended_actions=actions,
        summary=summary,
    )


def _estimate_dark_vessels(vessels: list[dict], weather: dict) -> int:
    """
    Heuristic: vessels >60m with low detection confidence are likely AIS-dark.
    Real implementation would cross-reference with live AIS feed.
    """
    dark = 0
    for v in vessels:
        length = v.get("length_m", 0)
        conf = v.get("confidence", 1.0)
        if length > 60 and conf < 0.45:
            dark += 1
        elif length > 100 and weather.get("visibility_km", 10) < 5:
            dark += 1
    return dark


def build_qwen_context(intel: dict, risk: RiskReport, detections: dict) -> str:
    """
    Build a rich context string to inject into Qwen prompt.
    This makes Qwen reports dramatically more specific and useful.
    """
    w = intel.get("weather", {})
    news = intel.get("news", [])
    sentinel = intel.get("sentinel", {})
    coords = intel.get("coordinates", {})

    vessels = detections.get("vessels", [])
    n_vessels = len(vessels)
    oil_area = detections.get("oil_spill_area_m2", 0)

    fleet_info = []
    for v in vessels:
        fm = v.get("fleet_match")
        if fm:
            route = f"{fm.get('origin', '?')} to {fm.get('destination', '?')}" if fm.get('destination') else "Unknown Dest"
            cargo = fm.get('cargo', 'Unknown Cargo')
            prot = "Protected" if fm.get('has_protection') else "Unescorted"
            fleet_info.append(f"  - {fm['name']} ({fm.get('vessel_type','Ship')}): Route [{route}] | Cargo [{cargo}] | Security [{prot}]")

    fleet_str = "\n".join(fleet_info) if fleet_info else "  No known fleet vessels detected."
    news_str = "\n".join([f"  - {n['title']} ({n['source']})" for n in news[:3]]) or "  No recent news."

    context = f"""=== TRACE INTELLIGENCE CONTEXT ===

You are a professional maritime intelligence analyst. Use the following real-time data in your report.

LOCATION: {coords.get('lat', 0):.4f}°N {coords.get('lon', 0):.4f}°E
SATELLITE: {sentinel.get('collection', 'Unknown')} · {sentinel.get('date', 'Unknown')} · {sentinel.get('resolution', '?')} resolution
TIMESTAMP: {intel.get('timestamp', '')}

WEATHER CONDITIONS:
  Condition: {w.get('condition', 'Unknown')}
  Temperature: {w.get('temp_c', '?')}°C
  Wind: {w.get('wind_ms', '?')}m/s from {w.get('wind_dir', '?')} (Beaufort {w.get('wind_beaufort', '?')})
  Visibility: {w.get('visibility_km', '?')}km
  Cloud cover: {w.get('clouds_pct', '?')}%

FLEET & ROUTING (KNOWN VESSELS):
{fleet_str}

DETECTION RESULTS:
  Vessels detected: {n_vessels}
  Large vessels (>80m): {len([v for v in vessels if v.get('length_m',0)>80])}
  Estimated AIS-dark: {sum(1 for f in risk.factors if 'AIS' in f.name and f.score > 0)}
  Oil spill area: {f"{oil_area:.0f} m²" if oil_area else "Not detected"}

RISK ASSESSMENT: {risk.total}/100 ({risk.level})
  Top risk factors: {", ".join([f.name for f in risk.factors[:3]]) or "None"}

RECENT MARITIME NEWS IN REGION:
{news_str}

=== INSTRUCTIONS ===
Based on ALL of the above intelligence data, write a tactical maritime intelligence report with EXACTLY these 5 sections:

**1. THREAT ASSESSMENT** — cite specific vessel counts, suspicious activity, and AIS status.
**2. FLEET & ROUTE OPTIMIZATION** — For any known fleet vessels listed above: evaluate their route [{route} if known] against current weather (wind/visibility) and piracy/security risks. Suggest the safest optimal heading or speed adjustment. If no fleet detected, skip this analysis.
**3. ENVIRONMENTAL RISK** — oil spill analysis with wind drift projection using the actual wind data.
**4. RECOMMENDED ACTIONS** — 3-4 specific tactical or navigational actions based on threats and weather.
**5. STRATEGIC OUTLOOK** — next 24h assessment based on weather forecast and regional news.

Be specific — cite the actual numbers, vessel names, and ports from the intelligence context above. Do not use generic phrases.
"""
    return context
