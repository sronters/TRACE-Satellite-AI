"""
TRACE Anomaly Scorer
====================
Binary anomaly scoring for AIS vessel records using a trained XGBoost model.

Label definition (from trace-anomaly-xgb.ipynb):
  1 = anomalous vessel record
  0 = normal vessel record

Anomaly conditions used during training:
  - Time gap between pings > 20 min          (cond_gap)
  - Implied speed > 60 knots                 (cond_speed)
  - Heading change > 120° in 2 pings with SOG > 10 kn  (cond_turn)
  - AIS status = 1 or 5 but moving > 3 kn   (cond_status)

Model: XGBoostClassifier trained with 5-fold CV (avg ROC-AUC ~0.9999)
Weights: models/anomaly_xgb.json
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

                                                                             
FEATURE_COLS = [
    "MMSI",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
    "VesselType",
    "Status",
    "Length",
    "Width",
    "Draft",
    "Cargo",
    "dt_min",
    "dist_nm",
    "implied_speed_kn",
    "delta_heading",
    "delta_heading_2min",
    "speed_diff",
    "turn_rate",
    "is_fast",
]

                                                           
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "anomaly_xgb.json"

                                                                             
def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in nautical miles."""
    R = 3440.065                      
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class AnomalyScorer:
    """
    Loads the XGBoost anomaly model and provides:

      score_record(record)  -> {"is_anomaly": bool, "probability": float}
      score_vessel_history(records) -> list[dict]  — per-ping scores + vessel summary
    """

    def __init__(self, model_path: Union[str, Path, None] = None) -> None:
        try:
            from xgboost import XGBClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "xgboost and scikit-learn are required. "
                "Install with: pip install xgboost scikit-learn"
            ) from exc

        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

                            
        self._model = XGBClassifier()
        self._model.load_model(str(model_path))

                                                                   
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()

                                                                     
        self._sklearn_fitted = False

                                                                        
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Impute NaN then scale."""
        if not self._sklearn_fitted:
            self._imputer.fit(X)
            X_imp = self._imputer.transform(X)
            self._scaler.fit(X_imp)
            self._sklearn_fitted = True
        X_imp = self._imputer.transform(X)
        return self._scaler.transform(X_imp)

                                                                        
    @staticmethod
    def engineer_features(
        current: dict,
        prev: Optional[dict] = None,
        prev2: Optional[dict] = None,
    ) -> dict:
        """
        Build the feature dict expected by the model from raw AIS fields.

        Parameters
        ----------
        current : dict
            Current AIS ping. Expected keys (all optional except MMSI):
            MMSI, LAT, LON, SOG, COG, Heading, VesselType, Status,
            Length, Width, Draft, Cargo, BaseDateTime (ISO-string or datetime).
        prev : dict | None
            Previous AIS ping for the same vessel (sorted ascending by time).
        prev2 : dict | None
            The ping before *prev* (for 2-step heading delta).

        Returns
        -------
        dict  with all FEATURE_COLS populated (NaN where unavailable).
        """
        from datetime import datetime

        def _to_dt(v):
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            try:
                return datetime.fromisoformat(str(v))
            except Exception:
                return None

        feat: dict = {}

                      
        for col in ["MMSI", "LAT", "LON", "SOG", "COG", "VesselType",
                    "Status", "Length", "Width", "Draft", "Cargo"]:
            feat[col] = current.get(col, np.nan)

                                        
        heading = current.get("Heading", np.nan)
        if heading == 511:
            heading = np.nan
        feat["Heading"] = heading

                                                                        
        if prev is not None:
            t_cur = _to_dt(current.get("BaseDateTime"))
            t_pre = _to_dt(prev.get("BaseDateTime"))
            if t_cur is not None and t_pre is not None:
                feat["dt_min"] = (t_cur - t_pre).total_seconds() / 60.0
            else:
                feat["dt_min"] = np.nan
        else:
            feat["dt_min"] = np.nan

                                                                        
        if prev is not None:
            try:
                dist = _haversine_nm(
                    prev["LAT"], prev["LON"],
                    current["LAT"], current["LON"],
                )
                feat["dist_nm"] = dist
                dt_h = feat["dt_min"] / 60.0 if not np.isnan(feat["dt_min"]) else np.nan
                feat["implied_speed_kn"] = dist / dt_h if (dt_h and dt_h > 0) else np.nan
            except (KeyError, TypeError, ZeroDivisionError):
                feat["dist_nm"] = np.nan
                feat["implied_speed_kn"] = np.nan
        else:
            feat["dist_nm"] = np.nan
            feat["implied_speed_kn"] = np.nan

                                                                        
        prev_heading = np.nan
        if prev is not None:
            ph = prev.get("Heading", np.nan)
            prev_heading = np.nan if ph == 511 else ph

        prev2_heading = np.nan
        if prev2 is not None:
            ph2 = prev2.get("Heading", np.nan)
            prev2_heading = np.nan if ph2 == 511 else ph2

        feat["delta_heading"] = (
            abs(heading - prev_heading)
            if not (np.isnan(heading) or np.isnan(prev_heading))
            else np.nan
        )
        feat["delta_heading_2min"] = (
            abs(heading - prev2_heading)
            if not (np.isnan(heading) or np.isnan(prev2_heading))
            else np.nan
        )

                                                                        
        sog = current.get("SOG", np.nan)
        impl = feat["implied_speed_kn"]
        feat["speed_diff"] = (
            (sog - impl) if not (np.isnan(sog) or np.isnan(impl)) else np.nan
        )

        dt = feat["dt_min"]
        dh = feat["delta_heading"]
        feat["turn_rate"] = (
            dh / (dt + 1e-6) if not (np.isnan(dh) or np.isnan(dt)) else np.nan
        )

        feat["is_fast"] = int(sog > 20) if not np.isnan(sog) else 0

        return feat

                                                                        
    def score_record(
        self,
        record: dict,
        prev: Optional[dict] = None,
        prev2: Optional[dict] = None,
    ) -> dict:
        """
        Score a single AIS ping.

        Parameters
        ----------
        record : dict  — current AIS ping (raw fields)
        prev   : dict  — previous ping for the same vessel (optional)
        prev2  : dict  — ping before prev (optional, for heading 2-step delta)

        Returns
        -------
        {
          "is_anomaly": bool,
          "probability": float,   # P(anomaly)
          "features": dict        # engineered feature values
        }
        """
        features = self.engineer_features(record, prev, prev2)
        X = np.array([[features.get(c, np.nan) for c in FEATURE_COLS]])
        X_proc = self._preprocess(X)

        prob = float(self._model.predict_proba(X_proc)[0, 1])
        label = bool(self._model.predict(X_proc)[0])

        return {
            "is_anomaly": label,
            "probability": prob,
            "features": features,
        }

    def score_vessel_history(self, records: list[dict]) -> dict:
        """
        Score a chronological list of AIS pings for one vessel.

        The records should be sorted by BaseDateTime ascending and all belong
        to the same MMSI.

        Returns
        -------
        {
          "mmsi": int | None,
          "pings": list[dict],        # per-ping {"is_anomaly", "probability", "features"}
          "anomaly_count": int,
          "anomaly_ratio": float,
          "vessel_is_anomalous": bool  # True if any ping flagged
        }
        """
        results = []
        for i, rec in enumerate(records):
            prev = records[i - 1] if i > 0 else None
            prev2 = records[i - 2] if i > 1 else None
            results.append(self.score_record(rec, prev, prev2))

        anomaly_count = sum(1 for r in results if r["is_anomaly"])
        total = len(results)

        return {
            "mmsi": records[0].get("MMSI") if records else None,
            "pings": results,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": anomaly_count / total if total > 0 else 0.0,
            "vessel_is_anomalous": anomaly_count > 0,
        }


_scorer: Optional[AnomalyScorer] = None


def _get_scorer() -> AnomalyScorer:
    global _scorer
    if _scorer is None:
        _scorer = AnomalyScorer()
    return _scorer


def score_ais_record(
    record: dict,
    prev: Optional[dict] = None,
    prev2: Optional[dict] = None,
) -> dict:
    """
    Module-level convenience wrapper around AnomalyScorer.score_record().
    Loads the model once (lazy singleton).

    Example
    -------
    >>> result = score_ais_record(
    ...     {"MMSI": 123456789, "LAT": 29.78, "LON": -95.08,
    ...      "SOG": 0.1, "COG": 226.5, "Heading": 340,
    ...      "BaseDateTime": "2022-03-31T00:00:01"},
    ... )
    >>> print(result["is_anomaly"], result["probability"])
    """
    return _get_scorer().score_record(record, prev, prev2)


def score_vessel_history(records: list[dict]) -> dict:
    """
    Module-level convenience wrapper around AnomalyScorer.score_vessel_history().
    """
    return _get_scorer().score_vessel_history(records)
