"""Standalone inference API module for Flight Delay Prediction.

Loads saved pipelines, validates user input, executes binary classification,
estimates delay duration, and assesses risk drivers for real-time predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np

from src.pipeline import load_preprocessor, transform_single_flight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Global lazy cache for inference artifacts
_CACHED_CLASSIFIER = None
_CACHED_REGRESSOR = None
_CACHED_PREPROCESSOR = None
_CACHED_METADATA = None


def get_inference_artifacts() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Retrieve and cache serialized inference models and preprocessor.

    Returns:
        Tuple: (classifier, regressor, preprocessor, metadata)
    """
    global _CACHED_CLASSIFIER, _CACHED_REGRESSOR, _CACHED_PREPROCESSOR, _CACHED_METADATA

    clf_path = MODELS_DIR / "flight_delay_classifier.pkl"
    reg_path = MODELS_DIR / "flight_delay_regressor.pkl"
    prep_path = MODELS_DIR / "preprocessor_pipeline.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"

    if _CACHED_CLASSIFIER is None:
        if not clf_path.exists():
            raise FileNotFoundError(f"Classifier artifact not found at {clf_path}. Run training first.")
        _CACHED_CLASSIFIER = joblib.load(clf_path)

    if _CACHED_REGRESSOR is None:
        if not reg_path.exists():
            raise FileNotFoundError(f"Regressor artifact not found at {reg_path}. Run training first.")
        _CACHED_REGRESSOR = joblib.load(reg_path)

    if _CACHED_PREPROCESSOR is None:
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}. Run training first.")
        _CACHED_PREPROCESSOR = load_preprocessor(prep_path)

    if _CACHED_METADATA is None:
        if meta_path.exists():
            with open(meta_path, "r") as f:
                _CACHED_METADATA = json.load(f)
        else:
            _CACHED_METADATA = {}

    return _CACHED_CLASSIFIER, _CACHED_REGRESSOR, _CACHED_PREPROCESSOR, _CACHED_METADATA


def identify_risk_drivers(flight_input: Dict[str, Any], delay_prob: float) -> List[str]:
    """Identify key factors contributing to flight delay risk.

    Args:
        flight_input: Pre-flight flight details.
        delay_prob: Predicted probability of delay.

    Returns:
        List[str]: Explanatory risk driver messages.
    """
    drivers = []
    dep_time = flight_input.get("CRSDepTime", 1200)
    hour = dep_time // 100 if isinstance(dep_time, int) else 12

    # Time of day drivers
    if hour in [7, 8, 9]:
        drivers.append("Morning Rush Hour Departure (07:00–09:59)")
    elif hour in [16, 17, 18, 19]:
        drivers.append("Evening Peak Departure Window (16:00–19:59)")
    elif hour >= 20:
        drivers.append("Late Evening Departure: Cumulative downstream propagation risk")
    elif hour <= 6:
        drivers.append("Early Morning Departure: Low historical congestion")

    # High traffic hubs
    origin = str(flight_input.get("Origin", "")).upper()
    dest = str(flight_input.get("Dest", "")).upper()
    major_congested_hubs = {"ORD", "DFW", "DEN", "ATL", "EWR", "JFK", "LGA", "CLT"}

    if origin in major_congested_hubs:
        drivers.append(f"High-Volume Origin Hub ({origin}) with frequent ground taxi delays")
    if dest in major_congested_hubs:
        drivers.append(f"Congested Destination Airspace ({dest})")

    # Flight distance
    distance = flight_input.get("Distance", 800)
    if distance > 1500:
        drivers.append(f"Cross-Country Long-Haul ({distance} miles): Extended weather exposure")

    # Weekend / Weekend travel
    dow = flight_input.get("DayOfWeek", 1)
    if dow in [5, 7]:  # Friday / Sunday
        drivers.append("Heavy Weekend Passenger Demand Period (Friday/Sunday)")

    if not drivers:
        drivers.append("Standard Scheduled Operational Profile")

    return drivers


def predict_flight_delay(
    flight_input: Dict[str, Any],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Execute real-time flight delay prediction and duration estimation.

    Args:
        flight_input: Dictionary specifying pre-flight attributes:
            - 'Airline': carrier code (e.g. 'DL', 'AA', 'UA')
            - 'Origin': 3-letter IATA code (e.g. 'ATL')
            - 'Dest': 3-letter IATA code (e.g. 'LGA')
            - 'CRSDepTime': scheduled departure time in HHMM (e.g. 1430)
            - 'DayOfWeek': 1 (Mon) to 7 (Sun)
            - 'DayofMonth': 1 to 31
            - 'Distance': route distance in miles (e.g. 762.0)
            - 'CRSElapsedTime': scheduled flight duration in mins (e.g. 135.0)
        threshold: Classification probability threshold for 'Delayed' (default: 0.50).

    Returns:
        Dict: Full prediction response containing status, probabilities,
              estimated delay minutes, and risk factors.
    """
    classifier, regressor, preprocessor, metadata = get_inference_artifacts()

    # Clean and validate input dictionary with safe defaults
    cleaned_input = {
        "Airline": str(flight_input.get("Airline", "AA")).strip().upper(),
        "Origin": str(flight_input.get("Origin", "ATL")).strip().upper(),
        "Dest": str(flight_input.get("Dest", "ORD")).strip().upper(),
        "CRSDepTime": int(flight_input.get("CRSDepTime", 1200)),
        "DayOfWeek": int(flight_input.get("DayOfWeek", 3)),
        "DayofMonth": int(flight_input.get("DayofMonth", 15)),
        "Distance": float(flight_input.get("Distance", 800.0)),
        "CRSElapsedTime": float(flight_input.get("CRSElapsedTime", 140.0)),
    }

    # Transform input through preprocessor
    X_single = transform_single_flight(cleaned_input, preprocessor)

    # Classification prediction & probability
    prob_delayed = float(classifier.predict_proba(X_single)[0, 1])
    prob_ontime = 1.0 - prob_delayed

    is_delayed = bool(prob_delayed >= threshold)
    status_label = "Delayed" if is_delayed else "On-Time"

    # Delay duration regression estimate
    if is_delayed:
        est_delay_raw = float(regressor.predict(X_single)[0])
        estimated_delay_mins = max(15, int(round(est_delay_raw)))
    else:
        estimated_delay_mins = 0

    # Risk level categorization
    if prob_delayed >= 0.65:
        risk_level = "High"
    elif prob_delayed >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    drivers = identify_risk_drivers(cleaned_input, prob_delayed)

    result = {
        "status": status_label,
        "is_delayed": is_delayed,
        "delay_probability": round(prob_delayed, 4),
        "delay_probability_pct": round(prob_delayed * 100, 1),
        "ontime_probability": round(prob_ontime, 4),
        "ontime_probability_pct": round(prob_ontime * 100, 1),
        "estimated_delay_minutes": estimated_delay_mins,
        "risk_level": risk_level,
        "key_drivers": drivers,
        "input_summary": cleaned_input,
    }
    return result
