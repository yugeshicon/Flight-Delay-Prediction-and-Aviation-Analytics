"""Unit tests for src/predict.py."""

import pytest
from src.predict import predict_flight_delay, identify_risk_drivers, get_inference_artifacts


def test_artifacts_exist_and_load():
    clf, reg, prep, meta = get_inference_artifacts()
    assert clf is not None
    assert reg is not None
    assert prep is not None
    assert "champion_model" in meta
    assert meta["champion_model"] == "LightGBM"


def test_predict_flight_delay_valid_schema():
    input_data = {
        "Airline": "AA",
        "Origin": "DFW",
        "Dest": "MIA",
        "CRSDepTime": 1400,
        "DayOfWeek": 3,
        "DayofMonth": 12,
        "Distance": 1120.0,
        "CRSElapsedTime": 165.0,
    }
    result = predict_flight_delay(input_data)

    assert result["status"] in ["On-Time", "Delayed"]
    assert isinstance(result["is_delayed"], bool)
    assert 0.0 <= result["delay_probability"] <= 1.0
    assert 0.0 <= result["ontime_probability"] <= 1.0
    assert abs(result["delay_probability"] + result["ontime_probability"] - 1.0) < 1e-3
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    assert isinstance(result["key_drivers"], list)
    assert len(result["key_drivers"]) > 0

    if result["status"] == "Delayed":
        assert result["estimated_delay_minutes"] >= 15
    else:
        assert result["estimated_delay_minutes"] == 0


def test_predict_flight_delay_missing_fields_defaults():
    # Calling with empty or partial input must not crash
    result = predict_flight_delay({"Airline": "DL"})
    assert result["status"] in ["On-Time", "Delayed"]
    assert 0.0 <= result["delay_probability"] <= 1.0


def test_identify_risk_drivers():
    drivers_evening = identify_risk_drivers({"CRSDepTime": 1830, "Origin": "ORD", "Distance": 2000}, 0.75)
    assert any("Evening Peak" in d for d in drivers_evening)
    assert any("ORD" in d for d in drivers_evening)
    assert any("Long-Haul" in d for d in drivers_evening)
