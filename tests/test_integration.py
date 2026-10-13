"""End-to-End Integration tests for the Flight Delay Prediction & Aviation Analytics platform."""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_flight_data
from src.cleaning import clean_flight_data, extract_features_and_targets
from src.pipeline import fit_preprocessor, transform_single_flight, ALL_MODELING_FEATURES
from src.predict import predict_flight_delay, get_inference_artifacts
from src.analytics import compute_executive_kpis, compute_daily_trends, compute_airline_benchmarks


def test_end_to_end_prediction_flow():
    """Verify complete path: flight dictionary -> pipeline -> prediction & duration."""
    clf, reg, prep, meta = get_inference_artifacts()
    assert clf is not None
    assert reg is not None
    assert prep is not None

    test_inputs = [
        {"Airline": "DL", "Origin": "ATL", "Dest": "LGA", "CRSDepTime": 830, "DayOfWeek": 2, "DayofMonth": 12, "Distance": 762, "CRSElapsedTime": 130},
        {"Airline": "AA", "Origin": "DFW", "Dest": "ORD", "CRSDepTime": 1745, "DayOfWeek": 5, "DayofMonth": 2, "Distance": 802, "CRSElapsedTime": 145},
        {"Airline": "WN", "Origin": "MDW", "Dest": "DEN", "CRSDepTime": 1900, "DayOfWeek": 7, "DayofMonth": 15, "Distance": 888, "CRSElapsedTime": 160},
    ]

    for inp in test_inputs:
        res = predict_flight_delay(inp)
        assert res["status"] in ["On-Time", "Delayed"]
        assert 0.0 <= res["delay_probability"] <= 1.0
        assert 0.0 <= res["ontime_probability"] <= 1.0
        assert res["risk_level"] in ["Low", "Moderate", "High"]
        assert len(res["key_drivers"]) > 0
        if res["is_delayed"]:
            assert res["estimated_delay_minutes"] >= 15
        else:
            assert res["estimated_delay_minutes"] == 0


def test_end_to_end_analytics_integration():
    """Verify analytics computations reconcile with data loader output."""
    df_sample = load_flight_data("selected", nrows=2500)
    kpis = compute_executive_kpis(df_sample)
    trends = compute_daily_trends(df_sample)
    benchmarks = compute_airline_benchmarks(df_sample)

    assert kpis["total_flights"] == 2500
    assert trends["Total_Flights"].sum() == 2500
    assert benchmarks["Total_Flights"].sum() == 2500
