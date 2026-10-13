"""Unit tests for src/cleaning.py."""

import pytest
import pandas as pd
import numpy as np

from src.cleaning import clean_flight_data, assert_no_leakage, LEAKAGE_COLUMNS
from src.data_loader import load_flight_data


def test_clean_flight_data_states():
    # Load sample
    df_raw = load_flight_data(dataset_type="selected", nrows=1000)
    df_clean = clean_flight_data(df_raw, for_modeling=False)

    assert "Flight_State" in df_clean.columns
    assert set(df_clean["Flight_State"].unique()).issubset({"Completed", "Cancelled", "Diverted"})

    # Check for_modeling=True filters out Cancelled and Diverted
    df_model = clean_flight_data(df_raw, for_modeling=True)
    assert (df_model["Flight_State"] == "Cancelled").sum() == 0
    assert (df_model["Flight_State"] == "Diverted").sum() == 0
    assert "is_delayed" in df_model.columns
    assert "delay_minutes" in df_model.columns
    assert set(df_model["is_delayed"].unique()).issubset({0, 1})
    assert (df_model["delay_minutes"] < 0).sum() == 0


def test_delay_causes_imputed_zero_for_ontime():
    df_raw = load_flight_data(dataset_type="selected", nrows=1000)
    df_clean = clean_flight_data(df_raw, for_modeling=False)

    # For on-time completed flights, cause columns must be 0.0, not NaN
    ontime_flights = df_clean[(df_clean["Flight_State"] == "Completed") & (df_clean["ArrDel15"] == 0)]
    if len(ontime_flights) > 0:
        for col in ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]:
            if col in ontime_flights.columns:
                assert ontime_flights[col].isna().sum() == 0
                assert (ontime_flights[col] == 0.0).all()


def test_assert_no_leakage_passes_valid():
    valid_features = ["Airline", "Origin", "Dest", "Distance", "DayOfWeek", "CRSDepTime"]
    assert assert_no_leakage(valid_features) is True


def test_assert_no_leakage_catches_leakage():
    for leaky_col in ["DepDelay", "ArrDelay", "Actual Depart Time", "Taxi-out", "CarrierDelay"]:
        with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
            assert_no_leakage(["Airline", "Origin", leaky_col])
