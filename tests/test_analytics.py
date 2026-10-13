"""Unit tests for src/analytics.py."""

import pytest
import pandas as pd
from src.data_loader import load_flight_data
from src.analytics import (
    compute_executive_kpis,
    compute_daily_trends,
    compute_airline_benchmarks,
    compute_airport_analytics,
    compute_dep_vs_arr_dynamics,
    compute_delay_causes_breakdown,
)


@pytest.fixture(scope="module")
def sample_flights_df():
    return load_flight_data(dataset_type="selected", nrows=5000)


def test_compute_executive_kpis(sample_flights_df):
    kpis = compute_executive_kpis(sample_flights_df)

    assert kpis["total_flights"] == len(sample_flights_df)
    assert kpis["total_flights"] == kpis["completed_flights"] + kpis["cancelled_flights"] + kpis["diverted_flights"]
    assert kpis["completed_flights"] == kpis["ontime_flights"] + kpis["delayed_flights"]
    assert 0.0 <= kpis["ontime_rate_pct"] <= 100.0
    assert 0.0 <= kpis["delayed_rate_pct"] <= 100.0


def test_compute_daily_trends(sample_flights_df):
    trends = compute_daily_trends(sample_flights_df)
    assert isinstance(trends, pd.DataFrame)
    assert "Day" in trends.columns
    assert "Delay_Rate_Pct" in trends.columns
    assert trends["Total_Flights"].sum() == len(sample_flights_df)


def test_compute_airline_benchmarks(sample_flights_df):
    benchmarks = compute_airline_benchmarks(sample_flights_df)
    assert isinstance(benchmarks, pd.DataFrame)
    assert "Airline" in benchmarks.columns
    assert "Delay_Rate_Pct" in benchmarks.columns
    assert benchmarks["Total_Flights"].sum() == len(sample_flights_df)


def test_compute_airport_analytics(sample_flights_df):
    busiest, most_delayed = compute_airport_analytics(sample_flights_df, min_flights=10)
    assert isinstance(busiest, pd.DataFrame)
    assert isinstance(most_delayed, pd.DataFrame)
    assert "Origin" in busiest.columns


def test_compute_dep_vs_arr_dynamics(sample_flights_df):
    dynamics = compute_dep_vs_arr_dynamics(sample_flights_df)
    assert "correlation" in dynamics
    assert -1.0 <= dynamics["correlation"] <= 1.0
    assert 0.0 <= dynamics["recovery_rate_pct"] <= 100.0


def test_compute_delay_causes_breakdown(sample_flights_df):
    breakdown = compute_delay_causes_breakdown(sample_flights_df)
    assert "grand_total_delay_minutes" in breakdown
    assert "causes" in breakdown
    causes_sum = sum(d["total_minutes"] for d in breakdown["causes"].values())
    assert abs(causes_sum - breakdown["grand_total_delay_minutes"]) < 1e-1
