"""Unit tests for src/data_loader.py."""

import pytest
import pandas as pd
from src.data_loader import load_flight_data, get_dataset_path, get_metadata


def test_get_dataset_path_valid():
    path = get_dataset_path("selected")
    assert path.exists()
    assert path.name == "Flight Delay Selected Dataset.csv"


def test_get_dataset_path_invalid():
    with pytest.raises(ValueError):
        get_dataset_path("non_existent_dataset")


def test_load_flight_data_sample():
    df = load_flight_data(dataset_type="selected", nrows=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "Cancelled" in df.columns
    assert "Diverted" in df.columns
    assert "Origin" in df.columns
    assert "Dest" in df.columns


def test_get_metadata():
    df = load_flight_data(dataset_type="selected", nrows=200)
    metadata = get_metadata(df)
    assert metadata["total_flights"] == 200
    assert metadata["completed_flights"] <= 200
    assert "unique_airlines" in metadata
    assert isinstance(metadata["unique_origins_count"], int)
