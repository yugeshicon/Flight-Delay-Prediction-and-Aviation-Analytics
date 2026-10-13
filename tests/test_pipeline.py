"""Unit tests for src/pipeline.py."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_flight_data
from src.cleaning import clean_flight_data
from src.pipeline import (
    build_preprocessor,
    prepare_features,
    fit_preprocessor,
    transform_single_flight,
    save_preprocessor,
    load_preprocessor,
    ALL_MODELING_FEATURES,
)


def test_prepare_features():
    df_raw = load_flight_data(dataset_type="selected", nrows=100)
    df_clean = clean_flight_data(df_raw, for_modeling=True)
    X_df = prepare_features(df_clean)

    assert isinstance(X_df, pd.DataFrame)
    for col in ALL_MODELING_FEATURES:
        assert col in X_df.columns
    assert X_df.isna().sum().sum() == 0


def test_fit_and_transform_pipeline(tmp_path):
    df_raw = load_flight_data(dataset_type="selected", nrows=500)
    df_clean = clean_flight_data(df_raw, for_modeling=True)

    preprocessor, X_trans = fit_preprocessor(df_clean)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(df_clean)
    assert not np.isnan(X_trans).any()

    # Test single flight transform (matching Streamlit form)
    single_flight = {
        "Airline": "DL",
        "Origin": "ATL",
        "Dest": "LGA",
        "CRSDepTime": 1430,
        "DayOfWeek": 5,
        "DayofMonth": 15,
        "Distance": 762.0,
        "CRSElapsedTime": 135.0,
    }
    x_single = transform_single_flight(single_flight, preprocessor)
    assert isinstance(x_single, np.ndarray)
    assert x_single.shape == (1, X_trans.shape[1])
    assert not np.isnan(x_single).any()

    # Test serialization and reloading
    model_path = tmp_path / "test_preprocessor.pkl"
    save_preprocessor(preprocessor, model_path)
    loaded_preprocessor = load_preprocessor(model_path)

    x_loaded = transform_single_flight(single_flight, loaded_preprocessor)
    np.testing.assert_array_almost_equal(x_single, x_loaded)
