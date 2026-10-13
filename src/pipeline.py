"""Scikit-Learn Preprocessing Pipeline for Flight Delay Prediction.

Constructs, fits, and serializes the unified ColumnTransformer handling
categorical encoding and numerical scaling for both batch training and real-time inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cleaning import assert_no_leakage
from src.feature_engineering import engineer_preflight_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical feature definitions
CATEGORICAL_FEATURES: List[str] = ["Airline", "Origin", "Dest", "Depart_Time_Block"]
NUMERICAL_FEATURES: List[str] = [
    "Depart_Hour",
    "Distance",
    "CRSElapsedTime",
    "Day_Of_Week",
    "Day_Of_Month",
    "Is_Weekend",
    "Peak_Hour",
]
ALL_MODELING_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build an unfitted Scikit-Learn ColumnTransformer for pre-flight attributes.

    Returns:
        ColumnTransformer: Preprocessor ready to be fitted.
    """
    # Categorical encoder: OneHot with max_categories to handle 360+ airports cleanly
    cat_transformer = OneHotEncoder(
        handle_unknown="ignore",
        max_categories=40,
        sparse_output=False,
    )

    # Numerical scaler
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ("num", num_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and engineer features before passing to ColumnTransformer.

    Args:
        df: Input DataFrame with flight records.

    Returns:
        pd.DataFrame: DataFrame containing ALL_MODELING_FEATURES columns.
    """
    # 1. Feature engineering
    df_eng = engineer_preflight_features(df)

    # 2. Standardize Airline, Origin, Dest
    if "Airline" not in df_eng.columns:
        for cand in ["Marketing_Airline_Network", "Airline Network"]:
            if cand in df_eng.columns:
                df_eng["Airline"] = df_eng[cand].astype(str).str.strip()
                break
        if "Airline" not in df_eng.columns:
            df_eng["Airline"] = "UNKNOWN"

    if "Origin" not in df_eng.columns:
        for cand in ["Origin Code"]:
            if cand in df_eng.columns:
                df_eng["Origin"] = df_eng[cand].astype(str).str.strip()
                break
        if "Origin" not in df_eng.columns:
            df_eng["Origin"] = "UNKNOWN"

    if "Dest" not in df_eng.columns:
        for cand in ["Destination Code"]:
            if cand in df_eng.columns:
                df_eng["Dest"] = df_eng[cand].astype(str).str.strip()
                break
        if "Dest" not in df_eng.columns:
            df_eng["Dest"] = "UNKNOWN"

    # Leakage assertion
    assert_no_leakage(ALL_MODELING_FEATURES)

    # Return only modeling features
    return df_eng[ALL_MODELING_FEATURES].copy()


def fit_preprocessor(df_train: pd.DataFrame) -> Tuple[ColumnTransformer, np.ndarray]:
    """Prepare features and fit preprocessor on training data.

    Args:
        df_train: Raw or cleaned training flight DataFrame.

    Returns:
        Tuple[ColumnTransformer, np.ndarray]: Fitted preprocessor and transformed training array.
    """
    X_train_df = prepare_features(df_train)
    preprocessor = build_preprocessor()
    logger.info("Fitting preprocessor on training data (shape=%s)...", X_train_df.shape)
    X_train_transformed = preprocessor.fit_transform(X_train_df)
    logger.info("Transformed feature matrix shape: %s", X_train_transformed.shape)
    return preprocessor, X_train_transformed


def transform_single_flight(
    flight_dict: Dict[str, Any],
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Transform a single flight input dictionary (e.g. from Streamlit) for inference.

    Args:
        flight_dict: Single flight specification.
        preprocessor: Fitted ColumnTransformer.

    Returns:
        np.ndarray: Transformed 2D feature array ready for model.predict().
    """
    df_single = pd.DataFrame([flight_dict])
    X_df = prepare_features(df_single)
    return preprocessor.transform(X_df)


def save_preprocessor(preprocessor: ColumnTransformer, output_path: Union[str, Path]) -> Path:
    """Serialize fitted preprocessor to disk.

    Args:
        preprocessor: Fitted ColumnTransformer.
        output_path: File path to save.

    Returns:
        Path: Verified file path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, p)
    logger.info("Saved preprocessor to %s", p)
    return p


def load_preprocessor(input_path: Union[str, Path]) -> ColumnTransformer:
    """Deserialize fitted preprocessor from disk.

    Args:
        input_path: File path to load.

    Returns:
        ColumnTransformer: Deserialized ColumnTransformer.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {p}")
    return joblib.load(p)
