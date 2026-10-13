"""Data cleaning and leakage prevention module.

Provides rigorous data hygiene, cancellation/diversion handling,
cause null imputation, target creation, and pre-flight feature segregation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical delay cause columns
CAUSE_COLS_BTS = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
CAUSE_COLS_CLEANED = ["Carrier Delay", "Weather Delay", "NAS Delay", "Security Delay", "Aircraft Delay"]

# Explicit list of post-departure/post-arrival columns that MUST NOT be used for pre-flight prediction
LEAKAGE_COLUMNS: Set[str] = {
    # Actual departure info
    "DepTime", "Actual Depart Time", "Actual_Depart_Time",
    "DepDelay", "Depart Delay (Mins)", "Depart_Delay_(Mins)",
    "DepDelayMinutes", "Depart Delay Mins", "Depart_Delay_Mins",
    "DepDel15", "Depart Delay 15 Mins", "Depart_Delay_15_Mins",
    "DepartureDelayGroups", "Depart Delay Group", "Depart_Delay_Group",
    "TaxiOut", "Taxi-out", "Taxi_Out",
    "WheelsOff", "Wheels Off", "Wheels_Off",
    # En route & arrival info
    "WheelsOn", "Wheels On", "Wheels_On",
    "TaxiIn", "Taxi-in", "Taxi_In",
    "ArrTime", "Actual Arrival Time", "Actual_Arrival_Time",
    "ArrDelay", "Arrival Delay (Mins)", "Arrival_Delay_(Mins)",
    "ArrDelayMinutes", "Arrival Delay Mins", "Arrival_Delay_Mins",
    "ArrDel15", "Arrival Delay 15 Mins", "Arrival_Delay_15_Mins",
    "ArrivalDelayGroups", "Arrival Delay Group", "Arrival_Delay_Group",
    "ActualElapsedTime", "Actual Elapsed Time", "Actual_Elapsed_Time",
    "AirTime", "Air Time", "Air_Time",
    # Post-outcome delay causes
    "CarrierDelay", "Carrier Delay", "Carrier_Delay",
    "WeatherDelay", "Weather Delay", "Weather_Delay",
    "NASDelay", "NAS Delay", "NAS_Delay",
    "SecurityDelay", "Security Delay", "Security_Delay",
    "LateAircraftDelay", "Aircraft Delay", "Aircraft_Delay",
    # Target / Post-outcome labels
    "Flight Status", "Flight_Status", "CancellationCode",
}


def assert_no_leakage(feature_cols: List[str]) -> bool:
    """Validate that feature matrix does not contain any post-departure columns.

    Args:
        feature_cols: List of column names planned for modeling.

    Returns:
        bool: True if strictly leakage-free.

    Raises:
        ValueError: If any post-departure feature is detected.
    """
    detected_leakage = [col for col in feature_cols if col.strip() in LEAKAGE_COLUMNS]
    if detected_leakage:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! The following post-departure columns were found in features: {detected_leakage}"
        )
    return True


def clean_flight_data(df: pd.DataFrame, for_modeling: bool = False) -> pd.DataFrame:
    """Clean and standardize flight records with domain-specific rules.

    Args:
        df: Input raw or selected flight DataFrame.
        for_modeling: If True, filters strictly to completed flights and builds targets.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("Cleaning flight data (rows=%d, for_modeling=%s)...", len(df), for_modeling)
    # Fast bypass if DataFrame is already cleaned and standardized
    if not for_modeling and "Flight_State" in df.columns and "Airline" in df.columns:
        return df

    df_clean = df.copy()

    # Strip column names
    df_clean.columns = [c.strip() for c in df_clean.columns]

    # Standardize Airline column
    if "Marketing_Airline_Network" in df_clean.columns:
        df_clean["Airline"] = df_clean["Marketing_Airline_Network"].astype(str).str.strip()
    elif "Airline Network" in df_clean.columns:
        df_clean["Airline"] = df_clean["Airline Network"].astype(str).str.strip()
    elif "Airline" in df_clean.columns:
        df_clean["Airline"] = df_clean["Airline"].astype(str).str.strip()

    # Standardize Origin & Dest
    if "Origin" in df_clean.columns:
        df_clean["Origin"] = df_clean["Origin"].astype(str).str.strip()
    elif "Origin Code" in df_clean.columns:
        df_clean["Origin"] = df_clean["Origin Code"].astype(str).str.strip()

    if "Dest" in df_clean.columns:
        df_clean["Dest"] = df_clean["Dest"].astype(str).str.strip()
    elif "Destination Code" in df_clean.columns:
        df_clean["Dest"] = df_clean["Destination Code"].astype(str).str.strip()

    # Standardize Tail Number
    tail_col = "Tail_Number" if "Tail_Number" in df_clean.columns else "Tail Number"
    if tail_col in df_clean.columns:
        df_clean["Tail_Number"] = df_clean[tail_col].fillna("UNKNOWN").astype(str).str.strip()

    # Assign comprehensive Flight State
    cancelled_mask = df_clean["Cancelled"] == 1
    diverted_mask = df_clean["Diverted"] == 1
    df_clean["Flight_State"] = "Completed"
    df_clean.loc[cancelled_mask, "Flight_State"] = "Cancelled"
    df_clean.loc[diverted_mask, "Flight_State"] = "Diverted"

    # Impute delay causes for on-time completed flights
    # In BTS data, delay causes are only populated if ArrDel15 == 1.
    # For on-time flights, cause delays are 0.0.
    arr_del_col = None
    for cand in ["ArrDel15", "Arrival Delay 15 Mins"]:
        if cand in df_clean.columns:
            arr_del_col = cand
            break

    cause_cols_present = [c for c in CAUSE_COLS_BTS + CAUSE_COLS_CLEANED if c in df_clean.columns]
    for col in cause_cols_present:
        # Fill completed on-time flights with 0.0
        if arr_del_col:
            ontime_mask = (df_clean["Flight_State"] == "Completed") & (df_clean[arr_del_col] == 0)
            df_clean.loc[ontime_mask, col] = 0.0
        # Fill remaining missing with 0.0
        df_clean[col] = df_clean[col].fillna(0.0)

    # If requested for modeling, filter to completed flights and create canonical targets
    if for_modeling:
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean["Flight_State"] == "Completed"].copy()
        logger.info(
            "Filtered to completed flights for modeling: %d -> %d (removed %d cancelled/diverted)",
            initial_len,
            len(df_clean),
            initial_len - len(df_clean),
        )

        # Standardize targets
        # Classification Target: is_delayed (1 if arrival delay >= 15 min, else 0)
        if "ArrDel15" in df_clean.columns:
            df_clean["is_delayed"] = (df_clean["ArrDel15"] == 1.0).astype(int)
        elif "Arrival Delay 15 Mins" in df_clean.columns:
            df_clean["is_delayed"] = (df_clean["Arrival Delay 15 Mins"] == 1.0).astype(int)
        elif "ArrDelay" in df_clean.columns:
            df_clean["is_delayed"] = (df_clean["ArrDelay"] >= 15.0).astype(int)
        elif "Arrival Delay (Mins)" in df_clean.columns:
            df_clean["is_delayed"] = (df_clean["Arrival Delay (Mins)"] >= 15.0).astype(int)
        else:
            raise KeyError("Cannot derive 'is_delayed' target: no arrival delay indicator found.")

        # Regression Target: delay_minutes (actual arrival delay in minutes, non-negative)
        arr_delay_val_col = None
        for cand in ["ArrDelay", "Arrival Delay (Mins)", "ArrDelayMinutes", "Arrival Delay Mins"]:
            if cand in df_clean.columns:
                arr_delay_val_col = cand
                break
        
        if arr_delay_val_col:
            df_clean["delay_minutes"] = np.maximum(0.0, df_clean[arr_delay_val_col].fillna(0.0).astype(float))
        else:
            df_clean["delay_minutes"] = 0.0

    return df_clean


def extract_features_and_targets(
    df_clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract strictly pre-flight feature matrix X and targets y_cls, y_reg.

    Args:
        df_clean: Cleaned DataFrame with for_modeling=True.

    Returns:
        Tuple: (X, y_cls, y_reg)
    """
    if "is_delayed" not in df_clean.columns or "delay_minutes" not in df_clean.columns:
        raise ValueError("DataFrame must be cleaned with for_modeling=True before extracting targets.")

    y_cls = df_clean["is_delayed"].copy()
    y_reg = df_clean["delay_minutes"].copy()

    # Determine pre-flight features available
    candidate_features = [
        "Airline",
        "Origin",
        "Dest",
        "Distance",
        "CRSElapsedTime",
        "Scheduled Elapsed Time",
        "DayOfWeek",
        "Day Of Week",
        "DayofMonth",
        "Day Of Month",
        "CRSDepTime",
        "Scheduled Depart Time",
        "CRSArrTime",
        "Scheduled Arrival Time",
    ]

    selected_cols = []
    for col in candidate_features:
        if col in df_clean.columns and col not in selected_cols:
            selected_cols.append(col)

    # Double-check against leakage
    assert_no_leakage(selected_cols)

    X = df_clean[selected_cols].copy()
    logger.info("Extracted pre-flight features matrix X: shape %s", X.shape)
    return X, y_cls, y_reg
