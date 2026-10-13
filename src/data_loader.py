"""Data loader module for Flight Delay Prediction and Aviation Analytics.

Provides optimized loading, caching, and sampling of flight datasets from
the project's Dataset/ directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

DATASET_FILES = {
    "original": DATASET_DIR / "Flight Delay Original Dataset.csv",
    "selected": DATASET_DIR / "Flight Delay Selected Dataset.csv",
    "cleaned": DATASET_DIR / "Flight Delay Cleaned Dataset.csv",
}

# Optimized dtypes to reduce memory footprint by >60%
OPTIMIZED_DTYPES = {
    "Year": "int16",
    "Quarter": "int8",
    "Month": "int8",
    "Day Of Month": "int8",
    "Day Of Week": "int8",
    "DayofMonth": "int8",
    "DayOfWeek": "int8",
    "Airline Network": "category",
    "Operating Airline": "category",
    "Operating_Airline ": "category",
    "Marketing_Airline_Network": "category",
    "Origin Code": "category",
    "Destination Code": "category",
    "Origin": "category",
    "Dest": "category",
    "Origin State": "category",
    "Destination State": "category",
    "Cancelled": "int8",
    "Diverted": "int8",
    "Distance Group": "int8",
    "DistanceGroup": "int8",
    "Depart Delay 15 Mins": "float32",
    "Arrival Delay 15 Mins": "float32",
    "DepDel15": "float32",
    "ArrDel15": "float32",
    "Depart Delay (Mins)": "float32",
    "Arrival Delay (Mins)": "float32",
    "DepDelay": "float32",
    "ArrDelay": "float32",
    "Carrier Delay": "float32",
    "Weather Delay": "float32",
    "NAS Delay": "float32",
    "Security Delay": "float32",
    "Aircraft Delay": "float32",
    "CarrierDelay": "float32",
    "WeatherDelay": "float32",
    "NASDelay": "float32",
    "SecurityDelay": "float32",
    "LateAircraftDelay": "float32",
    "Flight Status": "category",
}


def get_dataset_path(dataset_type: str = "selected") -> Path:
    """Retrieve absolute file path for a specified dataset type.

    Args:
        dataset_type: One of 'selected', 'cleaned', or 'original'.

    Returns:
        Path: Verified filesystem path to the CSV file.

    Raises:
        FileNotFoundError: If the dataset file does not exist on disk.
    """
    if dataset_type not in DATASET_FILES:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Options: {list(DATASET_FILES.keys())}")
    
    path = DATASET_FILES[dataset_type]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    return path


def load_flight_data(
    dataset_type: str = "selected",
    nrows: Optional[int] = None,
    use_optimized_dtypes: bool = True,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load flight data with optimized memory footprint and optional sampling.

    Args:
        dataset_type: 'selected' (recommended default), 'cleaned', or 'original'.
        nrows: Optional maximum number of rows to load (useful for fast testing).
        use_optimized_dtypes: Whether to downcast dtypes for memory optimization.
        columns: Optional list of specific columns to load.

    Returns:
        pd.DataFrame: Loaded flight data DataFrame.
    """
    path = get_dataset_path(dataset_type)
    logger.info("Loading flight data from %s (nrows=%s)...", path.name, nrows or "ALL")

    # Select available dtypes for the requested columns
    dtypes_to_use = None
    if use_optimized_dtypes:
        dtypes_to_use = {col: dtype for col, dtype in OPTIMIZED_DTYPES.items()}

    # Handle duplicate columns if loading 'cleaned' dataset
    if dataset_type == "cleaned":
        # Read header first to deduplicate
        header = pd.read_csv(path, nrows=0).columns.tolist()
        seen = {}
        dedup_cols = []
        for col in header:
            if col in seen:
                seen[col] += 1
                dedup_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                dedup_cols.append(col)
        
        df = pd.read_csv(
            path,
            nrows=nrows,
            usecols=columns,
            low_memory=False,
        )
        if len(df.columns) == len(dedup_cols):
            df.columns = dedup_cols
    else:
        df = pd.read_csv(
            path,
            nrows=nrows,
            usecols=columns,
            low_memory=False,
        )

    # Clean whitespace in column names (e.g. 'Operating_Airline ')
    df.columns = [c.strip() for c in df.columns]

    # Apply type casting safely
    if use_optimized_dtypes:
        for col, dtype in OPTIMIZED_DTYPES.items():
            col_clean = col.strip()
            if col_clean in df.columns:
                try:
                    df[col_clean] = df[col_clean].astype(dtype)
                except Exception as exc:
                    logger.debug("Could not cast column '%s' to %s: %s", col_clean, dtype, exc)

    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info("Loaded %d rows, %d columns (Memory: %.2f MB)", len(df), len(df.columns), mem_usage_mb)
    return df


def get_metadata(df: pd.DataFrame) -> Dict[str, Union[int, float, List[str]]]:
    """Extract operational metadata from a flight dataframe.

    Args:
        df: Input flight dataset.

    Returns:
        Dict: Metadata dictionary with row count, airlines, airports, and status distributions.
    """
    airline_col = "Marketing_Airline_Network" if "Marketing_Airline_Network" in df.columns else "Airline Network"
    origin_col = "Origin" if "Origin" in df.columns else "Origin Code"
    dest_col = "Dest" if "Dest" in df.columns else "Destination Code"

    airlines = sorted(df[airline_col].dropna().unique().tolist()) if airline_col in df.columns else []
    origins = sorted(df[origin_col].dropna().unique().tolist()) if origin_col in df.columns else []
    dests = sorted(df[dest_col].dropna().unique().tolist()) if dest_col in df.columns else []

    total_flights = len(df)
    cancelled_count = int(df["Cancelled"].sum()) if "Cancelled" in df.columns else 0
    diverted_count = int(df["Diverted"].sum()) if "Diverted" in df.columns else 0

    return {
        "total_flights": total_flights,
        "completed_flights": total_flights - (cancelled_count + diverted_count),
        "cancelled_flights": cancelled_count,
        "diverted_flights": diverted_count,
        "unique_airlines_count": len(airlines),
        "unique_airlines": airlines,
        "unique_origins_count": len(origins),
        "unique_destinations_count": len(dests),
    }
