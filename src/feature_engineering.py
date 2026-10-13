"""Feature engineering module for Flight Delay Prediction.

Derives scheduled temporal features, rush-hour indicators, and network
congestion statistics from pre-flight attributes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def engineer_preflight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer strictly pre-flight predictive features from flight data.

    Args:
        df: Input DataFrame containing pre-flight flight attributes.

    Returns:
        pd.DataFrame: Augmented DataFrame with engineered features.
    """
    df_feat = df.copy()

    # Scheduled departure time parsing (HHMM format)
    dep_time_col = None
    for cand in ["CRSDepTime", "Scheduled Depart Time", "Scheduled_Depart_Time"]:
        if cand in df_feat.columns:
            dep_time_col = cand
            break

    if dep_time_col:
        # Fill missing with 1200 as default noon
        dep_raw = pd.to_numeric(df_feat[dep_time_col], errors="coerce").fillna(1200).astype(int)
        df_feat["Depart_Hour"] = (dep_raw // 100).clip(0, 23)
        df_feat["Depart_Minute"] = (dep_raw % 100).clip(0, 59)
    elif "Depart Hour" in df_feat.columns:
        df_feat["Depart_Hour"] = pd.to_numeric(df_feat["Depart Hour"], errors="coerce").fillna(12).astype(int).clip(0, 23)
        df_feat["Depart_Minute"] = 0
    else:
        df_feat["Depart_Hour"] = 12
        df_feat["Depart_Minute"] = 0

    # Scheduled arrival time parsing (HHMM format)
    arr_time_col = None
    for cand in ["CRSArrTime", "Scheduled Arrival Time", "Scheduled_Arrival_Time"]:
        if cand in df_feat.columns:
            arr_time_col = cand
            break

    if arr_time_col:
        arr_raw = pd.to_numeric(df_feat[arr_time_col], errors="coerce").fillna(1400).astype(int)
        df_feat["Arrival_Hour"] = (arr_raw // 100).clip(0, 23)
    elif "Arrival Hour" in df_feat.columns:
        df_feat["Arrival_Hour"] = pd.to_numeric(df_feat["Arrival Hour"], errors="coerce").fillna(14).astype(int).clip(0, 23)
    else:
        df_feat["Arrival_Hour"] = (df_feat["Depart_Hour"] + 2) % 24

    # Depart Time Block (4-hour intervals)
    # 0: Night (00-05), 1: Morning (06-11), 2: Afternoon (12-17), 3: Evening (18-23)
    df_feat["Depart_Time_Block"] = pd.cut(
        df_feat["Depart_Hour"],
        bins=[-1, 5, 11, 17, 24],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    ).astype(str)

    # Day of Week & Weekend flag
    dow_col = None
    for cand in ["DayOfWeek", "Day Of Week", "Day_Of_Week"]:
        if cand in df_feat.columns:
            dow_col = cand
            break

    if dow_col:
        dow = pd.to_numeric(df_feat[dow_col], errors="coerce").fillna(1).astype(int)
        df_feat["Day_Of_Week"] = dow
        df_feat["Is_Weekend"] = dow.isin([6, 7]).astype(int)
    else:
        df_feat["Day_Of_Week"] = 1
        df_feat["Is_Weekend"] = 0

    # Day of Month
    dom_col = None
    for cand in ["DayofMonth", "Day Of Month", "Day_Of_Month"]:
        if cand in df_feat.columns:
            dom_col = cand
            break

    if dom_col:
        df_feat["Day_Of_Month"] = pd.to_numeric(df_feat[dom_col], errors="coerce").fillna(15).astype(int).clip(1, 31)
    else:
        df_feat["Day_Of_Month"] = 15

    # Peak Rush Hour indicator (07:00-09:59 or 16:00-19:59)
    df_feat["Peak_Hour"] = df_feat["Depart_Hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Route String
    origin_col = "Origin" if "Origin" in df_feat.columns else "Origin Code"
    dest_col = "Dest" if "Dest" in df_feat.columns else "Destination Code"
    if origin_col in df_feat.columns and dest_col in df_feat.columns:
        df_feat["Route"] = (
            df_feat[origin_col].astype(str).str.strip() + "-" + df_feat[dest_col].astype(str).str.strip()
        )
    else:
        df_feat["Route"] = "UNKNOWN"

    # Distance & Elapsed Time normalization
    dist_col = "Distance" if "Distance" in df_feat.columns else None
    if dist_col:
        df_feat["Distance"] = pd.to_numeric(df_feat[dist_col], errors="coerce").fillna(800.0).astype(float)
    else:
        df_feat["Distance"] = 800.0

    elapsed_col = "CRSElapsedTime" if "CRSElapsedTime" in df_feat.columns else "Scheduled Elapsed Time"
    if elapsed_col in df_feat.columns:
        df_feat["CRSElapsedTime"] = pd.to_numeric(df_feat[elapsed_col], errors="coerce").fillna(140.0).astype(float)
    else:
        df_feat["CRSElapsedTime"] = 140.0

    return df_feat
