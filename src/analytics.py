"""Aviation Analytics module for Flight Delay Prediction & Analytics Dashboard.

Computes executive KPIs, daily disruption timelines across January 2022,
airline benchmarking, airport/route congestion metrics, and 5-factor delay breakdowns.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.cleaning import clean_flight_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AIRLINE_NAMES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
}


def compute_executive_kpis(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Compute top-level operational KPIs across flight records.

    Args:
        df: Input raw or cleaned flight DataFrame.

    Returns:
        Dict: KPI dictionary with counts and percentages.
    """
    df_clean = clean_flight_data(df, for_modeling=False)
    total_flights = len(df_clean)

    cancelled_count = int((df_clean["Flight_State"] == "Cancelled").sum())
    diverted_count = int((df_clean["Flight_State"] == "Diverted").sum())
    completed_count = int((df_clean["Flight_State"] == "Completed").sum())

    completed_mask = df_clean["Flight_State"] == "Completed"
    completed_df = df_clean[completed_mask]

    # Delayed flights (ArrDel15 == 1)
    arr_del_col = "ArrDel15" if "ArrDel15" in completed_df.columns else "Arrival Delay 15 Mins"
    arr_delay_val_col = "ArrDelay" if "ArrDelay" in completed_df.columns else "Arrival Delay (Mins)"

    if arr_del_col in completed_df.columns:
        delayed_count = int((completed_df[arr_del_col] == 1).sum())
        ontime_count = int((completed_df[arr_del_col] == 0).sum())
    elif arr_delay_val_col in completed_df.columns:
        delayed_count = int((completed_df[arr_delay_val_col] >= 15.0).sum())
        ontime_count = completed_count - delayed_count
    else:
        delayed_count = 0
        ontime_count = completed_count

    avg_arrival_delay = float(completed_df[arr_delay_val_col].mean()) if arr_delay_val_col in completed_df.columns else 0.0
    
    delayed_flights = completed_df[completed_df[arr_del_col] == 1] if arr_del_col in completed_df.columns else completed_df
    avg_delayed_delay = float(delayed_flights[arr_delay_val_col].mean()) if arr_delay_val_col in delayed_flights.columns and len(delayed_flights) > 0 else 0.0

    return {
        "total_flights": total_flights,
        "completed_flights": completed_count,
        "cancelled_flights": cancelled_count,
        "diverted_flights": diverted_count,
        "cancellation_rate_pct": round((cancelled_count / total_flights * 100) if total_flights else 0.0, 2),
        "diverted_rate_pct": round((diverted_count / total_flights * 100) if total_flights else 0.0, 2),
        "ontime_flights": ontime_count,
        "ontime_rate_pct": round((ontime_count / completed_count * 100) if completed_count else 0.0, 2),
        "delayed_flights": delayed_count,
        "delayed_rate_pct": round((delayed_count / completed_count * 100) if completed_count else 0.0, 2),
        "avg_arrival_delay_mins": round(avg_arrival_delay, 2),
        "avg_delayed_arrival_delay_mins": round(avg_delayed_delay, 2),
    }


def compute_daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily flight counts, cancellations, delays, and delay rate across January.

    Args:
        df: Input flight DataFrame.

    Returns:
        pd.DataFrame: Table with Day_Of_Month, total_flights, cancelled, delayed, delay_rate_pct.
    """
    df_clean = clean_flight_data(df, for_modeling=False)
    dom_col = "DayofMonth" if "DayofMonth" in df_clean.columns else "Day Of Month"
    arr_del_col = "ArrDel15" if "ArrDel15" in df_clean.columns else "Arrival Delay 15 Mins"

    records = []
    for day, group in df_clean.groupby(dom_col):
        total = len(group)
        cancelled = int((group["Flight_State"] == "Cancelled").sum())
        diverted = int((group["Flight_State"] == "Diverted").sum())
        completed = total - (cancelled + diverted)
        delayed = int(((group["Flight_State"] == "Completed") & (group[arr_del_col] == 1)).sum()) if arr_del_col in group.columns else 0
        ontime = completed - delayed
        delay_rate = round((delayed / completed * 100) if completed > 0 else 0.0, 2)
        cancel_rate = round((cancelled / total * 100) if total > 0 else 0.0, 2)

        records.append({
            "Day": int(day),
            "Total_Flights": total,
            "Completed": completed,
            "Cancelled": cancelled,
            "Diverted": diverted,
            "Delayed": delayed,
            "On_Time": ontime,
            "Delay_Rate_Pct": delay_rate,
            "Cancellation_Rate_Pct": cancel_rate,
        })

    trend_df = pd.DataFrame(records).sort_values("Day").reset_index(drop=True)
    return trend_df


def compute_airline_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Benchmark performance across all operating airlines.

    Args:
        df: Input flight DataFrame.

    Returns:
        pd.DataFrame: Performance leaderboard per airline.
    """
    df_clean = clean_flight_data(df, for_modeling=False)
    arr_del_col = "ArrDel15" if "ArrDel15" in df_clean.columns else "Arrival Delay 15 Mins"
    arr_delay_val_col = "ArrDelay" if "ArrDelay" in df_clean.columns else "Arrival Delay (Mins)"

    benchmarks = []
    for airline_code, group in df_clean.groupby("Airline"):
        total = len(group)
        cancelled = int((group["Flight_State"] == "Cancelled").sum())
        diverted = int((group["Flight_State"] == "Diverted").sum())
        completed = total - (cancelled + diverted)
        delayed = int(((group["Flight_State"] == "Completed") & (group[arr_del_col] == 1)).sum()) if arr_del_col in group.columns else 0
        ontime = completed - delayed

        delay_rate = round((delayed / completed * 100) if completed > 0 else 0.0, 2)
        cancel_rate = round((cancelled / total * 100) if total > 0 else 0.0, 2)
        ontime_rate = round((ontime / completed * 100) if completed > 0 else 0.0, 2)

        completed_group = group[group["Flight_State"] == "Completed"]
        avg_delay = round(float(completed_group[arr_delay_val_col].mean()), 2) if arr_delay_val_col in completed_group.columns and len(completed_group) > 0 else 0.0

        benchmarks.append({
            "Airline": str(airline_code),
            "Airline_Name": AIRLINE_NAMES.get(str(airline_code), str(airline_code)),
            "Total_Flights": total,
            "Completed": completed,
            "Cancelled": cancelled,
            "Cancellation_Rate_Pct": cancel_rate,
            "On_Time": ontime,
            "On_Time_Rate_Pct": ontime_rate,
            "Delayed": delayed,
            "Delay_Rate_Pct": delay_rate,
            "Avg_Arrival_Delay_Mins": avg_delay,
        })

    bench_df = pd.DataFrame(benchmarks).sort_values("Total_Flights", ascending=False).reset_index(drop=True)
    return bench_df


def compute_airport_analytics(
    df: pd.DataFrame,
    min_flights: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze busiest and most delayed origin airports.

    Args:
        df: Input flight DataFrame.
        min_flights: Minimum flight threshold for ranking delayed airports.

    Returns:
        Tuple: (busiest_airports_df, most_delayed_airports_df)
    """
    df_clean = clean_flight_data(df, for_modeling=False)
    arr_del_col = "ArrDel15" if "ArrDel15" in df_clean.columns else "Arrival Delay 15 Mins"
    arr_delay_val_col = "ArrDelay" if "ArrDelay" in df_clean.columns else "Arrival Delay (Mins)"

    records = []
    for origin, group in df_clean.groupby("Origin"):
        total = len(group)
        cancelled = int((group["Flight_State"] == "Cancelled").sum())
        completed = int((group["Flight_State"] == "Completed").sum())
        delayed = int(((group["Flight_State"] == "Completed") & (group[arr_del_col] == 1)).sum()) if arr_del_col in group.columns else 0
        ontime = completed - delayed

        delay_rate = round((delayed / completed * 100) if completed > 0 else 0.0, 2)
        completed_grp = group[group["Flight_State"] == "Completed"]
        avg_delay = round(float(completed_grp[arr_delay_val_col].mean()), 2) if arr_delay_val_col in completed_grp.columns and len(completed_grp) > 0 else 0.0

        records.append({
            "Origin": str(origin),
            "Total_Flights": total,
            "Completed": completed,
            "Cancelled": cancelled,
            "Delayed": delayed,
            "Delay_Rate_Pct": delay_rate,
            "Avg_Arrival_Delay_Mins": avg_delay,
        })

    all_airports = pd.DataFrame(records)
    busiest = all_airports.sort_values("Total_Flights", ascending=False).head(20).reset_index(drop=True)
    
    # Filter by threshold for delay rankings
    qualified = all_airports[all_airports["Total_Flights"] >= min_flights]
    most_delayed = qualified.sort_values("Delay_Rate_Pct", ascending=False).head(20).reset_index(drop=True)
    return busiest, most_delayed


def compute_dep_vs_arr_dynamics(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """Compute correlation, recovery rate, and delay compounding between departure and arrival.

    Args:
        df: Input flight DataFrame.

    Returns:
        Dict: Quantitative relationship dynamics.
    """
    df_clean = clean_flight_data(df, for_modeling=True)
    dep_delay_col = "DepDelay" if "DepDelay" in df_clean.columns else "Depart Delay (Mins)"
    arr_delay_col = "ArrDelay" if "ArrDelay" in df_clean.columns else "Arrival Delay (Mins)"

    valid = df_clean.dropna(subset=[dep_delay_col, arr_delay_col])
    if len(valid) == 0:
        return {"correlation": 0.0, "recovery_rate_pct": 0.0, "compounding_rate_pct": 0.0}

    dep_vals = valid[dep_delay_col].values
    arr_vals = valid[arr_delay_col].values

    corr = float(np.corrcoef(dep_vals, arr_vals)[0, 1])

    # Minor departure delay (1-14 min late)
    minor_dep_delayed = valid[(valid[dep_delay_col] >= 1.0) & (valid[dep_delay_col] < 15.0)]
    recovered = (minor_dep_delayed[arr_delay_col] <= 0.0).sum() if len(minor_dep_delayed) > 0 else 0
    recovery_rate = round((recovered / len(minor_dep_delayed) * 100) if len(minor_dep_delayed) > 0 else 0.0, 2)

    # Major departure delay (>= 30 mins) that stay delayed >= 30 mins
    major_dep_delayed = valid[valid[dep_delay_col] >= 30.0]
    still_delayed = (major_dep_delayed[arr_delay_col] >= 30.0).sum() if len(major_dep_delayed) > 0 else 0
    compounding_rate = round((still_delayed / len(major_dep_delayed) * 100) if len(major_dep_delayed) > 0 else 0.0, 2)

    return {
        "correlation": round(corr, 4),
        "minor_delayed_flights_evaluated": int(len(minor_dep_delayed)),
        "flights_recovered_ontime": int(recovered),
        "recovery_rate_pct": recovery_rate,
        "major_delayed_flights_evaluated": int(len(major_dep_delayed)),
        "compounding_rate_pct": compounding_rate,
    }


def compute_delay_causes_breakdown(df: pd.DataFrame) -> Dict[str, Any]:
    """Quantify the 5 official FAA/BTS delay categories across delayed flights.

    Args:
        df: Input flight DataFrame.

    Returns:
        Dict: Breakdown containing total minutes, affected flights, and percentage share.
    """
    df_clean = clean_flight_data(df, for_modeling=True)
    delayed = df_clean[df_clean["is_delayed"] == 1]

    causes = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
    # Fallback to cleaned column names if needed
    if "CarrierDelay" not in delayed.columns and "Carrier Delay" in delayed.columns:
        causes = ["Carrier Delay", "Weather Delay", "NAS Delay", "Security Delay", "Aircraft Delay"]

    total_delayed_count = len(delayed)
    cause_data = {}
    grand_total_minutes = 0.0

    for cause in causes:
        if cause in delayed.columns:
            tot_mins = float(delayed[cause].sum())
            affected_count = int((delayed[cause] > 0).sum())
            avg_mins = round((tot_mins / affected_count) if affected_count > 0 else 0.0, 1)
        else:
            tot_mins = 0.0
            affected_count = 0
            avg_mins = 0.0

        cause_data[cause] = {
            "total_minutes": round(tot_mins, 1),
            "affected_flights": affected_count,
            "affected_pct": round((affected_count / total_delayed_count * 100) if total_delayed_count > 0 else 0.0, 2),
            "avg_minutes_when_present": avg_mins,
        }
        grand_total_minutes += tot_mins

    for cause, d in cause_data.items():
        d["share_of_delay_minutes_pct"] = round((d["total_minutes"] / grand_total_minutes * 100) if grand_total_minutes > 0 else 0.0, 2)

    return {
        "total_delayed_flights": total_delayed_count,
        "grand_total_delay_minutes": round(grand_total_minutes, 1),
        "causes": cause_data,
    }
