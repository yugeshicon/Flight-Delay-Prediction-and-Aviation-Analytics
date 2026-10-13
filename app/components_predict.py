"""Streamlit UI component for Flight Delay Prediction.

Renders an interactive flight input form, executes real-time prediction,
displays probability gauges, estimated delay duration, and risk drivers.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.predict import predict_flight_delay
from src.analytics import AIRLINE_NAMES

# Top 50 major US airports for quick dropdown selection
POPULAR_AIRPORTS = [
    ("ATL", "Atlanta - Hartsfield-Jackson (ATL)"),
    ("ORD", "Chicago O'Hare (ORD)"),
    ("DFW", "Dallas/Fort Worth (DFW)"),
    ("DEN", "Denver International (DEN)"),
    ("CLT", "Charlotte Douglas (CLT)"),
    ("LAX", "Los Angeles International (LAX)"),
    ("LGA", "New York LaGuardia (LGA)"),
    ("PHX", "Phoenix Sky Harbor (PHX)"),
    ("SEA", "Seattle-Tacoma (SEA)"),
    ("LAS", "Las Vegas Harry Reid (LAS)"),
    ("MCO", "Orlando International (MCO)"),
    ("BOS", "Boston Logan (BOS)"),
    ("EWR", "Newark Liberty (EWR)"),
    ("DTW", "Detroit Metropolitan (DTW)"),
    ("MSP", "Minneapolis-St. Paul (MSP)"),
    ("SFO", "San Francisco International (SFO)"),
    ("PHL", "Philadelphia International (PHL)"),
    ("SLC", "Salt Lake City (SLC)"),
    ("BWI", "Baltimore/Washington (BWI)"),
    ("FLL", "Fort Lauderdale-Hollywood (FLL)"),
    ("IAD", "Washington Dulles (IAD)"),
    ("DCA", "Ronald Reagan Washington National (DCA)"),
    ("SAN", "San Diego International (SAN)"),
    ("TPA", "Tampa International (TPA)"),
    ("MDW", "Chicago Midway (MDW)"),
    ("BNA", "Nashville International (BNA)"),
    ("AUS", "Austin-Bergstrom (AUS)"),
    ("DAL", "Dallas Love Field (DAL)"),
    ("JFK", "New York JFK (JFK)"),
    ("IAH", "Houston George Bush (IAH)"),
    ("MIA", "Miami International (MIA)"),
    ("STL", "St. Louis Lambert (STL)"),
    ("HOU", "Houston William P. Hobby (HOU)"),
    ("PDX", "Portland International (PDX)"),
    ("MSY", "New Orleans Louis Armstrong (MSY)"),
    ("RDU", "Raleigh-Durham (RDU)"),
    ("SMF", "Sacramento International (SMF)"),
    ("SJC", "San Jose Mineta (SJC)"),
    ("SAT", "San Antonio International (SAT)"),
    ("CLE", "Cleveland Hopkins (CLE)"),
]

# Estimated standard route distances for auto-fill
DEFAULT_ROUTE_DISTANCES = {
    ("ATL", "LGA"): 762,
    ("ORD", "LGA"): 733,
    ("LAX", "JFK"): 2475,
    ("SFO", "JFK"): 2586,
    ("ORD", "DEN"): 888,
    ("ATL", "MCO"): 404,
    ("DFW", "ORD"): 802,
    ("SEA", "SFO"): 679,
    ("BOS", "DCA"): 399,
    ("DFW", "MIA"): 1120,
    ("LAX", "ORD"): 1745,
    ("DEN", "PHX"): 602,
}


def render_prediction_page():
    """Render the full Flight Delay Prediction page."""
    st.markdown("## ✈️ Flight Delay Predictor")
    st.markdown(
        "Enter planned flight details to obtain instant machine learning delay risk classification, "
        "calibrated probability scores, and estimated delay duration."
    )

    with st.form("flight_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🏢 Carrier & Route")
            airline_options = list(AIRLINE_NAMES.keys())
            selected_airline = st.selectbox(
                "Operating Airline",
                airline_options,
                format_func=lambda x: f"{x} — {AIRLINE_NAMES.get(x, x)}",
                index=1,  # DL default
                help="Select the marketing/operating carrier.",
            )

            airport_codes = [code for code, _ in POPULAR_AIRPORTS]
            selected_origin = st.selectbox(
                "Origin Airport",
                airport_codes,
                format_func=lambda x: dict(POPULAR_AIRPORTS).get(x, x),
                index=0,  # ATL
                help="Departure airport (3-letter IATA code).",
            )

            dest_default_idx = 6 if len(airport_codes) > 6 else 1  # LGA
            selected_dest = st.selectbox(
                "Destination Airport",
                airport_codes,
                format_func=lambda x: dict(POPULAR_AIRPORTS).get(x, x),
                index=dest_default_idx,
                help="Arrival airport (3-letter IATA code).",
            )

        with col2:
            st.markdown("#### 🕒 Schedule & Departure")
            dep_hour = st.slider("Scheduled Departure Hour", 0, 23, 14, help="24-hour format (e.g., 14 = 2:00 PM).")
            dep_minute = st.slider("Scheduled Departure Minute", 0, 59, 30, step=5)

            col_time1, col_time2 = st.columns(2)
            with col_time1:
                day_of_month = st.number_input("Day of Month (Jan 2022)", min_value=1, max_value=31, value=15)
            with col_time2:
                days_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
                day_of_week = st.selectbox(
                    "Day of Week",
                    list(days_map.keys()),
                    format_func=lambda x: f"{x} - {days_map[x]}",
                    index=4,  # Friday
                )

        with col3:
            st.markdown("#### 📏 Flight Parameters")
            # Auto-lookup distance or fallback to realistic default
            default_dist = DEFAULT_ROUTE_DISTANCES.get(
                (selected_origin, selected_dest),
                DEFAULT_ROUTE_DISTANCES.get((selected_dest, selected_origin), 750)
            )
            distance = st.number_input("Route Distance (miles)", min_value=50, max_value=5000, value=default_dist)

            # Estimated flight duration based on 450 mph + 45 min taxi/climb
            est_elapsed = max(45, int(distance / 7.5 + 45))
            crs_elapsed = st.number_input("Scheduled Duration (mins)", min_value=30, max_value=600, value=est_elapsed)

            threshold_pct = st.slider(
                "Decision Threshold (%)",
                min_value=30,
                max_value=70,
                value=50,
                help="Probability cutoff for classifying as Delayed. Lower thresholds increase sensitivity.",
            )

        submitted = st.form_submit_button("🔮 Predict Flight Delay", use_container_width=True, type="primary")

    if submitted:
        if selected_origin == selected_dest:
            st.error("⚠️ Origin and Destination airports cannot be identical. Please select different airports.")
            return

        with st.spinner("Executing calibrated inference pipeline..."):
            crs_dep_time = dep_hour * 100 + dep_minute
            flight_payload = {
                "Airline": selected_airline,
                "Origin": selected_origin,
                "Dest": selected_dest,
                "CRSDepTime": crs_dep_time,
                "DayOfWeek": day_of_week,
                "DayofMonth": day_of_month,
                "Distance": float(distance),
                "CRSElapsedTime": float(crs_elapsed),
            }

            result = predict_flight_delay(flight_payload, threshold=threshold_pct / 100.0)

        # Render Results Card
        st.markdown("---")
        st.markdown("### 📋 Prediction Analysis Result")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if result["status"] == "Delayed":
                st.error(
                    f"### 🔴 Predicted Status: DELAYED\n"
                    f"**Arrival Delay >= 15 minutes expected**\n\n"
                    f"⏱️ **Estimated Delay Duration: ~{result['estimated_delay_minutes']} minutes**"
                )
            else:
                st.success(
                    f"### 🟢 Predicted Status: ON-TIME\n"
                    f"**Flight expected to arrive on schedule (< 15 mins delay)**\n\n"
                    f"⏱️ **Estimated Delay Duration: Minimal (0–14 mins)**"
                )

            # Gauge Chart for Probability
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["delay_probability_pct"],
                title={'text': "Delay Risk Probability (%)", 'font': {'size': 18}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#d9534f" if result["is_delayed"] else "#5cb85c"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(92, 184, 92, 0.2)"},
                        {'range': [40, 65], 'color': "rgba(240, 173, 78, 0.2)"},
                        {'range': [65, 100], 'color': "rgba(217, 83, 79, 0.2)"},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': threshold_pct,
                    }
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("#### 🔍 Operational Risk Breakdown")
            st.metric("Risk Level", result["risk_level"])
            st.markdown(f"**Delayed Probability:** `{result['delay_probability_pct']}%`")
            st.markdown(f"**On-Time Probability:** `{result['ontime_probability_pct']}%`")

            st.markdown("##### 📌 Key Influencing Drivers:")
            for driver in result["key_drivers"]:
                st.markdown(f"- {driver}")

            st.info(
                f"ℹ️ **Operational Context**: Route {selected_origin} $\\to$ {selected_dest} with {AIRLINE_NAMES.get(selected_airline, selected_airline)} "
                f"scheduled for {dep_hour:02d}:{dep_minute:02d}. Classification applied at a {threshold_pct}% decision threshold."
            )
