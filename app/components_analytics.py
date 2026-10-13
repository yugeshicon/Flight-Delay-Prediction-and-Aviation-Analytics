"""Streamlit UI component for Aviation Analytics Dashboard.

Renders executive KPI cards, daily January timelines, carrier leaderboards,
airport congestion rankings, departure vs arrival dynamics, and 5-factor delay breakdowns.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.analytics import (
    compute_executive_kpis,
    compute_daily_trends,
    compute_airline_benchmarks,
    compute_airport_analytics,
    compute_dep_vs_arr_dynamics,
    compute_delay_causes_breakdown,
    AIRLINE_NAMES,
)


def render_analytics_page(df: pd.DataFrame):
    """Render the full Aviation Analytics Dashboard.

    Args:
        df: Loaded historical flight dataset.
    """
    st.markdown("## 📊 Aviation Analytics Dashboard")
    st.markdown(
        "Empirical performance metrics, daily operational timelines, carrier benchmarks, "
        "and delay root cause breakdowns for US commercial domestic flights (January 2022)."
    )

    # 1. Interactive Sidebar Filters
    st.sidebar.markdown("### 🎛️ Dashboard Filters")

    all_airlines = sorted(df["Marketing_Airline_Network"].dropna().unique().tolist()) if "Marketing_Airline_Network" in df.columns else sorted(df["Airline Network"].dropna().unique().tolist())
    selected_airlines = st.sidebar.multiselect(
        "Filter Airlines",
        options=all_airlines,
        default=all_airlines,
        format_func=lambda x: f"{x} - {AIRLINE_NAMES.get(x, x)}",
    )

    origin_col = "Origin" if "Origin" in df.columns else "Origin Code"
    all_origins = sorted(df[origin_col].dropna().unique().tolist())
    top_origin_defaults = ["ATL", "ORD", "DFW", "DEN", "CLT", "LAX", "LGA", "PHX", "SEA", "LAS"]
    selected_origins = st.sidebar.multiselect(
        "Filter Origin Hubs (Leave empty for All)",
        options=all_origins,
        default=[],
        help="Filter specific airports. If empty, all 369 origin hubs are included.",
    )

    dom_col = "DayofMonth" if "DayofMonth" in df.columns else "Day Of Month"
    day_range = st.sidebar.slider(
        "January Date Range (Days 1–31)",
        min_value=1,
        max_value=31,
        value=(1, 31),
    )

    # Filter dataset based on controls
    airline_col = "Marketing_Airline_Network" if "Marketing_Airline_Network" in df.columns else "Airline Network"
    filtered_df = df[
        (df[airline_col].isin(selected_airlines)) &
        (df[dom_col] >= day_range[0]) &
        (df[dom_col] <= day_range[1])
    ]
    if selected_origins:
        filtered_df = filtered_df[filtered_df[origin_col].isin(selected_origins)]

    if len(filtered_df) == 0:
        st.warning("⚠️ No flights match the selected filter criteria. Please broaden your selection.")
        return

    # 2. Executive KPI Cards
    kpis = compute_executive_kpis(filtered_df)

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        st.metric("Total Flights", f"{kpis['total_flights']:,}")
    with kpi_col2:
        st.metric("Completed Flights", f"{kpis['completed_flights']:,}", f"{100 - kpis['cancellation_rate_pct']:.1f}%")
    with kpi_col3:
        st.metric("Cancelled Flights", f"{kpis['cancelled_flights']:,}", f"-{kpis['cancellation_rate_pct']}%", delta_color="inverse")
    with kpi_col4:
        st.metric("On-Time Rate", f"{kpis['ontime_rate_pct']}%", f"{kpis['ontime_flights']:,}")
    with kpi_col5:
        st.metric("Delay Rate (≥15m)", f"{kpis['delayed_rate_pct']}%", f"{kpis['delayed_flights']:,}", delta_color="inverse")
    with kpi_col6:
        st.metric("Avg Arrival Delay", f"{kpis['avg_arrival_delay_mins']:+.1f} m", f"Delayed: {kpis['avg_delayed_arrival_delay_mins']:.0f}m")

    st.markdown("---")

    # 3. Daily Disruption Timeline Across January 2022
    st.markdown("### 📈 Daily Flight Disruption Timeline (January 1–31, 2022)")
    daily_trends = compute_daily_trends(filtered_df)

    fig_timeline = go.Figure()
    # Bar for completed flights
    fig_timeline.add_trace(go.Bar(
        x=daily_trends["Day"],
        y=daily_trends["Completed"],
        name="Completed Flights",
        marker_color="#2b5c8f",
        opacity=0.8,
    ))
    # Bar for cancelled flights
    fig_timeline.add_trace(go.Bar(
        x=daily_trends["Day"],
        y=daily_trends["Cancelled"],
        name="Cancelled Flights",
        marker_color="#d9534f",
    ))
    # Line for delay rate
    fig_timeline.add_trace(go.Scatter(
        x=daily_trends["Day"],
        y=daily_trends["Delay_Rate_Pct"],
        name="Delay Rate (%)",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#f0ad4e", width=3),
    ))

    # Add peak annotation if Jan 2 is in view
    if 2 in daily_trends["Day"].values:
        fig_timeline.add_annotation(
            x=2,
            y=daily_trends.loc[daily_trends["Day"] == 2, "Delay_Rate_Pct"].values[0],
            yref="y2",
            text="Jan 2 Peak Surge (43.2% Delayed)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#d9534f",
            bgcolor="white",
            bordercolor="#d9534f",
        )

    fig_timeline.update_layout(
        height=380,
        barmode="stack",
        xaxis=dict(title="Day of Month (January 2022)", tickmode="linear", dtick=2),
        yaxis=dict(title="Flight Volume"),
        yaxis2=dict(title="Delay Rate (%)", overlaying="y", side="right", range=[0, 60]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # 4. Airline Benchmarking & Airport Analytics in 2 Columns
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### 🏢 Airline Performance Leaderboard")
        airline_bench = compute_airline_benchmarks(filtered_df)

        fig_airline = px.bar(
            airline_bench,
            x="Airline",
            y=["On_Time_Rate_Pct", "Delay_Rate_Pct"],
            title="Carrier On-Time vs. Delay Share (%)",
            labels={"value": "Percentage (%)", "variable": "Metric"},
            color_discrete_map={"On_Time_Rate_Pct": "#5cb85c", "Delay_Rate_Pct": "#d9534f"},
            barmode="group",
            height=360,
        )
        fig_airline.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_airline, use_container_width=True)

    with chart_col2:
        st.markdown("### 🛫 Busiest & Most Delayed Origin Hubs")
        busiest_airports, most_delayed_airports = compute_airport_analytics(filtered_df, min_flights=200)

        fig_hubs = px.bar(
            busiest_airports.head(10),
            x="Origin",
            y="Total_Flights",
            color="Delay_Rate_Pct",
            color_continuous_scale="Reds",
            title="Top 10 Busiest Hubs (Color = Delay %)",
            labels={"Total_Flights": "Total Departures", "Delay_Rate_Pct": "Delay %"},
            height=360,
        )
        fig_hubs.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hubs, use_container_width=True)

    # 5. Delay Root Causes & Departure vs Arrival Dynamics
    dyn_col1, dyn_col2 = st.columns(2)

    with dyn_col1:
        st.markdown("### 🧩 Delay Root Causes Breakdown (5 FAA Factors)")
        causes_data = compute_delay_causes_breakdown(filtered_df)
        cause_dict = causes_data["causes"]

        cause_labels = {
            "CarrierDelay": "Carrier (Airline Operations)",
            "LateAircraftDelay": "Late Arriving Aircraft",
            "NASDelay": "National Aviation System (NAS)",
            "WeatherDelay": "Extreme Weather",
            "SecurityDelay": "Security Delays",
        }

        donut_labels = [cause_labels.get(k, k) for k in cause_dict.keys()]
        donut_values = [v["total_minutes"] for v in cause_dict.values()]

        fig_donut = px.pie(
            values=donut_values,
            names=donut_labels,
            hole=0.45,
            title=f"Total Delay Minutes: {causes_data['grand_total_delay_minutes']:,.0f} mins",
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=360,
        )
        fig_donut.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with dyn_col2:
        st.markdown("### 🔄 Departure vs. Arrival Delay Dynamics")
        dynamics = compute_dep_vs_arr_dynamics(filtered_df)

        st.info(
            f"📈 **Correlation**: `{dynamics['correlation']}` (Strongest Predictor)\n\n"
            f"✈️ **In-Flight Recovery Rate**: `{dynamics['recovery_rate_pct']}%` of flights departing "
            f"with minor delays (1–14 mins) recovered time and arrived on time (<= 0 min delay).\n\n"
            f"⚠️ **Delay Compounding**: `{dynamics['compounding_rate_pct']}%` of flights departing "
            f">= 30 mins late remained heavily delayed upon arrival."
        )

        # Sampled scatter plot for responsive rendering
        dep_col = "DepDelay" if "DepDelay" in filtered_df.columns else "Depart Delay (Mins)"
        arr_col = "ArrDelay" if "ArrDelay" in filtered_df.columns else "Arrival Delay (Mins)"
        valid_scatter = filtered_df.dropna(subset=[dep_col, arr_col])
        sample_scatter = valid_scatter.sample(n=min(1500, len(valid_scatter)), random_state=42)

        fig_scatter = px.scatter(
            sample_scatter,
            x=dep_col,
            y=arr_col,
            opacity=0.45,
            title="Dep Delay vs. Arr Delay (Sampled 1,500 flights)",
            labels={dep_col: "Departure Delay (mins)", arr_col: "Arrival Delay (mins)"},
            height=280,
        )

        if len(sample_scatter) > 1:
            x_vals = sample_scatter[dep_col].to_numpy(dtype=float)
            y_vals = sample_scatter[arr_col].to_numpy(dtype=float)
            # Filter out non-finite values if any
            finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            if np.sum(finite_mask) > 1:
                slope, intercept = np.polyfit(x_vals[finite_mask], y_vals[finite_mask], 1)
                x_min, x_max = float(x_vals[finite_mask].min()), float(x_vals[finite_mask].max())
                x_fit = np.linspace(x_min, x_max, 50)
                y_fit = slope * x_fit + intercept
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_fit,
                        mode="lines",
                        name=f"Trendline (Slope: {slope:.2f})",
                        line=dict(color="red", width=2, dash="dash"),
                    )
                )

        fig_scatter.update_layout(
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
