"""Design system, custom CSS, animations, and UI components for the modern interface.

Provides an Aviation Glassmorphism theme with subtle keyframe animations,
sleek card containers, custom KPI chips, and unified Plotly styling.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


# Primary Modern Dark Aviation Color Palette
THEME = {
    "bg_dark": "#0B0F19",
    "surface": "rgba(17, 24, 39, 0.75)",
    "surface_hover": "rgba(30, 41, 59, 0.85)",
    "border": "rgba(255, 255, 255, 0.08)",
    "border_glow": "rgba(56, 189, 248, 0.35)",
    "cyan": "#38BDF8",
    "indigo": "#818CF8",
    "green": "#10B981",
    "amber": "#F59E0B",
    "crimson": "#EF4444",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
}


def inject_custom_css():
    """Inject modern glassmorphism CSS, typography, and animation keyframes."""
    css = f"""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Resets */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {THEME['text_primary']};
        letter-spacing: -0.01em;
    }}

    code, pre, .mono {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Keyframe Animations */
    @keyframes fadeInUp {{
        0% {{
            opacity: 0;
            transform: translateY(14px);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes pulseGlowRed {{
        0% {{
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
        }}
        70% {{
            box-shadow: 0 0 0 10px rgba(239, 68, 68, 0);
        }}
        100% {{
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
        }}
    }}

    @keyframes pulseGlowGreen {{
        0% {{
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
        }}
        70% {{
            box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
        }}
        100% {{
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }}
    }}

    @keyframes planeDrift {{
        0% {{ transform: translateX(-4px); }}
        50% {{ transform: translateX(4px); }}
        100% {{ transform: translateX(-4px); }}
    }}

    /* Global Glass Cards */
    .glass-card {{
        background: {THEME['surface']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {THEME['border']};
        border-radius: 16px;
        padding: 20px 24px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeInUp 0.45s ease-out;
        margin-bottom: 16px;
    }}

    .glass-card:hover {{
        border-color: rgba(255, 255, 255, 0.16);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.35);
    }}

    /* Executive KPI Cards */
    .kpi-container {{
        background: {THEME['surface']};
        backdrop-filter: blur(16px);
        border: 1px solid {THEME['border']};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        animation: fadeInUp 0.4s ease-out;
    }}
    .kpi-container:hover {{
        transform: translateY(-3px);
        border-color: {THEME['border_glow']};
    }}
    .kpi-title {{
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {THEME['text_secondary']};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {THEME['text_primary']};
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        color: {THEME['text_muted']};
        margin-top: 4px;
    }}

    /* Flight Boarding Pass / Route Card */
    .route-card {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 22px 28px;
        margin: 14px 0 24px 0;
        animation: fadeInUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }}
    .route-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {THEME['cyan']}, {THEME['indigo']});
    }}
    .route-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }}
    .route-tag {{
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-radius: 8px;
        background: rgba(56, 189, 248, 0.15);
        color: {THEME['cyan']};
        border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .route-trajectory {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 10px 0 16px 0;
    }}
    .airport-node {{
        text-align: center;
    }}
    .airport-code {{
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {THEME['text_primary']};
    }}
    .airport-sub {{
        font-size: 0.8rem;
        color: {THEME['text_secondary']};
    }}
    .flight-path {{
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0 20px;
    }}
    .path-line {{
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.2), {THEME['cyan']}, rgba(129, 140, 248, 0.2));
        position: relative;
        margin: 8px 0;
    }}
    .plane-icon {{
        font-size: 1.4rem;
        animation: planeDrift 3s ease-in-out infinite;
    }}

    /* Prediction Result Cards */
    .result-banner {{
        border-radius: 16px;
        padding: 24px;
        margin-top: 14px;
        animation: fadeInUp 0.4s ease-out;
    }}
    .result-banner.delayed {{
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        animation: pulseGlowRed 2.5s infinite;
    }}
    .result-banner.ontime {{
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        animation: pulseGlowGreen 2.5s infinite;
    }}
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-delayed {{
        background: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }}
    .badge-ontime {{
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}

    /* Risk Driver Chip */
    .driver-chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        font-size: 0.82rem;
        color: {THEME['text_primary']};
        margin: 4px 6px 4px 0;
        transition: background 0.2s ease;
    }}
    .driver-chip:hover {{
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(56, 189, 248, 0.3);
    }}

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {{
        background-color: #070B14;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def apply_modern_chart_theme(fig: go.Figure, height: Optional[int] = None) -> go.Figure:
    """Apply unified aviation glassmorphism theme to Plotly figures."""
    layout_kwargs: Dict[str, Any] = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            color=THEME["text_primary"],
            size=12,
        ),
        margin=dict(l=20, r=20, t=40, b=25),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            bordercolor="rgba(255, 255, 255, 0.15)",
        ),
    )
    if height:
        layout_kwargs["height"] = height

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255, 255, 255, 0.05)",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255, 255, 255, 0.05)",
        zeroline=False,
    )
    return fig


def render_glass_kpi(title: str, value: str, sub: str, status_color: str = "#38BDF8"):
    """Render a modern glassmorphic KPI card with accent bar."""
    html = f"""
    <div class="kpi-container" style="border-top: 3px solid {status_color};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {status_color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_boarding_pass(
    airline: str,
    airline_name: str,
    origin: str,
    origin_name: str,
    dest: str,
    dest_name: str,
    dep_time_str: str,
    distance_miles: int,
    duration_mins: int,
):
    """Render an animated aviation route badge / digital boarding pass."""
    html = f"""
    <div class="route-card">
        <div class="route-header">
            <div>
                <span class="route-tag">✈️ Flight Plan Preview</span>
                <span style="margin-left: 10px; font-weight: 600; font-size: 0.9rem; color: {THEME['text_primary']};">
                    {airline} • {airline_name}
                </span>
            </div>
            <div style="font-size: 0.8rem; color: {THEME['text_secondary']};">
                Scheduled Dep: <strong style="color: {THEME['cyan']};">{dep_time_str}</strong>
            </div>
        </div>
        <div class="route-trajectory">
            <div class="airport-node">
                <div class="airport-code">{origin}</div>
                <div class="airport-sub">{origin_name}</div>
            </div>
            <div class="flight-path">
                <div style="font-size: 0.75rem; color: {THEME['text_secondary']}; font-weight: 500;">
                    {distance_miles:,} miles • ~{duration_mins} mins
                </div>
                <div class="path-line"></div>
                <div class="plane-icon">✈️</div>
            </div>
            <div class="airport-node">
                <div class="airport-code">{dest}</div>
                <div class="airport-sub">{dest_name}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
