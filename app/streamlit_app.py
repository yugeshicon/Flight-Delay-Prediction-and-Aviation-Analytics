"""Main Streamlit application entrypoint for Flight Delay Prediction & Aviation Analytics.

Combines the Flight Delay Predictor, Aviation Analytics Dashboard,
and Model Intelligence into an integrated, multi-tab interface.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root and app directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

for _p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from app.components_analytics import render_analytics_page
    from app.components_predict import render_prediction_page
except (ImportError, ModuleNotFoundError):
    from components_analytics import render_analytics_page
    from components_predict import render_prediction_page

from src.data_loader import load_flight_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Flight Delay Prediction & Aviation Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading historical aviation records...")
def get_cached_flight_data() -> pd.DataFrame:
    """Load and cache flight data for dashboard analytics."""
    return load_flight_data(dataset_type="selected")


def render_model_insights_page():
    """Render Model Performance, Comparison Table, and Feature Importances."""
    st.markdown("## 🤖 Model Performance & Machine Learning Intelligence")
    st.markdown(
        "Empirical benchmarks, candidate model comparison, and feature importance analysis "
        "trained on US Bureau of Transportation Statistics flight data."
    )

    meta_file = MODELS_DIR / "model_metadata.json"
    if not meta_file.exists():
        st.warning("⚠️ Model metadata file not found. Please run `python -m src.train` first.")
        return

    with open(meta_file, "r") as f:
        metadata = json.load(f)

    # 1. Champion Model Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Champion Classifier", metadata.get("champion_model", "LightGBM"))
    with col2:
        champ_metrics = next((m for m in metadata.get("comparison_metrics", []) if m["model_name"] == metadata.get("champion_model")), {})
        st.metric("Champion ROC-AUC", f"{champ_metrics.get('roc_auc', 0.0):.4f}")
    with col3:
        st.metric("Test Accuracy", f"{champ_metrics.get('accuracy', 0.0) * 100:.1f}%")
    with col4:
        st.metric("Delay Regressor MAE", f"~{metadata.get('regression_metrics', {}).get('mae_mins', 0.0):.1f} mins")

    st.markdown("---")

    # 2. Comprehensive Model Comparison Table
    st.markdown("### 📊 Candidate Model Comparison Table")
    metrics_list = metadata.get("comparison_metrics", [])
    if metrics_list:
        comp_df = pd.DataFrame([
            {
                "Candidate Model": m["model_name"],
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision (Delayed)": f"{m['precision'] * 100:.2f}%",
                "Recall (Delayed)": f"{m['recall'] * 100:.2f}%",
                "F1-Score": f"{m['f1_score']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "PR-AUC": f"{m['pr_auc']:.4f}",
                "True Negatives (TN)": f"{m['confusion_matrix']['true_negatives']:,}",
                "False Positives (FP)": f"{m['confusion_matrix']['false_positives']:,}",
                "False Negatives (FN)": f"{m['confusion_matrix']['false_negatives']:,}",
                "True Positives (TP)": f"{m['confusion_matrix']['true_positives']:,}",
            }
            for m in metrics_list
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Plot comparison
        fig_comp = px.bar(
            pd.DataFrame(metrics_list),
            x="model_name",
            y=["roc_auc", "f1_score", "accuracy"],
            barmode="group",
            title="Candidate Models Metric Comparison",
            labels={"value": "Score", "variable": "Metric", "model_name": "Model"},
            color_discrete_sequence=px.colors.qualitative.Safe,
            height=340,
        )
        fig_comp.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    # 3. Top Feature Importances
    st.markdown("### 🏆 Top Predictive Feature Importances")
    top_features = metadata.get("top_features", [])
    if top_features:
        feat_df = pd.DataFrame(top_features).sort_values("importance", ascending=True)
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Feature Importances (Champion Model)",
            labels={"importance": "Relative Importance (Splits/Gain)", "feature": "Pre-Flight Feature"},
            color="importance",
            color_continuous_scale="Blues",
            height=420,
        )
        fig_feat.update_layout(margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_feat, use_container_width=True)

    # 4. Methodology & Data Hygiene Documentation
    st.markdown("### 📖 Methodology & Data Hygiene Principles")
    st.markdown(
        r"""
        - **Target Formulation**: Follows official FAA/DOT standard: binary target `Delayed` $\iff \text{Arrival Delay} \ge 15$ minutes (`ArrDel15 == 1`).
        - **Zero Post-Departure Leakage**: Excludes all post-gate attributes (`DepDelay`, `ArrDelay`, `Taxi-out`, `Wheels Off/On`, `AirTime`, cause breakdowns).
        - **Clean Operational Filtering**: Cancelled ($35,667$) and diverted ($1,176$) flights are cleanly isolated so only completed flights are modeled for delay duration.
        - **Dual-Model Inference Architecture**:
          1. **Classification Stage**: Calibrated probability of delay $P(\text{Delayed})$ via LightGBM.
          2. **Duration Regression Stage**: Predicts expected delay duration in minutes when $P(\text{Delayed}) \ge \text{threshold}$.
        """
    )


def main():
    st.title("✈️ Flight Delay Prediction & Aviation Analytics Platform")
    st.caption("US Bureau of Transportation Statistics (BTS) • Comprehensive Operational Intelligence")

    tab1, tab2, tab3 = st.tabs([
        "🔮 Flight Delay Predictor",
        "📊 Aviation Analytics Dashboard",
        "🤖 Model Intelligence & Benchmarks",
    ])

    with tab1:
        render_prediction_page()

    with tab2:
        df_flights = get_cached_flight_data()
        render_analytics_page(df_flights)

    with tab3:
        render_model_insights_page()


if __name__ == "__main__":
    main()
