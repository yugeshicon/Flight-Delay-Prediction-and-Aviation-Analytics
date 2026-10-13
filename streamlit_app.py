"""Root launcher for Streamlit application.

Allows running the application via:
    streamlit run streamlit_app.py
or
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"

for _p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

# Configure wide layout in the root entrypoint script
# Ensures that browser refreshes and reruns preserve the wide layout
try:
    st.set_page_config(
        page_title="Flight Delay Prediction & Aviation Analytics",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

from app.streamlit_app import main

if __name__ == "__main__":
    main()

