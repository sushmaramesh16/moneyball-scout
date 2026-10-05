"""Moneyball Scout: Streamlit front end.

streamlit run app/streamlit_app.py                 # loads artifacts directly
SCOUT_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py   # via FastAPI
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import sidebar  # noqa: E402

st.set_page_config(page_title="Moneyball Scout", page_icon="⚽", layout="wide")

if "score" not in st.session_state:
    st.session_state.score = "debiased"

pages = [
    st.Page("views/scout.py", title="Scout a player", icon="🔎", default=True),
    st.Page("views/gems.py", title="Undervalued Gems", icon="💎"),
    st.Page("views/backtest.py", title="Backtest", icon="📈"),
    st.Page("views/model.py", title="Model", icon="🧠"),
]
navigation = st.navigation(pages)
sidebar()
navigation.run()
