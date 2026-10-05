"""Shared state for the Streamlit pages: cached backend calls and the sidebar."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.client import get_backend

SCORE_LABELS = {
    "debiased": "De-biased (vs league & age peers)",
    "raw": "Raw ((predicted − actual) / actual)",
}
VALUE_FLOOR_OPTIONS = [0, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
STAT_LABELS = {
    "apps": "Appearances",
    "starts": "Starts",
    "minutes": "Minutes",
    "goals": "Goals",
    "assists": "Assists",
    "ga_per90": "Goals + assists / 90",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "league_minutes_share": "Share of club's league minutes",
    "euro_minutes": "European minutes",
    "captain_games": "Games as captain",
    "prev_minutes": "Minutes last season",
    "club_position": "Club league position",
    "club_ppg": "Club points per game",
    "club_in_europe": "Club in Europe",
}


@st.cache_resource
def backend():
    return get_backend()


@st.cache_data
def filter_options():
    return backend().filter_options()


@st.cache_data
def player_index() -> pd.DataFrame:
    df = pd.DataFrame(backend().all_players())
    return df.sort_values("target_value", ascending=False).reset_index(drop=True)


@st.cache_data
def backtest() -> dict:
    return {k: pd.DataFrame(v) for k, v in backend().backtest_summary().items()}


def score_key() -> str:
    """Field holding the score chosen in the sidebar (API field names)."""
    return "undervalued_score" if st.session_state.score == "debiased" else "undervalued_score_raw"


def sidebar():
    st.sidebar.radio(
        "Undervalued score",
        list(SCORE_LABELS),
        key="score",
        format_func=lambda k: SCORE_LABELS[k],
        help="De-biased compares each player with peers in the same league and "
        "age band; it predicted real value rises better in the backtest.",
    )
    opts = filter_options()
    st.sidebar.caption(
        f"Season {opts['season_label']} · {len(player_index()):,} players in 14 European "
        f"leagues · data: Transfermarkt via Kaggle (CC0) · backend: {backend().name}"
    )
