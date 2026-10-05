"""Scout a player: predicted vs actual value, SHAP drivers, season stats."""

import pandas as pd
import streamlit as st
import ui
from common import (
    STAT_LABELS,
    backend,
    filter_options,
    player_index,
    score_key,
)

idx = player_index()
ids = idx["player_id"].tolist()
labels = {r.player_id: f"{r.name} · {r.club_name} ({r.league})" for r in idx.itertuples()}
requested = st.query_params.get("player")
default = int(requested) if requested and int(requested) in labels else None
if default is None:
    top = backend().undervalued(min_value=filter_options()["default_min_value"], limit=1)
    default = top[0]["player_id"] if top else ids[0]

st.title("Scout a player")
pid = st.selectbox(
    "Search by player or club", ids, index=ids.index(default), format_func=labels.get
)
st.query_params["player"] = str(pid)
p, ex = backend().player(pid), backend().explain(pid)

st.header(p["name"])
st.caption(
    " · ".join(
        str(x)
        for x in [
            p["position"],
            f"{p['age']:.0f} years",
            p["club_name"],
            p["league_name"],
            p["season_label"],
        ]
        if x
    )
)
key = score_key()
other = "undervalued_score_raw" if key == "undervalued_score" else "undervalued_score"
c1, c2, c3 = st.columns(3)
c1.metric(
    "Actual value (Transfermarkt)",
    ui.eur(p["actual_value"]),
    help=f"Valuation dated {p['valuation_date']}",
)
c2.metric("Predicted value", ui.eur(p["predicted_value"]))
names = {"undervalued_score": "de-biased", "undervalued_score_raw": "raw"}
c3.metric(
    f"Undervalued score ({names[key]})",
    ui.pct(p[key]),
    help="Positive = the model values the player above the market. Raw = (predicted − "
    "actual) / actual; de-biased = the same gap relative to league & age-band peers.",
)
c3.caption(f"{names[other].capitalize()} score: {ui.pct(p[other])}")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("What drives the predicted value")
    st.plotly_chart(
        ui.shap_chart(ex["contributions"]),
        width="stretch",
        config={"displayModeBar": False},
        theme="streamlit",
    )
    st.caption(
        "SHAP values: each bar is how much one feature moves this player's "
        "predicted value away from the average player's. Blue pushes it up, red "
        "down; labels are the multiplicative effect."
    )
with right:
    st.subheader("Season stats")
    stats = pd.DataFrame(
        [(STAT_LABELS.get(k, k), v) for k, v in p["stats"].items() if v is not None],
        columns=["Stat", "Value"],
    )
    stats["Value"] = stats["Value"].map(lambda v: f"{v:,.2f}" if v % 1 else f"{v:,.0f}")
    st.dataframe(stats, hide_index=True, width="stretch")
    if p["contract_expiration_date"]:
        st.caption(
            f"Contract until {p['contract_expiration_date']} (shown for context; "
            "not used by the model because its history is unknown)."
        )

if backend().reports_enabled():
    st.subheader("Scouting report")
    st.info("Scouting reports are generated on request.")
