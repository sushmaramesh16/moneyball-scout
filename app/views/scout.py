"""Scout a player: predicted vs actual value, SHAP drivers, season stats."""

import html

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

from src.explain import report as report_module
from src.explain.shap_explain import display_value

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

subtitle = " · ".join(
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
# Initials avatar (player photos are not covered by the dataset's CC0 licence).
st.markdown(
    f"""<div style="display:flex;align-items:center;gap:16px;margin:8px 0 4px">
      <div aria-hidden="true" style="width:56px;height:56px;border-radius:50%;
           background:{ui.colors()["accent"]};color:#fff;display:flex;align-items:center;
           justify-content:center;font-weight:600;font-size:20px;flex:none">
        {html.escape(ui.initials(p["name"]))}</div>
      <div><div style="font-size:1.75rem;font-weight:700;line-height:1.2">
        {html.escape(p["name"])}</div>
        <div style="opacity:.7;font-size:.9rem">{html.escape(subtitle)}</div></div>
    </div>""",
    unsafe_allow_html=True,
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
rng = p.get("predicted_range")
if rng:
    lo50, hi50 = rng["middle_50"]
    lo80, hi80 = rng["middle_80"]
    c2.caption(
        f"Likely range {ui.eur(lo50)}–{ui.eur(hi50)} (middle 50%) · "
        f"{ui.eur(lo80)}–{ui.eur(hi80)} (middle 80%)",
        help="How far actual values fell from the model's predictions for players in "
        f"the same predicted-value band ({rng['band']}) on the held-out 2024/25 season.",
    )
names = {"undervalued_score": "de-biased", "undervalued_score_raw": "raw"}
c3.metric(
    f"Undervalued score ({names[key]})",
    ui.pct(p[key]),
    help="Positive = the model values the player above the market. Raw = (predicted − "
    "actual) / actual; de-biased = the same gap relative to league & age-band peers.",
)
c3.caption(f"{names[other].capitalize()} score: {ui.pct(p[other])}")

d = ex["drivers"]
if d["driver"] == "context":
    st.info(
        f"**Why this player is flagged:** most of the model's lift comes from context "
        f"(age, club and league: {d['context_effect_pct']:+.0f}% together) rather than "
        f"on-pitch performance ({d['performance_effect_pct']:+.0f}%). The model is saying "
        "players with this profile are usually valued higher, not that this season's "
        "output stands out.",
        icon=":material/info:",
    )
elif d["driver"] == "performance":
    st.caption(
        f"Driven mainly by on-pitch performance ({d['performance_effect_pct']:+.0f}%) "
        f"rather than context ({d['context_effect_pct']:+.0f}%)."
    )
else:
    st.caption(
        f"Context ({d['context_effect_pct']:+.0f}%) and performance "
        f"({d['performance_effect_pct']:+.0f}%) both shape this valuation."
    )

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
    stats["Value"] = [display_value(k, v) for k, v in p["stats"].items() if v is not None]
    st.dataframe(stats, hide_index=True, width="stretch")
    if p["contract_expiration_date"]:
        st.caption(
            f"Contract until {p['contract_expiration_date']} (shown for context; "
            "not used by the model because its history is unknown)."
        )

if backend().reports_enabled():
    st.subheader("Scouting report")
    st.caption(
        ":material/smart_toy: **AI-generated.** Written by a language model from the "
        "numbers on this page only; it can be wrong."
    )
    state_key = f"report_{pid}"
    if state_key not in st.session_state and st.button("Generate scouting report"):
        with st.spinner("Writing report…"):
            try:
                st.session_state[state_key] = backend().report(pid)
                st.rerun()
            except report_module.ReportRateLimited as exc:
                st.warning(
                    "The AI report service is busy (free-tier rate limit). Please try "
                    f"again in about {exc.retry_after:.0f} seconds.",
                    icon=":material/hourglass_top:",
                )
            except (report_module.ReportError, report_module.ReportUnavailable):
                st.error("The scouting report could not be generated right now.")
    if state_key in st.session_state:
        rep = st.session_state[state_key]
        with st.container(border=True):
            st.markdown(rep["report"])
            st.caption(f"Model: {rep['model']} · {rep['disclaimer']}")
