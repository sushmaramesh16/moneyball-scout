"""Undervalued Gems leaderboard."""

import pandas as pd
import streamlit as st
import ui
from common import (
    VALUE_FLOOR_OPTIONS,
    backend,
    filter_options,
    score_key,
)

opts = filter_options()
st.title("Undervalued Gems")
st.caption(
    "Players the model values most above their current Transfermarkt value. "
    "Ranked by the score chosen in the sidebar."
)

f1, f2, f3, f4 = st.columns([1, 1.4, 1.4, 1.2])
group = f1.selectbox("Position", ["All", *opts["position_groups"]])
leagues = {code: name for code, name in opts["leagues"]}
league = f2.selectbox(
    "League",
    ["All", *leagues],
    format_func=lambda c: "All" if c == "All" else leagues[c],
)
lo, hi = opts["age_range"]
ages = f3.slider("Age", lo, hi, (lo, hi))
floor = f4.select_slider(
    "Minimum actual value",
    VALUE_FLOOR_OPTIONS,
    value=opts["default_min_value"],
    format_func=ui.eur,
    help="Default €500K: ratio scores explode for near-zero valuations. Lower it to see everyone.",
)

rows = backend().undervalued(
    position_group=None if group == "All" else group,
    league=None if league == "All" else league,
    min_age=ages[0],
    max_age=ages[1],
    min_value=floor,
    score=st.session_state.score,
    limit=100,
)
if not rows:
    st.info("No players match these filters.")
    st.stop()
key = score_key()
df = pd.DataFrame(rows)
table = pd.DataFrame(
    {
        "Rank": df["rank"],
        "Player": df["name"],
        "Club": df["club_name"],
        "League": df["league"],
        "Position": df["position"],
        "Age": df["age"].round(1),
        "Actual (€M)": df["actual_value"] / 1e6,
        "Predicted (€M)": df["predicted_value"] / 1e6,
        "Score": df[key] * 100,
    }
)
event = st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Actual (€M)": st.column_config.NumberColumn(format="%.2f"),
        "Predicted (€M)": st.column_config.NumberColumn(format="%.2f"),
        "Score": st.column_config.NumberColumn(
            "Undervalued ("
            + ("de-biased" if st.session_state.score == "debiased" else "raw")
            + ")",
            format="%+.0f%%",
        ),
    },
)
st.caption("Select a row to open the player.")
if event.selection.rows:
    selected = int(df.loc[event.selection.rows[0], "player_id"])
    st.switch_page("views/scout.py", query_params={"player": str(selected)})
