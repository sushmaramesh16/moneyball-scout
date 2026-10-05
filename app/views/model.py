"""How the model works: test metrics and global SHAP importance."""

import pandas as pd
import streamlit as st
import ui
from common import (
    backend,
    backtest,
)

st.title("How the model works")
tm = backtest()["test_metrics"]
main = tm[tm.model == "lightgbm_tuned"].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("R² on 2024/25 (held-out test)", f"{main.r2:.3f}")
c2.metric(
    "Typical error",
    f"{main.median_pct_error:.0%}",
    help="Median absolute error, as a multiplicative % miss",
)
c3.metric("Median absolute error", ui.eur(main.medae_eur))
st.markdown(
    "A **LightGBM** model predicts each player's log market value relative to the "
    "market's overall level that season, from data available at the end of the season "
    "only: age, position, minutes and output (this season, last season and the last "
    "three), club strength and spending, and league price level. It never sees the "
    "player's own past valuations or transfer fees. Trained on 2015/16–2024/25; tuned "
    "on 2023/24; tested once on 2024/25."
)
st.subheader("What matters most, across all players")
st.plotly_chart(
    ui.importance_chart(backend().global_importance(15)),
    width="stretch",
    config={"displayModeBar": False},
    theme="streamlit",
)
st.subheader("Held-out test, 2024/25")
names = {
    "lightgbm_tuned": "LightGBM (this app)",
    "linear": "Linear baseline",
    "ceiling_lightgbm": "With past valuations (reference only)",
    "last_value_baseline": "Last valuation carried forward (reference only)",
}
st.dataframe(
    pd.DataFrame(
        {
            "Model": tm.model.map(names),
            "RMSE (log)": tm.rmse.round(3),
            "R²": tm.r2.round(3),
            "Median abs. error": tm.medae_eur.map(ui.eur),
            "Typical miss": tm.median_pct_error.map(lambda v: f"{v:.0%}"),
        }
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "The reference models use the player's own past valuations, so they mostly "
    "echo the market; the gap to them is where 'undervalued' signals live."
)
