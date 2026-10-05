"""Backtest: headline 2024/25 (no floor) with CIs, baselines, missingness, value floor."""

import pandas as pd
import streamlit as st
import ui
from common import (
    backtest,
)

STRATEGY_LABELS = {
    ("model_debiased", "top_decile"): "Model, de-biased · top 10%",
    ("model_debiased", "top_100"): "Model, de-biased · top 100",
    ("model_raw", "top_decile"): "Model, raw · top 10%",
    ("model_raw", "top_100"): "Model, raw · top 100",
    ("linear_raw", "top_decile"): "Linear model · top 10%",
    ("young_regulars", "top_decile"): "Young regulars (U23 by minutes) · top 10%",
    ("mean_reversion", "top_decile"): "Mean reversion (biggest recent drop) · top 10%",
    ("all", "all"): "All players (base rate)",
}


def _ci(lo, hi, fmt="{:.2f}"):
    return "" if pd.isna(lo) else f" [{fmt.format(lo)}–{fmt.format(hi)}]"


def _ci(lo, hi, fmt="{:.2f}"):
    return "" if pd.isna(lo) else f" [{fmt.format(lo)}–{fmt.format(hi)}]"


bt = backtest()
res, boot = bt["results"], bt["bootstrap"]
head = res[res.season == 2024].set_index(["strategy", "k"])
b = boot.set_index(["strategy", "k_label"])
base = head.loc[("all", "all")]

st.title("Backtest: do flagged players' values actually rise?")
st.markdown(
    "At the end of each season we rank every player by the undervalued score and check "
    "their Transfermarkt value **about 12 months later**. Models only ever see earlier "
    "seasons. **Headline: 2024/25, out-of-sample, pre-planned, no value floor.** "
    "Brackets are 95% intervals (bootstrap over players, re-selecting the top group "
    "each time)."
)

d = head.loc[("model_debiased", "top_decile")]
db = b.loc[("model_debiased", "top_decile")]
ic = bt["rank_ic"].query("season == 2024 and outcome == 'outcome_log_change'").set_index("strategy")
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Hit rate: value rose (de-biased top 10%)",
    ui.pct(d.hit_raw, signed=False),
    help=f"Base rate for all players: {ui.pct(base.hit_raw, signed=False)}",
)
m2.metric(
    "Lift vs all players",
    f"{db.lift_raw:.2f}×",
    help=f"95% CI {db.lift_raw_ci_low:.2f}–{db.lift_raw_ci_high:.2f}",
)
m3.metric(
    "Matched lift (age × value)",
    f"{db.matched_lift_raw:.2f}×",
    help=f"95% CI {db.matched_lift_raw_ci_low:.2f}–{db.matched_lift_raw_ci_high:.2f}",
)
m4.metric(
    "Rank correlation with 12-month change",
    f"{ic.loc['model_debiased', 'ic']:.3f}",
    help=f"Raw score: {ic.loc['model_raw', 'ic']:.3f}",
)

rows, chart = [], []
for (strategy, k), label in STRATEGY_LABELS.items():
    r = head.loc[(strategy, k)]
    bb = b.loc[(strategy, k)] if (strategy, k) in b.index else None
    hit_ci = (
        _ci(bb.hit_raw_ci_low, bb.hit_raw_ci_high, "{:.0%}")
        if bb is not None
        else _ci(r.hit_raw_ci_low, r.hit_raw_ci_high, "{:.0%}")
    )
    rows.append(
        {
            "Strategy": label,
            "Players": int(r.n_flagged),
            "Hit rate [95% CI]": f"{r.hit_raw:.1%}{hit_ci}",
            "Lift [95% CI]": f"{r.lift_raw:.2f}"
            + (_ci(bb.lift_raw_ci_low, bb.lift_raw_ci_high) if bb is not None else ""),
            "Matched lift [95% CI]": f"{r.matched_lift_raw:.2f}"
            + (
                _ci(bb.matched_lift_raw_ci_low, bb.matched_lift_raw_ci_high)
                if bb is not None
                else ""
            ),
            "Mean change vs market": f"{r.mean_rel_change:+.3f}",
            "No later valuation": f"{r.missing_rate:.0%}",
        }
    )
    if bb is not None:
        chart.append(
            {
                "label": label,
                "value": bb.matched_lift_raw,
                "low": bb.matched_lift_raw_ci_low,
                "high": bb.matched_lift_raw_ci_high,
            }
        )
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption(
    "Hit = Transfermarkt value higher ~12 months later. Lift = hit rate ÷ base "
    "rate. Mean change vs market = log change minus the market index's change. "
    "Linear-model and base-rate intervals are Wilson intervals for the hit rate."
)

st.subheader("Matched lift: the honest comparison")
st.plotly_chart(
    ui.lift_chart(chart), width="stretch", config={"displayModeBar": False}, theme="streamlit"
)
st.markdown(
    "Young and cheap players rise more often whatever any model says (the age curve, "
    "and small values moving up more easily). So for each flagged group we compute the "
    "hit rate we would **expect from its age × value-band mix alone**, using every "
    "player's outcome in the same season. *Matched lift* = actual hit rate ÷ that "
    "expectation; 1.0 means the score adds nothing beyond age and price."
)

yr = head.loc[("young_regulars", "top_decile")]
mr = head.loc[("mean_reversion", "top_decile")]
st.subheader("Simple rules for comparison")
st.markdown(
    f"- **Young regulars** (under-23s with the most minutes) rise most often "
    f"({yr.hit_raw:.0%}), but almost all of that is their age and price: matched lift "
    f"{yr.matched_lift_raw:.2f} vs {d.matched_lift_raw:.2f} for the de-biased model, and "
    f"a mean change vs market of {yr.mean_rel_change:+.3f} vs {d.mean_rel_change:+.3f}.\n"
    f"- **Mean reversion** (players whose value just dropped) does the opposite of a "
    f"bounce-back: only {mr.hit_raw:.0%} rise (matched lift {mr.matched_lift_raw:.2f}). "
    f"Recent drops tend to continue."
)

st.subheader("Missing outcomes")
miss = bt["missingness"].query("season == 2024").set_index("strategy")
mm = miss.loc["model_debiased"]
st.markdown(
    f"Some players have no valuation ~12 months later (retired, left the covered leagues, "
    f"or simply not re-valued). Flagged players are missing more often: "
    f"**{mm.missing_flagged:.1%}** vs {mm.missing_others:.1%} for everyone else "
    f"(p = {mm.p_value:.4f}); they skew cheap, and cheap players are re-valued less "
    f"often. Robustness check, counting every missing outcome as *not risen*:"
)
robust = []
for (strategy, k), label in STRATEGY_LABELS.items():
    if strategy == "all":
        continue
    r = head.loc[(strategy, k)]
    robust.append(
        {
            "Strategy": label,
            "Missing (flagged)": f"{r.missing_rate:.0%}",
            "Hit rate, missing = not risen": f"{r.hit_raw_missing_as_fail:.1%}",
            "Matched lift, missing = not risen": f"{r.matched_lift_raw_missing_as_fail:.2f}",
        }
    )
st.dataframe(pd.DataFrame(robust), hide_index=True, width="stretch")

st.subheader("Value floor (€500K) used by the leaderboard")
fl = bt["floor"]
st.markdown(
    "The app hides players worth under €500K by default, because ratio scores explode "
    "for near-zero valuations. The floor was chosen after looking at 2024/25, so it is "
    "**validated on 2023/24**; the 2024/25 floor result is **exploratory/confirmatory "
    "only** and is not the headline."
)
COLUMNS = {
    (2023, False): "2023/24 · no floor",
    (2023, True): "2023/24 · €500K floor (validation)",
    (2024, False): "2024/25 · no floor (headline)",
    (2024, True): "2024/25 · €500K floor (exploratory)",
}
pivot = {}
for r in fl[fl.strategy.isin(["model_debiased", "model_raw"])].itertuples():
    col = COLUMNS[(r.season, r.floor_eur > 0)]
    cell = f"{r.hit_raw:.1%} · {r.matched_lift_raw:.2f}× · ρ {r.ic:.3f}"
    pivot.setdefault(STRATEGY_LABELS[(r.strategy, r.k)], {})[col] = cell
st.dataframe(
    pd.DataFrame.from_dict(pivot, orient="index")[list(COLUMNS.values())]
    .rename_axis("Strategy")
    .reset_index(),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Each cell: hit rate · matched lift · rank correlation (ρ) with the 12-month change. "
    "With the floor, the top 10%, base rate and matched baseline are recomputed among "
    "players worth ≥ €500K."
)

with st.expander("2023/24 (optimistic: hyperparameters were tuned on this season)"):
    old = res[res.season == 2023].set_index(["strategy", "k"])
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Strategy": label,
                    "Hit rate": f"{old.loc[(s, k)].hit_raw:.1%}",
                    "Lift": f"{old.loc[(s, k)].lift_raw:.2f}",
                    "Matched lift": f"{old.loc[(s, k)].matched_lift_raw:.2f}",
                }
                for (s, k), label in STRATEGY_LABELS.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        bt["rank_ic"]
        .query("outcome == 'outcome_log_change'")[
            ["season", "strategy", "n", "ic", "ci_low", "ci_high"]
        ]
        .round(3),
        hide_index=True,
        width="stretch",
    )
