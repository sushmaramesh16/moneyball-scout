"""Shared Streamlit helpers: formatting, theme-aware colours and the three charts."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Validated with the dataviz palette checker (CVD + contrast) in both modes.
PALETTE = {
    "light": {
        "up": "#2a78d6",
        "down": "#e34948",
        "accent": "#2a78d6",
        "ink": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
    },
    "dark": {
        "up": "#3987e5",
        "down": "#e66767",
        "accent": "#3987e5",
        "ink": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
    },
}
BAR_PX = 22  # bar thickness cap (<= 24px)
ROW_PX = 34


def colors() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except AttributeError:
        mode = "light"
    return PALETTE["dark" if mode == "dark" else "light"]


def eur(value: float | None) -> str:
    if value is None:
        return "–"
    for unit, div in (("bn", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= div:
            v = value / div
            return f"€{v:.1f}{unit}" if v < 100 else f"€{v:.0f}{unit}"
    return f"€{value:,.0f}"


def pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "–"
    return f"{value * 100:+.0f}%" if signed else f"{value * 100:.1f}%"


def _layout(fig: go.Figure, height: int, x_title: str) -> go.Figure:
    c = colors()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=56, t=8, b=40),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=max(0.0, 1 - BAR_PX / ROW_PX),
        hoverlabel=dict(align="left"),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=c["ink"]),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=c["muted"], size=12)),
        gridcolor=c["grid"],
        gridwidth=1,
        zeroline=True,
        zerolinecolor=c["baseline"],
        zerolinewidth=1,
        tickfont=dict(color=c["muted"]),
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=c["ink"]))
    return fig


def _pad_x(fig: go.Figure, xs: list[float], pad: float = 0.22) -> go.Figure:
    """Leave room inside the plot for 'outside' labels on both negative and positive
    bars, so they never run into the category labels or get clipped."""
    lo, hi = min(0.0, *xs), max(0.0, *xs)
    span = (hi - lo) or 1.0
    fig.update_xaxes(range=[lo - pad * span if lo < 0 else 0, hi + pad * span])
    return fig


def _feature_text(item: dict) -> str:
    v = item["value"]
    if v is None:
        shown = "unknown"
    elif isinstance(v, float):
        shown = f"{v:,.2f}" if abs(v) < 10 else f"{v:,.0f}"
    else:
        shown = str(v)
    return f"{item['label']} = {shown}"


def shap_chart(contributions: list[dict], top: int = 10) -> go.Figure:
    """Diverging bars: what pushes this player's predicted value up (blue) or down (red).
    Remaining features are folded into one 'All other features' bar."""
    c = colors()
    head, rest = contributions[:top], contributions[top:]
    items = [{**h, "text": _feature_text(h)} for h in head]
    if rest:
        total = sum(r["shap"] for r in rest)
        items.append(
            {
                "text": f"All other features ({len(rest)})",
                "shap": total,
                "effect_pct": (2.718281828**total - 1) * 100,
            }
        )
    fig = go.Figure(
        go.Bar(
            x=[i["shap"] for i in items],
            y=[i["text"] for i in items],
            orientation="h",
            marker=dict(
                color=[c["up"] if i["shap"] >= 0 else c["down"] for i in items], cornerradius=4
            ),
            text=[f"{i['effect_pct']:+.0f}%" for i in items],
            textposition="outside",
            textfont=dict(color=c["ink"]),
            cliponaxis=False,
            hovertemplate="%{y}<br>Effect on predicted value: %{text}"
            "<br>SHAP (log scale): %{x:+.3f}<extra></extra>",
        )
    )
    fig = _layout(
        fig,
        ROW_PX * len(items) + 60,
        "Contribution to predicted value (log scale; label = % effect)",
    )
    return _pad_x(fig, [i["shap"] for i in items])


def importance_chart(rows: list[dict]) -> go.Figure:
    """Single series: mean |SHAP| per feature across all live players."""
    c = colors()
    fig = go.Figure(
        go.Bar(
            x=[r["mean_abs_shap"] for r in rows],
            y=[r["label"] for r in rows],
            orientation="h",
            marker=dict(color=c["accent"], cornerradius=4),
            text=[f"{r['share'] * 100:.0f}%" for r in rows],
            textposition="outside",
            textfont=dict(color=c["ink"]),
            cliponaxis=False,
            hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f}<br>Share of total: %{text}<extra></extra>",
        )
    )
    fig = _layout(
        fig, ROW_PX * len(rows) + 60, "Mean |SHAP| (average impact on log value; label = share)"
    )
    return _pad_x(fig, [r["mean_abs_shap"] for r in rows])


def lift_chart(rows: list[dict]) -> go.Figure:
    """Dot + 95% CI whisker per strategy; reference line at 1.0 (= matched baseline)."""
    c = colors()
    fig = go.Figure(
        go.Scatter(
            x=[r["value"] for r in rows],
            y=[r["label"] for r in rows],
            mode="markers+text",
            marker=dict(size=11, color=c["accent"], line=dict(width=2, color="rgba(0,0,0,0)")),
            error_x=dict(
                type="data",
                symmetric=False,
                thickness=2,
                width=0,
                color=c["accent"],
                array=[r["high"] - r["value"] for r in rows],
                arrayminus=[r["value"] - r["low"] for r in rows],
            ),
            text=[f"{r['value']:.2f}" for r in rows],
            textposition="top center",
            cliponaxis=False,
            textfont=dict(color=c["ink"], size=12),
            customdata=[[r["low"], r["high"]] for r in rows],
            hovertemplate="%{y}<br>Matched lift %{x:.2f}"
            "<br>95% CI %{customdata[0]:.2f} – %{customdata[1]:.2f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=1.0,
        line=dict(color=c["muted"], width=1),
        annotation_text="baseline",
        annotation_position="bottom",
        annotation_font=dict(color=c["muted"], size=11),
    )
    fig = _layout(
        fig, ROW_PX * len(rows) + 90, "Matched lift: hit rate ÷ rate expected from age × value mix"
    )
    # room above the first row for its value label and the baseline annotation
    fig.update_layout(margin=dict(t=36))
    fig.update_yaxes(autorange=False, range=[len(rows) - 0.5, -1.0])
    return fig
