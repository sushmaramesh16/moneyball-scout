"""Metrics on the relative-log scale and in euros, plus error breakdowns by segment."""

from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BINS = [0, 21, 25, 29, 33, 99]
AGE_LABELS = ["<21", "21-24", "25-28", "29-32", "33+"]
VALUE_BINS = [0, 1e6, 5e6, 20e6, np.inf]
VALUE_LABELS = ["<€1M", "€1-5M", "€5-20M", "€20M+"]


def to_euros(y_rel: np.ndarray, market_index: np.ndarray) -> np.ndarray:
    """Relative-log prediction -> euros. exp() of a log-scale prediction estimates the
    median, not the mean, which is what we want for a typical-value estimate."""
    return np.exp(np.asarray(y_rel) + np.asarray(market_index))


def metrics(y, y_hat, value, value_hat) -> dict[str, float]:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    resid = y_hat - y
    abs_eur = np.abs(np.asarray(value_hat, float) - np.asarray(value, float))
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)),
        "bias": float(np.mean(resid)),
        "mae_eur": float(abs_eur.mean()),
        "medae_eur": float(np.median(abs_eur)),
        # typical multiplicative miss: exp(median |log error|) - 1
        "median_pct_error": float(np.expm1(np.median(np.abs(resid)))),
    }


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    df["value_band"] = pd.cut(df["target_value"], VALUE_BINS, labels=VALUE_LABELS, right=False)
    return df


BREAKDOWN_DIMS = ("position_group", "league", "age_band", "value_band", "pred_value_band")


def breakdown(df: pd.DataFrame, y_hat: np.ndarray, by=BREAKDOWN_DIMS) -> pd.DataFrame:
    """Per-segment metrics. ``df`` needs y, target_value, market_index, age.

    ``value_band`` groups by the *actual* value; any noisy predictor looks biased there
    (cheap players over-, stars under-predicted) because it conditions on the outcome.
    ``pred_value_band`` groups by the *predicted* value: bias there is miscalibration.
    """
    df = add_segments(df).assign(y_hat=y_hat)
    df["value_hat"] = to_euros(df["y_hat"], df["market_index"])
    df["pred_value_band"] = pd.cut(df["value_hat"], VALUE_BINS, labels=VALUE_LABELS, right=False)
    rows = []
    for dim in by:
        for level, g in df.groupby(dim, observed=True):
            m = metrics(g["y"], g["y_hat"], g["target_value"], g["value_hat"])
            rows.append({"dimension": dim, "segment": str(level), "n": len(g), **m})
    return pd.DataFrame(rows)


def paired_bootstrap_rmse_diff(
    y, y_hat_a, y_hat_b, n_boot: int = 2000, seed: int = 42
) -> dict[str, float]:
    """RMSE(a) - RMSE(b) on the same rows, with a 95% bootstrap interval.
    A negative difference whose interval excludes 0 means model a is reliably better."""
    rng = np.random.default_rng(seed)
    y, a, b = (np.asarray(v, float) for v in (y, y_hat_a, y_hat_b))
    ea, eb = (a - y) ** 2, (b - y) ** 2
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = np.sqrt(ea[idx].mean(axis=1)) - np.sqrt(eb[idx].mean(axis=1))
    return {
        "diff": float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
