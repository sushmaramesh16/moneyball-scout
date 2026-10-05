"""Undervalued scores.

raw:       (predicted - actual) / actual = exp(y_hat - y) - 1
de-biased: the log gap minus the mean log gap of the player's league x age-band cell in
           the same season, shrunk toward 0 for small cells. This removes systematic
           model bias (e.g. veterans or one league always looking cheap) so that
           players are compared with their peers.

Both use only information available at flag time: the current season's actual values
and predictions for every player.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.evaluate import AGE_BINS, AGE_LABELS, to_euros

SHRINK = 20  # pseudo-count: a cell of n players has n / (n + 20) of its mean gap removed


def score_frame(df: pd.DataFrame, y_hat: np.ndarray, shrink: int = SHRINK) -> pd.DataFrame:
    """Add predicted value and both scores to ``df`` (one season of panel rows)."""
    out = df.copy()
    out["y_hat"] = np.asarray(y_hat, float)
    out["pred_value"] = to_euros(out["y_hat"], out["market_index"])
    out["log_gap"] = out["y_hat"] - out["y"]
    out["undervalued_score"] = np.expm1(out["log_gap"])

    out["age_band"] = pd.cut(out["age"], AGE_BINS, labels=AGE_LABELS, right=False)
    cell = out.groupby(["league", "age_band"], observed=True)["log_gap"]
    out["cell_bias"] = cell.transform("sum") / (cell.transform("size") + shrink)
    out["log_gap_debiased"] = out["log_gap"] - out["cell_bias"]
    out["undervalued_score_debiased"] = np.expm1(out["log_gap_debiased"])
    return out
