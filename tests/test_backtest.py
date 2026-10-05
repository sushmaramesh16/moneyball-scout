import numpy as np
import pandas as pd
import pytest

from src.models import backtest


def _frame():
    """Ten players, two cells. Score s_x ranks players 0..9 descending."""
    df = pd.DataFrame(
        {
            "s_x": [9, 8, 7, 6, 5, 4, 3, 2, 1, np.nan],
            "cell": ["young"] * 5 + ["old"] * 5,
            "has_outcome": [True] * 8 + [False, False],
            "outcome_log_change": [0.5, 0.2, -0.1, 0.3, 0.1, -0.2, -0.3, 0.1, np.nan, np.nan],
        }
    )
    df["outcome_rel_change"] = df["outcome_log_change"] - 0.1
    df["hit_raw"] = (df.outcome_log_change > 0).where(df.has_outcome)
    df["hit_rel"] = (df.outcome_rel_change > 0).where(df.has_outcome)
    return df


def test_flag_takes_top_k_and_never_nan():
    df = _frame()
    assert backtest._flag(df, "x", 3).tolist() == [True] * 3 + [False] * 7
    assert backtest._flag(df, "x", 100).sum() == 9  # NaN score never flagged
    assert backtest._flag(df, "all", None).all()


def test_evaluate_strategy_rates_and_matched_baseline():
    df = _frame()
    r = backtest.evaluate_strategy(df, "x", 3, "top_3")
    assert r["n_flagged"] == 3 and r["n_with_outcome"] == 3
    assert r["hit_raw"] == pytest.approx(2 / 3)
    base = df[df.has_outcome].hit_raw.mean()  # 5 of 8
    assert r["lift_raw"] == pytest.approx((2 / 3) / base)
    # all three flagged are "young"; young hit rate among known = 4/5
    assert r["matched_expected_hit_raw"] == pytest.approx(4 / 5)
    assert r["matched_lift_raw"] == pytest.approx((2 / 3) / (4 / 5))
    young_mean = df[df.has_outcome & (df.cell == "young")].outcome_log_change.mean()
    assert r["matched_excess_log_change"] == pytest.approx(np.mean([0.5, 0.2, -0.1]) - young_mean)


def test_wilson_interval():
    lo, hi = backtest._wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert all(np.isnan(backtest._wilson(0, 0)))


def test_missingness_compares_flagged_with_others():
    df = _frame()
    m = backtest.missingness(df, "x", 9)  # flags rows 0-8; row 8 missing, row 9 missing
    assert m["missing_flagged"] == pytest.approx(1 / 9)
    assert m["missing_others"] == pytest.approx(1.0)


def test_missing_outcomes_count_as_failures_in_robustness_check():
    df = _frame()
    r = backtest.evaluate_strategy(df, "x", 9, "top_9")
    # 5 risers among the 9 flagged (one of which has no outcome)
    assert r["hit_raw_missing_as_fail"] == pytest.approx(5 / 9)
    assert r["lift_raw_missing_as_fail"] == pytest.approx((5 / 9) / (5 / 10))
