import json

import numpy as np
import pandas as pd
import pytest

from src.explain import shap_explain
from src.features import feature_sets as fs
from src.models import pipelines, undervalued


@pytest.fixture
def fitted(panel):
    features = fs.feature_columns("main")
    params = {"n_estimators": 20, "min_child_samples": 1, "num_leaves": 4}
    model = pipelines.build("lightgbm", features, params).fit(panel[features], panel[fs.TARGET])
    return model, panel[features]


def test_shap_values_sum_to_prediction(fitted):
    model, X = fitted
    shap_df = shap_explain.shap_frame(model, X)
    assert list(shap_df.columns[:-1]) == list(X.columns)
    np.testing.assert_allclose(shap_df.sum(axis=1), model.predict(X), atol=1e-8)


def test_shap_matches_shap_library(fitted):
    shap = pytest.importorskip("shap")
    model, X = fitted
    Xt = model[:-1].transform(X)
    try:
        ref = shap.TreeExplainer(model[-1]).shap_values(Xt)
    except Exception as exc:  # shap versions differ in categorical-split support
        pytest.skip(f"shap cannot parse this model: {exc}")
    ours = shap_explain.shap_frame(model, X).drop(columns=shap_explain.BASE).to_numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_top_factors_are_sorted_and_labelled(fitted):
    model, X = fitted
    shap_df = shap_explain.shap_frame(model, X)
    factors = [json.loads(s) for s in shap_explain.top_factors(shap_df, X, k=3)]
    assert all(len(f) == 3 for f in factors)
    for row_factors in factors:
        mags = [abs(f["shap"]) for f in row_factors]
        assert mags == sorted(mags, reverse=True)
        for f in row_factors:
            assert f["label"] and f["effect_pct"] == pytest.approx(
                np.expm1(f["shap"]) * 100, abs=0.06
            )


def test_global_importance_shares_sum_to_one(fitted):
    model, X = fitted
    imp = shap_explain.global_importance(shap_explain.shap_frame(model, X))
    assert imp["share"].sum() == pytest.approx(1.0)
    assert imp["mean_abs_shap"].is_monotonic_decreasing


def _season(n_a=40, n_b=10):
    return pd.DataFrame(
        {
            "league": ["A"] * n_a + ["B"] * n_b,
            "age": [25.0] * (n_a + n_b),
            "y": [0.0] * (n_a + n_b),
            "market_index": [np.log(1e6)] * (n_a + n_b),
        }
    )


def test_raw_undervalued_score():
    df = _season(2, 0)
    out = undervalued.score_frame(df, np.array([np.log(1.5), 0.0]))
    assert out["undervalued_score"].tolist() == pytest.approx([0.5, 0.0])
    assert out["pred_value"].tolist() == pytest.approx([1.5e6, 1e6])


def test_debiased_score_removes_shrunk_cell_mean():
    df = _season()
    y_hat = np.r_[np.full(40, 0.2), np.full(10, -0.1)]
    out = undervalued.score_frame(df, y_hat, shrink=20)
    a, b = out[out.league == "A"].iloc[0], out[out.league == "B"].iloc[0]
    assert a.cell_bias == pytest.approx(0.2 * 40 / 60)
    assert b.cell_bias == pytest.approx(-0.1 * 10 / 30)
    assert a.log_gap_debiased == pytest.approx(0.2 - 0.2 * 40 / 60)
    # within a cell the ranking is unchanged
    assert (out.groupby("league").log_gap_debiased.nunique() == 1).all()
