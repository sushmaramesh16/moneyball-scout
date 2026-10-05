import numpy as np
import pandas as pd
import pytest

from src.features import feature_sets as fs
from src.models import evaluate, pipelines


def test_category_caster_freezes_levels_and_pools_rare_ones():
    train = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 5 + ["rare"]})
    caster = pipelines.CategoryCaster(["c"], min_count=3).fit(train)
    assert caster.levels_["c"] == ["a", "b", "other"]
    out = caster.transform(pd.DataFrame({"c": ["a", "rare", "unseen", None]}))
    assert list(out["c"].cat.categories) == ["a", "b", "other"]
    assert out["c"].tolist()[:3] == ["a", "other", "other"]
    assert pd.isna(out["c"].iloc[3])


def test_category_caster_without_rare_levels_maps_unseen_to_nan():
    caster = pipelines.CategoryCaster(["c"], min_count=1).fit(pd.DataFrame({"c": ["a", "b"]}))
    assert pd.isna(caster.transform(pd.DataFrame({"c": ["zzz"]}))["c"].iloc[0])


def test_baselines():
    X = pd.DataFrame({"val_last_rel": [1.0, np.nan]})
    y = np.array([2.0, 4.0])
    assert pipelines.MeanBaseline().fit(X, y).predict(X).tolist() == [3.0, 3.0]
    assert pipelines.LastValueBaseline().fit(X, y).predict(X).tolist() == [1.0, 3.0]


@pytest.mark.parametrize("name", ["linear", "random_forest", "xgboost", "lightgbm"])
@pytest.mark.parametrize("kind", ["main", "ceiling"])
def test_every_pipeline_fits_and_predicts_on_the_panel(panel, name, kind):
    features = fs.feature_columns(kind)
    model = pipelines.build(name, features).fit(panel[features], panel[fs.TARGET])
    y_hat = model.predict(panel[features])
    assert y_hat.shape == (len(panel),) and np.isfinite(y_hat).all()


def test_metrics_perfect_and_euro_scale():
    y = np.array([0.0, 1.0, 2.0])
    m = evaluate.metrics(y, y, value=[1, 2, 3], value_hat=[1, 2, 5])
    assert m["rmse"] == 0 and m["r2"] == 1 and m["bias"] == 0
    assert m["mae_eur"] == pytest.approx(2 / 3) and m["medae_eur"] == 0
    assert evaluate.to_euros([0.0], [np.log(1e6)])[0] == pytest.approx(1e6)


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    good, bad = y + rng.normal(scale=0.1, size=2000), y + rng.normal(scale=0.5, size=2000)
    gap = evaluate.paired_bootstrap_rmse_diff(y, good, bad, n_boot=300)
    assert gap["ci_high"] < 0
    same = evaluate.paired_bootstrap_rmse_diff(y, good, good, n_boot=50)
    assert same["diff"] == 0 and same["ci_low"] == same["ci_high"] == 0
