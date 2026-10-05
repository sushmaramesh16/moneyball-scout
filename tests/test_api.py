"""API tests against the committed deployment artifacts (no raw data needed)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api.main import app
from src.features import feature_sets as fs

pytestmark = pytest.mark.skipif(
    not (config.ARTIFACTS_DIR / "model.joblib").exists(), reason="deployment artifacts not built"
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def live():
    return pd.read_parquet(config.ARTIFACTS_DIR / "players_live.parquet").set_index("player_id")


def test_health(client, live):
    body = client.get("/health").json()
    assert body == {
        "status": "ok",
        "season": "2025/26",
        "n_players": len(live),
        "reports_enabled": False,
    }


def test_undervalued_defaults_to_debiased_with_value_floor(client):
    rows = client.get("/undervalued", params={"limit": 25}).json()
    assert len(rows) == 25
    assert [r["rank"] for r in rows] == list(range(1, 26))
    assert all(r["actual_value"] >= config.VALUE_FLOOR for r in rows)
    scores = [r["undervalued_score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_undervalued_raw_score_and_no_floor(client):
    rows = client.get("/undervalued", params={"score": "raw", "min_value": 0, "limit": 50}).json()
    scores = [r["undervalued_score_raw"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert min(r["actual_value"] for r in rows) < config.VALUE_FLOOR


@pytest.mark.parametrize(
    "params, check",
    [
        ({"position_group": "Goalkeeper"}, lambda r, d: d["position_group"] == "Goalkeeper"),
        ({"league": "GB1"}, lambda r, d: r["league"] == "GB1"),
        ({"min_age": 20, "max_age": 23}, lambda r, d: 20 <= r["age"] <= 23),
        ({"min_value": 5_000_000}, lambda r, d: r["actual_value"] >= 5_000_000),
    ],
)
def test_undervalued_filters(client, live, params, check):
    rows = client.get("/undervalued", params={**params, "limit": 100}).json()
    assert rows
    for r in rows:
        assert check(r, live.loc[r["player_id"]])


def test_undervalued_rejects_bad_score(client):
    assert client.get("/undervalued", params={"score": "best"}).status_code == 422


def test_player_and_explain_are_consistent(client, live):
    pid = int(live["undervalued_score_debiased"].idxmax())
    player = client.get(f"/players/{pid}").json()
    assert player["actual_value"] == pytest.approx(live.loc[pid, "target_value"])
    ex = client.get(f"/explain/{pid}").json()
    assert len(ex["contributions"]) == len(fs.feature_columns("main"))
    total = ex["base_value"] + sum(c["shap"] for c in ex["contributions"])
    assert total == pytest.approx(ex["prediction"], abs=1e-6)
    assert np.exp(ex["prediction"] + ex["market_index"]) == pytest.approx(
        player["predicted_value"], rel=1e-9
    )
    mags = [abs(c["shap"]) for c in ex["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert len(ex["top_factors"]) == 5


def test_predict_reproduces_stored_prediction(client, live):
    pid = int(live.index[0])
    row = live.loc[pid]
    features = {
        f: (None if pd.isna(row[f]) else (row[f].item() if hasattr(row[f], "item") else row[f]))
        for f in fs.feature_columns("main")
    }
    body = client.post(
        "/predict", json={"features": features, "actual_value": float(row.target_value)}
    ).json()
    assert body["predicted_value"] == pytest.approx(row.pred_value, rel=1e-9)
    assert body["undervalued_score_raw"] == pytest.approx(row.undervalued_score, rel=1e-9)


def test_predict_with_partial_features_and_validation(client):
    ok = client.post("/predict", json={"features": {"age": 21, "league": "NL1"}})
    assert ok.status_code == 200 and ok.json()["predicted_value"] > 0
    assert client.post("/predict", json={"features": {"shoe_size": 44}}).status_code == 422
    bad_value = {"features": {"age": 21}, "actual_value": -5}
    assert client.post("/predict", json=bad_value).status_code == 422


def test_unknown_player_is_404(client):
    for path in ("/players/1", "/explain/1", "/report/1"):
        assert client.get(path).status_code == 404


def test_report_unavailable_until_enabled(client, live):
    assert client.get(f"/report/{int(live.index[0])}").status_code == 503


def test_search_and_backtest_and_importance(client):
    hits = client.get("/players", params={"q": "real madrid", "limit": 5}).json()
    assert hits and all("Real Madrid" in h["club_name"] for h in hits)
    bt = client.get("/backtest").json()
    assert set(bt) == {"results", "bootstrap", "missingness", "rank_ic", "floor", "test_metrics"}
    headline = [r for r in bt["bootstrap"] if r["strategy"] == "model_debiased"]
    assert {r["k_label"] for r in headline} == {"top_decile", "top_100"}
    imp = client.get("/model/importance", params={"top": 5}).json()
    assert len(imp) == 5 and imp[0]["mean_abs_shap"] >= imp[-1]["mean_abs_shap"]
