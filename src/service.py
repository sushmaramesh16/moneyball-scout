"""Scouting service: everything the API and the Streamlit app need, from small artifacts.

Loads only ``artifacts/`` (model pipeline + live player table + SHAP importance) and the
backtest CSVs in ``reports/stage4/``. No raw data, no DuckDB. FastAPI wraps this class
locally; the deployed Streamlit app uses it directly.
"""

from __future__ import annotations

import json
import math
from functools import cached_property, lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.explain import shap_explain
from src.features import feature_sets as fs

REPORTS_DIR = config.ROOT / "reports" / "stage4"
SCORE_COLUMNS = {"debiased": "undervalued_score_debiased", "raw": "undervalued_score"}

PROFILE_FIELDS = [
    "player_id",
    "name",
    "club_name",
    "league",
    "league_name",
    "position",
    "position_group",
    "age",
    "foot",
    "height_cm",
    "citizenship",
    "image_url",
    "season_label",
]
STAT_FIELDS = [
    "apps",
    "starts",
    "minutes",
    "goals",
    "assists",
    "ga_per90",
    "yellow_cards",
    "red_cards",
    "league_minutes_share",
    "euro_minutes",
    "captain_games",
    "prev_minutes",
    "club_position",
    "club_ppg",
    "club_in_europe",
]


class PlayerNotFound(KeyError):
    pass


def _clean(value):
    """JSON-safe scalar: NaN/NaT -> None, numpy -> python, timestamps -> ISO date."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date().isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _record(row: pd.Series, fields: list[str]) -> dict:
    return {f: _clean(row.get(f)) for f in fields}


class ScoutService:
    def __init__(self, artifacts_dir: Path = config.ARTIFACTS_DIR, reports_dir: Path = REPORTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)
        self.features = fs.feature_columns("main")

    # ---- loading -------------------------------------------------------------------
    @cached_property
    def model(self):
        return joblib.load(self.artifacts_dir / "model.joblib")

    @cached_property
    def players(self) -> pd.DataFrame:
        df = pd.read_parquet(self.artifacts_dir / "players_live.parquet")
        df["league_name"] = df["league"].map(config.LEAGUE_NAMES).fillna(df["league_name"])
        return df.set_index("player_id", drop=False)

    @cached_property
    def shap_global(self) -> pd.DataFrame:
        return pd.read_csv(self.artifacts_dir / "shap_global.csv")

    @property
    def market_index(self) -> float:
        """Live season market level (log euros); converts model output to euros."""
        return float(self.players["market_index"].iloc[0])

    @property
    def season_label(self) -> str:
        return str(self.players["season_label"].iloc[0])

    # ---- lookups -------------------------------------------------------------------
    def _row(self, player_id: int) -> pd.Series:
        try:
            return self.players.loc[int(player_id)]
        except KeyError as exc:
            raise PlayerNotFound(player_id) from exc

    def filter_options(self) -> dict:
        p = self.players
        return {
            "leagues": sorted(
                {(r.league, r.league_name) for r in p[["league", "league_name"]].itertuples()}
            ),
            "position_groups": sorted(p["position_group"].dropna().unique().tolist()),
            "positions": sorted(p["position"].dropna().unique().tolist()),
            "age_range": [math.floor(p["age"].min()), math.ceil(p["age"].max())],
            "season_label": self.season_label,
            "default_min_value": config.VALUE_FLOOR,
        }

    def search(self, query: str = "", limit: int = 20) -> list[dict]:
        p = self.players
        if query:
            q = query.strip().lower()
            hit = p["name"].str.lower().str.contains(q, regex=False) | p[
                "club_name"
            ].str.lower().str.contains(q, regex=False)
            p = p[hit]
        p = p.sort_values("target_value", ascending=False).head(limit)
        return [self._summary(r) for _, r in p.iterrows()]

    def _summary(self, row: pd.Series) -> dict:
        return {
            **_record(row, ["player_id", "name", "club_name", "league", "position", "age"]),
            "actual_value": _clean(row["target_value"]),
            "predicted_value": _clean(row["pred_value"]),
            "undervalued_score": _clean(row["undervalued_score_debiased"]),
            "undervalued_score_raw": _clean(row["undervalued_score"]),
        }

    def player(self, player_id: int) -> dict:
        row = self._row(player_id)
        return {
            **_record(row, PROFILE_FIELDS),
            "actual_value": _clean(row["target_value"]),
            "valuation_date": _clean(pd.Timestamp(row["target_date"])),
            "predicted_value": _clean(row["pred_value"]),
            "undervalued_score": _clean(row["undervalued_score_debiased"]),
            "undervalued_score_raw": _clean(row["undervalued_score"]),
            "contract_expiration_date": _clean(pd.Timestamp(row["contract_expiration_date"])),
            "stats": _record(row, STAT_FIELDS),
        }

    def explain(self, player_id: int) -> dict:
        """Full SHAP breakdown for a live player (stored at export time)."""
        row = self._row(player_id)
        contributions = []
        for f in self.features:
            shap = float(row[f"shap_{f}"])
            contributions.append(
                {
                    "feature": f,
                    "label": shap_explain.label(f),
                    "value": _clean(row[f]),
                    "shap": shap,
                    "effect_pct": float(np.expm1(shap) * 100),
                }
            )
        contributions.sort(key=lambda c: abs(c["shap"]), reverse=True)
        return {
            "player_id": int(player_id),
            "name": row["name"],
            "base_value": float(row[f"shap_{shap_explain.BASE}"]),
            "prediction": float(row["y_hat"]),
            "market_index": float(row["market_index"]),
            "predicted_value": float(row["pred_value"]),
            "actual_value": float(row["target_value"]),
            "contributions": contributions,
            "top_factors": json.loads(row["top_factors"]),
        }

    def undervalued(
        self,
        position_group: str | None = None,
        position: str | None = None,
        league: str | None = None,
        min_age: float | None = None,
        max_age: float | None = None,
        min_value: float | None = config.VALUE_FLOOR,
        score: str = "debiased",
        limit: int = 50,
    ) -> list[dict]:
        if score not in SCORE_COLUMNS:
            raise ValueError(f"score must be one of {sorted(SCORE_COLUMNS)}")
        p = self.players
        mask = pd.Series(True, index=p.index)
        if position_group:
            mask &= p["position_group"] == position_group
        if position:
            mask &= p["position"] == position
        if league:
            mask &= p["league"] == league
        if min_age is not None:
            mask &= p["age"] >= min_age
        if max_age is not None:
            mask &= p["age"] <= max_age
        if min_value:
            mask &= p["target_value"] >= min_value
        ranked = p[mask].sort_values(SCORE_COLUMNS[score], ascending=False).head(limit)
        return [{"rank": i + 1, **self._summary(r)} for i, (_, r) in enumerate(ranked.iterrows())]

    def predict(self, features: dict, actual_value: float | None = None) -> dict:
        """Score an arbitrary feature vector (missing features -> NaN, as in training)."""
        unknown = set(features) - set(self.features)
        if unknown:
            raise ValueError(f"unknown features: {sorted(unknown)}")
        X = pd.DataFrame([{f: features.get(f) for f in self.features}])
        numeric = [f for f in self.features if f not in fs.CATEGORICAL]
        X[numeric] = X[numeric].astype("float64")
        y_hat = float(self.model.predict(X)[0])
        shap_df = shap_explain.shap_frame(self.model, X)
        value = float(np.exp(y_hat + self.market_index))
        out = {
            "prediction": y_hat,
            "predicted_value": value,
            "market_index": self.market_index,
            "top_factors": json.loads(shap_explain.top_factors(shap_df, X)[0]),
        }
        if actual_value:
            out["actual_value"] = float(actual_value)
            out["undervalued_score_raw"] = value / float(actual_value) - 1
        return out

    # ---- model & backtest reports ----------------------------------------------------
    def _csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.reports_dir / name)

    def global_importance(self, top: int = 15) -> list[dict]:
        cols = ["feature", "label", "mean_abs_shap", "share"]
        return self.shap_global[cols].head(top).to_dict("records")

    def backtest_summary(self) -> dict:
        def records(df: pd.DataFrame) -> list[dict]:
            return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]

        return {
            "results": records(self._csv("backtest.csv")),
            "bootstrap": records(self._csv("backtest_bootstrap.csv")),
            "missingness": records(self._csv("backtest_missingness.csv")),
            "rank_ic": records(self._csv("backtest_rank_ic.csv")),
            "floor": records(self._csv("backtest_floor.csv")),
            "test_metrics": records(self._csv("test_metrics.csv")),
        }


@lru_cache(maxsize=1)
def get_service() -> ScoutService:
    return ScoutService()
