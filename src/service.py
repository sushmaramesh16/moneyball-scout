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
from src.explain import report, shap_explain
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


def driver_summary(contributions: list[dict]) -> dict:
    """How much of the prediction (vs the average player) comes from context (age, club,
    league...) versus on-pitch performance. 'context' / 'performance' when one side
    supplies at least twice the lift of the other, else 'mixed'."""
    groups = {"context": set(fs.CONTEXT_FEATURES), "performance": set(fs.PERFORMANCE_FEATURES)}
    totals = {
        g: sum(c["shap"] for c in contributions if c["feature"] in members)
        for g, members in groups.items()
    }
    ctx, perf = totals["context"], totals["performance"]
    if ctx > 0 and ctx >= 2 * max(perf, 0):
        driver = "context"
    elif perf > 0 and perf >= 2 * max(ctx, 0):
        driver = "performance"
    else:
        driver = "mixed"
    return {
        "context_shap": ctx,
        "performance_shap": perf,
        "context_effect_pct": float(np.expm1(ctx) * 100),
        "performance_effect_pct": float(np.expm1(perf) * 100),
        "driver": driver,
    }


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

    @cached_property
    def intervals(self) -> dict | None:
        path = self.artifacts_dir / "prediction_intervals.json"
        return json.loads(path.read_text()) if path.exists() else None

    def value_range(self, predicted_value: float) -> dict | None:
        """Likely range for the actual value, from how far actual values fell from
        predictions on the held-out 2024/25 season (per predicted-value band)."""
        if not self.intervals:
            return None
        band = self.intervals["band_labels"][
            int(np.searchsorted(self.intervals["band_edges_eur"], predicted_value, "right"))
        ]
        q = self.intervals["bands"][band]
        return {
            "middle_50": [
                predicted_value * math.exp(q["q25"]),
                predicted_value * math.exp(q["q75"]),
            ],
            "middle_80": [
                predicted_value * math.exp(q["q10"]),
                predicted_value * math.exp(q["q90"]),
            ],
            "band": band,
            "source": self.intervals["source"],
        }

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
            "predicted_range": self.value_range(float(row["pred_value"])),
            "undervalued_score": _clean(row["undervalued_score_debiased"]),
            "undervalued_score_raw": _clean(row["undervalued_score"]),
            "contract_expiration_date": _clean(pd.Timestamp(row["contract_expiration_date"])),
            "stats": _record(row, STAT_FIELDS),
        }

    def explain(self, player_id: int) -> dict:
        """Full SHAP breakdown for a live player (stored at export time)."""
        row = self._row(player_id)
        mi = float(row["market_index"])
        contributions = []
        for f in self.features:
            shap = float(row[f"shap_{f}"])
            value = _clean(row[f])
            contributions.append(
                {
                    "feature": f,
                    "label": shap_explain.label(f),
                    "value": value,
                    "display": shap_explain.display_value(f, value, mi),
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
            "top_factors": contributions[:5],
            "drivers": driver_summary(contributions),
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
            "predicted_range": self.value_range(value),
            "top_factors": [
                f
                | {
                    "display": shap_explain.display_value(
                        f["feature"], f["value"], self.market_index
                    )
                }
                for f in json.loads(shap_explain.top_factors(shap_df, X)[0])
            ],
        }
        if actual_value:
            out["actual_value"] = float(actual_value)
            out["undervalued_score_raw"] = value / float(actual_value) - 1
        return out

    # ---- AI scouting report -----------------------------------------------------------
    def reports_enabled(self) -> bool:
        return report.is_available()

    @cached_property
    def reliability(self) -> dict:
        """Numbers the report may quote about the model's own accuracy."""
        tm = self._csv("test_metrics.csv").set_index("model").loc["lightgbm_tuned"]
        bt = self._csv("backtest.csv").query("season == 2024").set_index(["strategy", "k"])
        return {
            "typical_error_pct": float(tm["median_pct_error"] * 100),
            "hit_rate": float(bt.loc[("model_debiased", "top_decile"), "hit_raw"]),
            "base_rate": float(bt.loc[("all", "all"), "hit_raw"]),
        }

    def scouting_report(self, player_id: int, force: bool = False) -> dict:
        return report.generate(
            self.player(player_id), self.explain(player_id), self.reliability, force=force
        )

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
