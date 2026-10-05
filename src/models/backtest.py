"""Backtest: do players flagged as undervalued at season end see their value rise?

Usage:
    python -m src.models.backtest            # flag seasons 2023/24 and 2024/25

Walk-forward: for flag season S the models are trained on seasons <= S-1 (tuned LightGBM
params; the 2023/24 result is optimistic because those params were tuned on 2023/24).
At flag time (the target valuation date) each strategy ranks every player; the outcome is
the valuation ~12 months later, as raw log change and relative to the market index.

Strategies (higher score = picked first):
  model_raw        LightGBM log gap (predicted - actual)
  model_debiased   log gap minus its league x age-band mean
  linear_raw       log gap of the linear baseline
  young_regulars   under-23s ranked by minutes played
  mean_reversion   largest drop at the latest valuation update (bounce-back)
  all              everyone (the base rate)

Baselines for each flagged group: the base rate, and an age x value-band matched rate
(what the group's age/price mix alone would predict, since young and cheap players
rise more often regardless of any model).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.features import feature_sets as fs
from src.models import evaluate, pipelines, undervalued
from src.models.train import TRACKING_URI, _git_commit
from src.models.tune import load_best_params

REPORTS_DIR = config.ROOT / "reports" / "stage4"
FLAG_SEASONS = {2023: "optimistic (tuning season)", 2024: "headline (out-of-sample)"}
MODEL_STRATEGIES = ["model_raw", "model_debiased", "linear_raw"]
YOUNG_MAX_AGE = 23


def season_scores(panel: pd.DataFrame, season: int, params: dict) -> pd.DataFrame:
    """Train on seasons < ``season``, score ``season``; return its rows with all
    strategy scores attached."""
    train = panel[panel.season < season]
    flag = panel[panel.season == season].reset_index(drop=True)
    features = fs.feature_columns("main")

    lgbm = pipelines.build("lightgbm", features, params).fit(train[features], train[fs.TARGET])
    out = undervalued.score_frame(flag, lgbm.predict(flag[features]))
    linear = pipelines.build("linear", features).fit(train[features], train[fs.TARGET])
    out["linear_log_gap"] = linear.predict(flag[features]) - flag[fs.TARGET].to_numpy()

    out["s_model_raw"] = out["log_gap"]
    out["s_model_debiased"] = out["log_gap_debiased"]
    out["s_linear_raw"] = out["linear_log_gap"]
    out["s_young_regulars"] = out["minutes"].where(out["age"] < YOUNG_MAX_AGE)
    drop = -out["bt_last_update_log_change"]
    out["s_mean_reversion"] = drop.where(drop > 0)
    out["s_all"] = 0.0

    out["has_outcome"] = out["outcome_future_value"].notna()
    out["value_band"] = pd.cut(
        out["target_value"], evaluate.VALUE_BINS, labels=evaluate.VALUE_LABELS, right=False
    )
    out["cell"] = out["age_band"].astype(str) + " | " + out["value_band"].astype(str)
    out["hit_raw"] = (out["outcome_log_change"] > 0).where(out["has_outcome"])
    out["hit_rel"] = (out["outcome_rel_change"] > 0).where(out["has_outcome"])
    return out


def _flag(df: pd.DataFrame, strategy: str, k: int | None) -> pd.Series:
    """Boolean mask of the top-k rows by score (all rows for 'all'); NaN never flagged."""
    if strategy == "all":
        return pd.Series(True, index=df.index)
    score = df[f"s_{strategy}"]
    order = score.dropna().sort_values(ascending=False, kind="mergesort")
    return df.index.isin(order.index[:k])


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = hits / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def evaluate_strategy(df: pd.DataFrame, strategy: str, k: int | None, k_label: str) -> dict:
    flagged = _flag(df, strategy, k)
    picked = df[flagged]
    known = picked[picked.has_outcome]
    universe = df[df.has_outcome]
    rose = (df["outcome_log_change"] > 0).astype(float)  # NaN (no outcome) -> 0
    worst_base = rose.mean()
    worst_cell = rose.groupby(df["cell"]).mean()

    base_raw, base_rel = universe.hit_raw.mean(), universe.hit_rel.mean()
    cell_hit_raw = universe.groupby("cell")["hit_raw"].mean()
    cell_hit_rel = universe.groupby("cell")["hit_rel"].mean()
    cell_mean_rel = universe.groupby("cell")["outcome_rel_change"].mean()
    cell_mean_raw = universe.groupby("cell")["outcome_log_change"].mean()
    exp_raw = known["cell"].map(cell_hit_raw).mean()
    exp_rel = known["cell"].map(cell_hit_rel).mean()

    hits = int(known.hit_raw.sum())
    lo, hi = _wilson(hits, len(known))
    return {
        "strategy": strategy,
        "k": k_label,
        "n_flagged": int(flagged.sum()),
        "n_with_outcome": len(known),
        "missing_rate": 1 - len(known) / max(int(flagged.sum()), 1),
        "hit_raw": known.hit_raw.mean(),
        "hit_raw_ci_low": lo,
        "hit_raw_ci_high": hi,
        "hit_rel": known.hit_rel.mean(),
        "mean_log_change": known.outcome_log_change.mean(),
        "median_log_change": known.outcome_log_change.median(),
        "mean_rel_change": known.outcome_rel_change.mean(),
        "median_rel_change": known.outcome_rel_change.median(),
        "lift_raw": known.hit_raw.mean() / base_raw,
        "lift_rel": known.hit_rel.mean() / base_rel,
        "matched_expected_hit_raw": exp_raw,
        "matched_lift_raw": known.hit_raw.mean() / exp_raw,
        "matched_expected_hit_rel": exp_rel,
        "matched_lift_rel": known.hit_rel.mean() / exp_rel,
        "matched_excess_log_change": (
            known.outcome_log_change - known["cell"].map(cell_mean_raw)
        ).mean(),
        "matched_excess_rel_change": (
            known.outcome_rel_change - known["cell"].map(cell_mean_rel)
        ).mean(),
        # Robustness: a missing 12-month valuation counts as "did not rise".
        "hit_raw_missing_as_fail": hits / max(len(picked), 1),
        "lift_raw_missing_as_fail": (hits / max(len(picked), 1)) / worst_base,
        "matched_lift_raw_missing_as_fail": (hits / max(len(picked), 1))
        / picked["cell"].map(worst_cell).mean(),
    }


def missingness(df: pd.DataFrame, strategy: str, k: int) -> dict:
    """Are flagged players missing a 12-month-later valuation more/less often?"""
    flagged = _flag(df, strategy, k)
    a, b = ~df.has_outcome[flagged], ~df.has_outcome[~flagged]
    pooled = (a.sum() + b.sum()) / (len(a) + len(b))
    se = np.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
    z = (a.mean() - b.mean()) / se if se > 0 else 0.0
    return {
        "strategy": strategy,
        "missing_flagged": a.mean(),
        "missing_others": b.mean(),
        "difference": a.mean() - b.mean(),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def rank_ic(df: pd.DataFrame, strategy: str, outcome: str) -> dict:
    """Spearman correlation between score and outcome over all players with an outcome
    (cutoff-free view of ranking quality), with a Fisher-z 95% interval."""
    d = df[df.has_outcome & df[f"s_{strategy}"].notna()]
    rho = stats.spearmanr(d[f"s_{strategy}"], d[outcome]).statistic
    half = 1.96 / np.sqrt(len(d) - 3)
    return {
        "strategy": strategy,
        "outcome": outcome,
        "n": len(d),
        "ic": rho,
        "ci_low": np.tanh(np.arctanh(rho) - half),
        "ci_high": np.tanh(np.arctanh(rho) + half),
    }


def run_season(panel: pd.DataFrame, season: int, params: dict):
    df = season_scores(panel, season, params)
    decile = int(np.ceil(len(df) / 10))
    rows = [evaluate_strategy(df, "all", None, "all")]
    for strategy in [*MODEL_STRATEGIES, "young_regulars", "mean_reversion"]:
        rows.append(evaluate_strategy(df, strategy, decile, "top_decile"))
    for strategy in MODEL_STRATEGIES:
        for k in (100, 50):
            rows.append(evaluate_strategy(df, strategy, k, f"top_{k}"))
    results = pd.DataFrame(rows)

    miss = pd.DataFrame(
        [
            missingness(df, s, decile)
            for s in [*MODEL_STRATEGIES, "young_regulars", "mean_reversion"]
        ]
    )
    ic = pd.DataFrame(
        [
            rank_ic(df, s, o)
            for s in [*MODEL_STRATEGIES, "mean_reversion"]
            for o in ("outcome_log_change", "outcome_rel_change")
        ]
    )
    for frame in (results, miss, ic):
        frame.insert(0, "season", season)
        frame.insert(1, "label", FLAG_SEASONS[season])
    return df, results, miss, ic


def _log_csv(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("moneyball-backtest")
    panel = pd.read_parquet(config.PANEL_PATH)
    params = load_best_params()

    all_results, all_miss, all_ic = [], [], []
    for season, label in FLAG_SEASONS.items():
        with mlflow.start_run(run_name=f"backtest-{season}"):
            mlflow.set_tags({"stage": "4-backtest", "label": label, "git_commit": _git_commit()})
            mlflow.log_params(
                {
                    "flag_season": season,
                    "train_through": season - 1,
                    "lgbm_params": json.dumps(params),
                }
            )
            df, results, miss, ic = run_season(panel, season, params)
            for r in results.itertuples():
                for metric in (
                    "hit_raw",
                    "hit_rel",
                    "matched_lift_raw",
                    "matched_lift_rel",
                    "mean_rel_change",
                    "missing_rate",
                ):
                    mlflow.log_metric(f"{r.strategy}_{r.k}_{metric}", getattr(r, metric))
            for r in ic.itertuples():
                mlflow.log_metric(f"ic_{r.strategy}_{r.outcome}", r.ic)
            _log_csv(results, f"backtest_{season}.csv")
            _log_csv(miss, f"missingness_{season}.csv")
            _log_csv(ic, f"rank_ic_{season}.csv")
        keep = [
            "player_id",
            "season",
            "name",
            "club_name",
            "league",
            "age",
            "target_value",
            "pred_value",
            "log_gap",
            "log_gap_debiased",
            "linear_log_gap",
            "outcome_future_value",
            "outcome_log_change",
            "outcome_rel_change",
        ]
        df[keep].to_parquet(config.PROCESSED_DIR / f"backtest_scores_{season}.parquet")
        all_results.append(results)
        all_miss.append(miss)
        all_ic.append(ic)
        print(f"season {season} ({label}) done")

    pd.concat(all_results).to_csv(REPORTS_DIR / "backtest.csv", index=False)
    pd.concat(all_miss).to_csv(REPORTS_DIR / "backtest_missingness.csv", index=False)
    pd.concat(all_ic).to_csv(REPORTS_DIR / "backtest_rank_ic.csv", index=False)


if __name__ == "__main__":
    main()
