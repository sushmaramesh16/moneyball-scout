"""Final evaluation on the 2024/25 test season (once) and the live 2025/26 model.

Usage:
    python -m src.models.final test     # train <= 2023/24, score 2024/25; refuses to rerun
    python -m src.models.final live     # train <= 2024/25, score 2025/26, export artifacts

``live`` writes the deployment artifacts:
    artifacts/model.joblib          the full sklearn pipeline (preprocessing + LightGBM)
    artifacts/players_live.parquet  one row per 2025/26 player: features, actual and
                                    predicted value, both undervalued scores, SHAP values
                                    and top-5 SHAP factors
    artifacts/shap_global.csv       global mean |SHAP| importance
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src import config
from src.explain import shap_explain
from src.features import feature_sets as fs
from src.models import evaluate, pipelines, undervalued
from src.models.train import TRACKING_URI, _git_commit
from src.models.tune import load_best_params

REPORTS_DIR = config.ROOT / "reports" / "stage4"
TEST_MARKER = REPORTS_DIR / "test_metrics.csv"
MODEL_PATH = config.ARTIFACTS_DIR / "model.joblib"
LIVE_TABLE_PATH = config.ARTIFACTS_DIR / "players_live.parquet"
SHAP_GLOBAL_PATH = config.ARTIFACTS_DIR / "shap_global.csv"

LIVE_DISPLAY = [
    "player_id",
    "season",
    "season_label",
    "name",
    "club_id",
    "club_name",
    "league",
    "league_name",
    "citizenship",
    "image_url",
    "contract_expiration_date",
    "target_date",
    "target_value",
    "market_index",
    "y",
]


def _log_csv(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def run_test(force: bool = False) -> pd.DataFrame:
    if TEST_MARKER.exists() and not force:
        raise SystemExit(
            f"The test season was already evaluated ({TEST_MARKER}). "
            "Re-running it would turn the test set into a tuning set; "
            "pass --force only if you mean to."
        )
    warnings.filterwarnings("ignore", category=UserWarning)
    panel = pd.read_parquet(config.PANEL_PATH)
    train = panel[panel.season <= config.VALID_SEASON]
    test = panel[panel.season == config.TEST_SEASON].reset_index(drop=True)
    params = load_best_params()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("moneyball-test")
    rows, breakdowns = [], []
    candidates = [
        ("lightgbm_tuned", "main", "lightgbm", params),
        ("linear", "main", "linear", None),
        ("ceiling_lightgbm", "ceiling", "lightgbm", None),
        ("last_value_baseline", "ceiling", "last_value_baseline", None),
    ]
    for label, kind, name, p in candidates:
        features = fs.feature_columns(kind)
        model = pipelines.build(name, features, p).fit(train[features], train[fs.TARGET])
        y_hat = model.predict(test[features])
        value_hat = evaluate.to_euros(y_hat, test[fs.INDEX_COL])
        scores = evaluate.metrics(test[fs.TARGET], y_hat, test[fs.TARGET_VALUE], value_hat)
        bd = evaluate.breakdown(test, y_hat)
        with mlflow.start_run(run_name=f"test-{label}"):
            mlflow.set_tags({"stage": "4-final-test", "git_commit": _git_commit()})
            mlflow.log_params(
                {
                    "model": label,
                    "feature_set": kind,
                    "train_seasons": f"{config.FIRST_PANEL_SEASON}-{config.VALID_SEASON}",
                    "test_season": config.TEST_SEASON,
                    "params": json.dumps(p) if p else "default",
                }
            )
            mlflow.log_metrics({f"test_{k}": v for k, v in scores.items()})
            _log_csv(bd, "test_breakdown.csv")
        rows.append({"model": label, "feature_set": kind, **scores})
        breakdowns.append(bd.assign(model=label))
        print(
            f"{label:22s} rmse={scores['rmse']:.4f} r2={scores['r2']:.4f} "
            f"medae=€{scores['medae_eur']:,.0f}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(breakdowns).to_csv(REPORTS_DIR / "test_breakdowns.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(TEST_MARKER, index=False)
    return result


def run_live() -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)
    panel = pd.read_parquet(config.PANEL_PATH)
    train = panel[panel.season <= config.TEST_SEASON]
    live = panel[panel.season == config.LIVE_SEASON].reset_index(drop=True)
    features = fs.feature_columns("main")
    params = load_best_params()

    model = pipelines.build("lightgbm", features, params).fit(train[features], train[fs.TARGET])
    scored = undervalued.score_frame(live, model.predict(live[features]))
    shap_df = shap_explain.shap_frame(model, live[features])
    assert (shap_df.sum(axis=1) - scored["y_hat"]).abs().max() < 1e-6, "SHAP not additive"
    scored["top_factors"] = shap_explain.top_factors(shap_df, live[features])

    table = pd.concat(
        [
            scored[
                LIVE_DISPLAY
                + [c for c in features if c not in LIVE_DISPLAY]
                + [
                    "y_hat",
                    "pred_value",
                    "log_gap",
                    "undervalued_score",
                    "cell_bias",
                    "log_gap_debiased",
                    "undervalued_score_debiased",
                    "top_factors",
                ]
            ],
            shap_df.add_prefix("shap_"),
        ],
        axis=1,
    )
    table["contract_expiration_date"] = pd.to_datetime(table["contract_expiration_date"])

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    table.to_parquet(LIVE_TABLE_PATH, index=False, compression="zstd")
    glob = shap_explain.global_importance(shap_df)
    glob.to_csv(SHAP_GLOBAL_PATH, index=False)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("moneyball-main")
    with mlflow.start_run(run_name="live-lightgbm"):
        mlflow.set_tags({"stage": "4-live", "git_commit": _git_commit()})
        mlflow.log_params(
            {
                "train_seasons": f"{config.FIRST_PANEL_SEASON}-{config.TEST_SEASON}",
                "live_season": config.LIVE_SEASON,
                **params,
            }
        )
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(SHAP_GLOBAL_PATH))
        mlflow.log_metric("n_live_players", len(table))

    size = {p.name: p.stat().st_size / 1e6 for p in (MODEL_PATH, LIVE_TABLE_PATH)}
    print(
        f"live: {len(table):,} players; artifacts "
        + ", ".join(f"{k} {v:.1f} MB" for k, v in size.items())
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["test", "live"])
    parser.add_argument("--force", action="store_true", help="re-run the one-time test")
    args = parser.parse_args()
    if args.command == "test":
        run_test(force=args.force)
    else:
        run_live()


if __name__ == "__main__":
    main()
