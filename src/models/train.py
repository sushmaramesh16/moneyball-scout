"""Train and compare models on the time split, tracking every run in MLflow.

Usage:
    python -m src.models.train compare     # fit on <= 2022/23, score 2023/24

Two MLflow experiments: ``moneyball-main`` (no valuation history) and
``moneyball-ceiling`` (adds valuation history; a reference point, not a product model).
The 2024/25 test season is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.features import feature_sets as fs
from src.models import evaluate, pipelines

REPORTS_DIR = config.ROOT / "reports" / "stage3"
TRACKING_URI = f"sqlite:///{config.ROOT / 'mlflow.db'}"

EXPERIMENTS = {
    "moneyball-main": ("main", ["mean_baseline", "linear", "random_forest", "xgboost", "lightgbm"]),
    "moneyball-ceiling": (
        "ceiling",
        ["last_value_baseline", "linear", "random_forest", "xgboost", "lightgbm"],
    ),
}


def load_split(panel_path: Path = config.PANEL_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (<= TRAIN_LAST_SEASON) and validation (VALID_SEASON). Test is not returned."""
    panel = pd.read_parquet(panel_path)
    train = panel[panel.season <= config.TRAIN_LAST_SEASON].reset_index(drop=True)
    valid = panel[panel.season == config.VALID_SEASON].reset_index(drop=True)
    return train, valid


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _model_params(model) -> dict:
    """Scalar hyperparameters of the final estimator, for MLflow."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return {
        f"hp_{k}": v
        for k, v in est.get_params().items()
        if isinstance(v, int | float | str | bool) and v is not None
    }


def _log_frame(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path))


def fit_and_score(name: str, feature_kind: str, train: pd.DataFrame, valid: pd.DataFrame):
    features = fs.feature_columns(feature_kind)
    model = pipelines.build(name, features)
    t0 = time.time()
    model.fit(train[features], train[fs.TARGET])
    fit_seconds = time.time() - t0
    y_hat = model.predict(valid[features])
    value_hat = evaluate.to_euros(y_hat, valid[fs.INDEX_COL])
    scores = evaluate.metrics(valid[fs.TARGET], y_hat, valid[fs.TARGET_VALUE], value_hat)
    return model, y_hat, scores, fit_seconds


def importance_table(model, valid: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Permutation importance on validation (RMSE increase), plus native gain if any."""
    perm = permutation_importance(
        model,
        valid[features],
        valid[fs.TARGET],
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=pipelines.SEED,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": features,
            "perm_rmse_increase": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    )
    est = model.steps[-1][1]
    if hasattr(est, "booster_"):  # LightGBM
        gain = est.booster_.feature_importance(importance_type="gain")
        table["gain_share"] = gain / gain.sum()
    elif hasattr(est, "get_booster"):  # XGBoost
        scores = est.get_booster().get_score(importance_type="total_gain")
        gain = np.array([scores.get(f, 0.0) for f in features])
        table["gain_share"] = gain / gain.sum()
    return table.sort_values("perm_rmse_increase", ascending=False, ignore_index=True)


def compare() -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=UserWarning)
    mlflow.set_tracking_uri(TRACKING_URI)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train, valid = load_split()
    print(
        f"train {len(train):,} rows (seasons {train.season.min()}-{train.season.max()}), "
        f"validation {len(valid):,} rows (season {config.VALID_SEASON})\n"
    )

    results, breakdowns, fitted, preds = [], [], {}, {}
    for experiment, (feature_kind, model_names) in EXPERIMENTS.items():
        mlflow.set_experiment(experiment)
        for name in model_names:
            with mlflow.start_run(run_name=name):
                model, y_hat, scores, secs = fit_and_score(name, feature_kind, train, valid)
                mlflow.set_tags({"stage": "3-comparison", "git_commit": _git_commit()})
                mlflow.log_params(
                    {
                        "model": name,
                        "feature_set": feature_kind,
                        "n_features": len(fs.feature_columns(feature_kind)),
                        "train_seasons": f"{config.FIRST_PANEL_SEASON}-{config.TRAIN_LAST_SEASON}",
                        "valid_season": config.VALID_SEASON,
                        "seed": pipelines.SEED,
                        "n_train": len(train),
                        "n_valid": len(valid),
                        **_model_params(model),
                    }
                )
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in scores.items()} | {"fit_seconds": secs}
                )
                bd = evaluate.breakdown(valid, y_hat)
                _log_frame(bd, "val_breakdown.csv")
                mlflow.log_text(
                    json.dumps(fs.feature_columns(feature_kind), indent=1), "features.json"
                )

            results.append({"experiment": experiment, "model": name, **scores, "fit_seconds": secs})
            breakdowns.append(bd.assign(experiment=experiment, model=name))
            fitted[(experiment, name)] = model
            preds[(experiment, name)] = y_hat
            print(
                f"{experiment:18s} {name:20s} rmse={scores['rmse']:.4f} "
                f"r2={scores['r2']:.4f} medae=€{scores['medae_eur']:,.0f} ({secs:.0f}s)"
            )

    results = pd.DataFrame(results)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    pd.concat(breakdowns).to_csv(REPORTS_DIR / "val_breakdowns.csv", index=False)

    # Feature importance for the best main model, logged to its run as well.
    main = results[results.experiment == "moneyball-main"]
    ranked = main.sort_values("rmse")["model"].tolist()
    best, runner_up = ranked[0], ranked[1]
    gap = evaluate.paired_bootstrap_rmse_diff(
        valid[fs.TARGET],
        preds[("moneyball-main", best)],
        preds[("moneyball-main", runner_up)],
    )
    print(
        f"\nRMSE {best} - {runner_up}: {gap['diff']:+.4f} "
        f"(95% CI {gap['ci_low']:+.4f} to {gap['ci_high']:+.4f})"
    )
    features = fs.feature_columns("main")
    imp = importance_table(fitted[("moneyball-main", best)], valid, features)
    imp.to_csv(REPORTS_DIR / f"importance_{best}.csv", index=False)
    mlflow.set_experiment("moneyball-main")
    run = mlflow.search_runs(
        filter_string=f"attributes.run_name = '{best}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    with mlflow.start_run(run_id=run.run_id.iloc[0]):
        _log_frame(imp, "permutation_importance.csv")
        mlflow.set_tag("best_main_stage3", "true")
        mlflow.log_metrics({f"rmse_gap_vs_{runner_up}_{k}": v for k, v in gap.items()})
    print(f"\nbest main model: {best}; reports in {REPORTS_DIR.relative_to(config.ROOT)}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["compare"])
    args = parser.parse_args()
    if args.command == "compare":
        compare()


if __name__ == "__main__":
    main()
