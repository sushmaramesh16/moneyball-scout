"""Optuna tuning of the main LightGBM model on the 2023/24 validation season.

Usage:
    python -m src.models.tune --trials 60

Each trial trains on 2015/16-2022/23 and is scored on 2023/24 RMSE (relative-log scale).
Every trial is a nested MLflow run under one parent. The best parameters are written to
artifacts/best_params.json, which the final test and live models read.
"""

from __future__ import annotations

import argparse
import json
import warnings

import mlflow
import optuna
import pandas as pd

from src import config
from src.features import feature_sets as fs
from src.models import evaluate, pipelines
from src.models.train import TRACKING_URI, _git_commit, load_split

BEST_PARAMS_PATH = config.ARTIFACTS_DIR / "best_params.json"
REPORTS_DIR = config.ROOT / "reports" / "stage4"


def search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 3000, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 50.0, log=True),
    }


def load_best_params() -> dict:
    return json.loads(BEST_PARAMS_PATH.read_text())["params"]


def tune(n_trials: int) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("moneyball-main")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train, valid = load_split()
    features = fs.feature_columns("main")
    X_tr, y_tr, X_va, y_va = train[features], train[fs.TARGET], valid[features], valid[fs.TARGET]

    def score(params: dict | None):
        model = pipelines.build("lightgbm", features, params).fit(X_tr, y_tr)
        y_hat = model.predict(X_va)
        value_hat = evaluate.to_euros(y_hat, valid[fs.INDEX_COL])
        return y_hat, evaluate.metrics(y_va, y_hat, valid[fs.TARGET_VALUE], value_hat)

    with mlflow.start_run(run_name="optuna-lightgbm") as parent:
        mlflow.set_tags({"stage": "4-tuning", "git_commit": _git_commit()})
        default_pred, default_scores = score(None)
        print(f"default params: rmse={default_scores['rmse']:.4f}")

        def objective(trial: optuna.Trial) -> float:
            params = search_space(trial)
            _, scores = score(params)
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics({f"val_{k}": v for k, v in scores.items()})
            print(f"trial {trial.number:3d} rmse={scores['rmse']:.4f}")
            return scores["rmse"]

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=pipelines.SEED),
            study_name="lightgbm-main",
        )
        study.enqueue_trial(
            pipelines.LGBM_DEFAULTS | {"reg_lambda": 1e-3, "reg_alpha": 1e-3, "cat_smooth": 10.0}
        )
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        tuned_pred, tuned_scores = score(best)
        gap = evaluate.paired_bootstrap_rmse_diff(y_va, tuned_pred, default_pred)
        mlflow.log_params({f"best_{k}": v for k, v in best.items()} | {"n_trials": n_trials})
        mlflow.log_metrics(
            {f"best_val_{k}": v for k, v in tuned_scores.items()}
            | {f"default_val_{k}": v for k, v in default_scores.items()}
            | {f"gap_vs_default_{k}": v for k, v in gap.items()}
        )
        trials = study.trials_dataframe()
        trials.to_csv(REPORTS_DIR / "optuna_trials.csv", index=False)
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        BEST_PARAMS_PATH.write_text(
            json.dumps(
                {
                    "params": best,
                    "val_rmse": tuned_scores["rmse"],
                    "default_val_rmse": default_scores["rmse"],
                    "n_trials": n_trials,
                    "features": features,
                    "mlflow_run_id": parent.info.run_id,
                },
                indent=2,
            )
        )
        mlflow.log_artifact(str(BEST_PARAMS_PATH))

    print(
        f"\nbest rmse={tuned_scores['rmse']:.4f} vs default {default_scores['rmse']:.4f}: "
        f"gap {gap['diff']:+.4f} (95% CI {gap['ci_low']:+.4f} to {gap['ci_high']:+.4f})"
    )
    print(json.dumps(best, indent=1))
    pd.DataFrame(
        [{"setting": "default", **default_scores}, {"setting": "tuned", **tuned_scores}]
    ).to_csv(REPORTS_DIR / "tuning_summary.csv", index=False)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trials", type=int, default=60)
    tune(parser.parse_args().trials)


if __name__ == "__main__":
    main()
