"""Model pipelines. Each is a single sklearn Pipeline (preprocessing + model) so it can be
saved and served as one artifact. All take the raw panel columns as input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features.feature_sets import CATEGORICAL

SEED = 42
MIN_CATEGORY_COUNT = 20  # rarer levels (mostly citizenships) are pooled into "other"


def split_columns(features: list[str]) -> tuple[list[str], list[str]]:
    cats = [c for c in features if c in CATEGORICAL]
    nums = [c for c in features if c not in CATEGORICAL]
    return nums, cats


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep exactly ``columns`` (in order), so the pipeline accepts the whole panel row."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Cast categoricals to pandas ``category`` with levels frozen at fit time.

    Levels seen fewer than ``min_count`` times become "other"; unseen levels at predict
    time also map to "other" (or NaN if "other" was never needed in training). This
    gives LightGBM / XGBoost native categorical handling with a stable encoding.
    """

    def __init__(self, columns: list[str], min_count: int = MIN_CATEGORY_COUNT):
        self.columns = columns
        self.min_count = min_count

    def fit(self, X, y=None):
        self.levels_ = {}
        for col in self.columns:
            counts = X[col].value_counts()
            keep = sorted(counts[counts >= self.min_count].index.astype(str))
            self.levels_[col] = keep + (["other"] if len(keep) < len(counts) else [])
        return self

    def transform(self, X):
        X = X.copy()
        for col, levels in self.levels_.items():
            s = X[col].astype("object")
            known = s.isin(levels) | s.isna()
            if "other" in levels:
                s = s.where(known, "other")
            X[col] = pd.Categorical(s, categories=levels)
        return X


class LastValueBaseline(BaseEstimator, RegressorMixin):
    """Naive ceiling baseline: predict the player's last known valuation (relative to the
    market index). Players with no prior valuation get the training mean."""

    column = "val_last_rel"

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X[self.column].fillna(self.fallback_).to_numpy(dtype=float)


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict the training mean: the floor every model must beat."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def _linear(nums, cats) -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                nums,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=MIN_CATEGORY_COUNT,
                        sparse_output=False,
                    ),
                ),
                cats,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])


def _random_forest(nums, cats) -> Pipeline:
    # sklearn forests handle NaN natively; categoricals are ordinal-encoded.
    pre = ColumnTransformer(
        [
            ("num", "passthrough", nums),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    min_frequency=MIN_CATEGORY_COUNT,
                ),
                cats,
            ),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=3, max_features=0.33, n_jobs=-1, random_state=SEED
    )
    return Pipeline([("pre", pre), ("model", model)])


def _xgboost(nums, cats) -> Pipeline:
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        n_jobs=-1,
        random_state=SEED,
    )
    return Pipeline(
        [("select", ColumnSelector(nums + cats)), ("cats", CategoryCaster(cats)), ("model", model)]
    )


def _lightgbm(nums, cats) -> Pipeline:
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline(
        [("select", ColumnSelector(nums + cats)), ("cats", CategoryCaster(cats)), ("model", model)]
    )


BUILDERS = {
    "mean_baseline": lambda nums, cats: MeanBaseline(),
    "linear": _linear,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def build(name: str, features: list[str]):
    if name == "last_value_baseline":
        return LastValueBaseline()
    nums, cats = split_columns(features)
    return BUILDERS[name](nums, cats)
