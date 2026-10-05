"""Validation for the player-season panel: schema, ranges and point-in-time audits.

``validate_panel`` returns a list of human-readable failures (empty = valid);
``assert_valid`` raises if there are any. The build script runs it before writing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.feature_sets import CATEGORICAL, CEILING_FEATURES, TARGET

REQUIRED_NON_NULL = [
    "player_id",
    "season",
    "league",
    "club_id",
    "end_date",
    "target_date",
    "target_value",
    "age",
    "minutes",
    "apps",
    "market_index",
    TARGET,
    "position_group",
]

# column -> (min, max), inclusive; NaN allowed unless the column is in REQUIRED_NON_NULL
RANGES = {
    "age": (15, 45),
    "height_cm": (150, 210),
    "minutes": (config.MIN_MINUTES, None),
    "target_value": (1, None),
    "league_minutes_share": (0, 1),
    "start_share": (0, 1),
    "club_foreign_share": (0, 1),
    "club_fee_disclosed_share": (0, 1),
    "club_position_pct": (0, 1),
    "goals_per90": (0, 10),
    "assists_per90": (0, 10),
    "club_ppg": (0, 3),
}


def _dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s)


def validate_panel(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if df.empty:
        return ["panel is empty"]

    missing = [c for c in CEILING_FEATURES + REQUIRED_NON_NULL if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors

    dupes = df.duplicated(["player_id", "season"]).sum()
    if dupes:
        errors.append(f"{dupes} duplicate (player_id, season) rows")

    for col in REQUIRED_NON_NULL:
        n = df[col].isna().sum()
        if n:
            errors.append(f"{col}: {n} nulls")

    for col, (lo, hi) in RANGES.items():
        s = df[col].dropna()
        if lo is not None and (s < lo).any():
            errors.append(f"{col}: {(s < lo).sum()} values below {lo}")
        if hi is not None and (s > hi).any():
            errors.append(f"{col}: {(s > hi).sum()} values above {hi}")

    if (df["minutes"] > df["apps"] * 130).any():
        errors.append("minutes exceed 130 per appearance")

    numeric = df[[c for c in CEILING_FEATURES if c not in CATEGORICAL]]
    inf = np.isinf(numeric.to_numpy(dtype=float)).sum()
    if inf:
        errors.append(f"{inf} infinite feature values")

    expected_y = np.log(df["target_value"]) - df["market_index"]
    if not np.allclose(df[TARGET], expected_y):
        errors.append("y != ln(target_value) - market_index")

    errors += point_in_time_errors(df)
    return errors


def point_in_time_errors(df: pd.DataFrame) -> list[str]:
    """Every source date behind a row's features must precede that row's cut-offs."""
    errors = []
    end = _dates(df["end_date"])
    target = _dates(df["target_date"])

    checks = {
        "_max_app_date <= end_date": _dates(df["_max_app_date"]) <= end,
        "_max_lineup_date <= end_date": _dates(df["_max_lineup_date"]) <= end,
        "_max_transfer_date <= end_date": _dates(df["_max_transfer_date"]) <= end,
        "_val_hist_date <= end_date": _dates(df["_val_hist_date"]) <= end,
        "_val_hist_date < target_date": _dates(df["_val_hist_date"]) < target,
        "_index_asof_date <= end_date": _dates(df["_index_asof_date"]) <= end,
    }
    for name, ok in checks.items():
        # NaT comparisons are False; a missing source date is not a violation
        source = name.split(" ")[0]
        bad = (~ok & df[source].notna()).sum()
        if bad:
            errors.append(f"point-in-time violation: {bad} rows fail {name}")

    gap = (target - end).dt.days.abs()
    if (gap > config.TARGET_WINDOW_DAYS).any():
        errors.append("target valuation outside the season-end window")
    return errors


def assert_valid(df: pd.DataFrame) -> None:
    errors = validate_panel(df)
    if errors:
        raise ValueError("Panel validation failed:\n  - " + "\n  - ".join(errors))
