"""SHAP explanations for the LightGBM pipeline.

Values come from LightGBM's own TreeSHAP (``pred_contrib=True``): exact, fast, and it
handles native categorical splits. Contributions are on the model's scale (log value
relative to the market index), so exp(contribution) is a multiplicative effect on price:
+0.30 means "this pushes the predicted value up by about 35%".
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

BASE = "base_value"

FEATURE_LABELS = {
    "age": "Age",
    "height_cm": "Height",
    "is_domestic": "Plays in home country",
    "is_eu_eea": "EU/EEA citizen",
    "apps": "Appearances",
    "starts": "Starts",
    "minutes": "Minutes played",
    "goals": "Goals",
    "assists": "Assists",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90",
    "ga_per90": "Goals + assists per 90",
    "cards_per90": "Cards per 90",
    "league_minutes": "League minutes",
    "league_minutes_share": "Share of club's league minutes",
    "start_share": "Share of games started",
    "captain_games": "Games as captain",
    "euro_minutes": "European minutes",
    "cup_minutes": "Domestic cup minutes",
    "n_clubs_season": "Clubs this season",
    "changed_club": "Changed club",
    "changed_league": "Changed league",
    "prev_observed": "Played in top-14 leagues last season",
    "prev_minutes": "Minutes last season",
    "prev_apps": "Appearances last season",
    "prev_ga_per90": "Goals + assists per 90 last season",
    "minutes_delta": "Change in minutes vs last season",
    "ga_per90_delta": "Change in goals + assists per 90",
    "l3_seasons": "Seasons in data (last 3)",
    "l3_minutes": "Minutes over last 3 seasons",
    "l3_goals": "Goals over last 3 seasons",
    "l3_assists": "Assists over last 3 seasons",
    "club_position": "Club league position",
    "club_position_pct": "Club league position (relative)",
    "club_ppg": "Club points per game",
    "club_gd_per_game": "Club goal difference per game",
    "club_in_europe": "Club in Europe",
    "club_players_used": "Club squad size used",
    "club_avg_age": "Club average age",
    "club_foreign_share": "Club share of foreign minutes",
    "club_net_spend_rel": "Club net transfer spend",
    "club_fees_in_rel": "Club transfer spending",
    "club_fees_out_rel": "Club transfer income",
    "club_n_transfers": "Club transfer activity",
    "club_fee_disclosed_share": "Club fees disclosed",
    "league_index_rel": "League price level",
    "league": "League",
    "position": "Position",
    "position_group": "Position group",
    "foot": "Preferred foot",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def shap_frame(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """One SHAP column per model feature plus ``base_value``; each row sums to the
    model's prediction for that row."""
    Xt = model[:-1].transform(X)
    contrib = model[-1].predict(Xt, pred_contrib=True)
    return pd.DataFrame(contrib, columns=[*Xt.columns, BASE], index=X.index)


def global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    vals = shap_df.drop(columns=BASE)
    out = pd.DataFrame(
        {
            "feature": vals.columns,
            "mean_abs_shap": vals.abs().mean().to_numpy(),
            "mean_shap": vals.mean().to_numpy(),
        }
    )
    out["label"] = out["feature"].map(label)
    out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def _display_value(v):
    if isinstance(v, float | np.floating):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, np.integer):
        return int(v)
    return None if pd.isna(v) else str(v)


def top_factors(shap_df: pd.DataFrame, X: pd.DataFrame, k: int = 5) -> list[str]:
    """Per row: JSON list of the k largest |SHAP| drivers with the feature's value."""
    vals = shap_df.drop(columns=BASE)
    out = []
    for idx, row in vals.iterrows():
        top = row.abs().nlargest(k).index
        out.append(
            json.dumps(
                [
                    {
                        "feature": f,
                        "label": label(f),
                        "value": _display_value(X.at[idx, f]),
                        "shap": round(float(row[f]), 4),
                        "effect_pct": round(float(np.expm1(row[f]) * 100), 1),
                    }
                    for f in top
                ]
            )
        )
    return out
