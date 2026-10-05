"""Which panel columns each model may use.

The panel also carries identifiers, display fields, the target, backtest outcomes and
audit columns. Those must never reach a model; ``tests/test_leakage.py`` enforces it.
"""

ID_COLS = ["player_id", "season"]

TARGET = "y"  # ln(target_value) - market_index (season-start global index)
TARGET_VALUE = "target_value"  # euros, for reporting on the original scale
INDEX_COL = "market_index"  # add back to predictions: value = exp(pred + market_index)

# Every string-valued panel column a model could use (pipelines one-hot / native-encode these).
CATEGORICAL_COLUMNS = ("league", "position", "position_group", "foot", "citizenship")

# citizenship itself is not a feature (stage-4 decision): the model would learn the market's
# nationality premium. It is replaced by is_domestic and is_eu_eea (point-in-time, Brexit).
CATEGORICAL = ["league", "position", "position_group", "foot"]

NUMERIC = [
    # profile
    "age",
    "height_cm",
    "is_domestic",
    "is_eu_eea",
    # season (all competitions, up to the league's final matchday)
    "apps",
    "starts",
    "minutes",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards",
    "goals_per90",
    "assists_per90",
    "ga_per90",
    "cards_per90",
    "league_minutes",
    "league_minutes_share",
    "start_share",
    "captain_games",
    "euro_minutes",
    "cup_minutes",
    "n_clubs_season",
    "changed_club",
    "changed_league",
    # trend vs previous season, and the last three seasons
    "prev_observed",
    "prev_minutes",
    "prev_apps",
    "prev_ga_per90",
    "minutes_delta",
    "ga_per90_delta",
    "l3_seasons",
    "l3_minutes",
    "l3_goals",
    "l3_assists",
    # club, that season
    "club_position",
    "club_position_pct",
    "club_ppg",
    "club_gd_per_game",
    "club_in_europe",
    "club_players_used",
    "club_avg_age",
    "club_foreign_share",
    "club_net_spend_rel",
    "club_fees_in_rel",
    "club_fees_out_rel",
    "club_n_transfers",
    "club_fee_disclosed_share",
    # league strength relative to the whole market
    "league_index_rel",
]

# Valuation history: only for the "ceiling" comparison model, never the main model.
VALUE_HISTORY = [
    "val_last_rel",
    "val_season_start_rel",
    "val_peak_rel",
    "val_12m_rel",
    "val_trend_12m",
    "days_since_val_last",
]

MAIN_FEATURES = NUMERIC + CATEGORICAL
CEILING_FEATURES = MAIN_FEATURES + VALUE_HISTORY

# Known at flag time but used only for backtest baselines / evaluation.
BACKTEST_COLS = [
    "bt_last_update_log_change",
    "index_at_target",
    "index_at_future",
    "outcome_future_value",
    "outcome_future_date",
    "outcome_log_change",
    "outcome_rel_change",
]

DISPLAY_COLS = [
    "name",
    "club_id",
    "club_name",
    "league_name",
    "season_label",
    "image_url",
    "contract_expiration_date",
    "end_date",
    "target_date",
]

# Substrings that must never appear in a model's feature list (snapshot or leaky fields).
FORBIDDEN_SUBSTRINGS = [
    "contract",
    "caps",
    "national_team",
    "highest",
    "market_value",
    "outcome_",
    "bt_",
    "target",
    "_date",
    "player_id",
]


def feature_columns(kind: str = "main") -> list[str]:
    if kind == "main":
        return list(MAIN_FEATURES)
    if kind == "ceiling":
        return list(CEILING_FEATURES)
    raise ValueError(f"Unknown feature set: {kind!r}")
