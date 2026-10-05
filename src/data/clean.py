"""Cleaning rules, as pure functions plus the DuckDB views built from them.

The ``raw_*`` views come from :mod:`src.data.load`; this module layers ``clean_*`` tables
on top as materialized tables. Feature code only ever reads ``clean_*``.
"""

from __future__ import annotations

import math

import duckdb
import pandas as pd

# Canonical sub-positions (as in players.sub_position) and their broad group.
SUB_POSITION_GROUP = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Defender", "Left-Back": "Defender", "Right-Back": "Defender",
    "Defensive Midfield": "Midfield", "Central Midfield": "Midfield",
    "Attacking Midfield": "Midfield", "Left Midfield": "Midfield", "Right Midfield": "Midfield",
    "Left Winger": "Attack", "Right Winger": "Attack", "Centre-Forward": "Attack",
    "Second Striker": "Attack",
}

# Spellings seen in game_lineups.position that are not canonical sub-positions.
POSITION_ALIASES = {
    "sweeper": "Centre-Back",
    "attack": "Attack", "defender": "Defender", "midfield": "Midfield",
    "goalkeeper": "Goalkeeper",
}

VALID_FEET = {"left", "right", "both"}
HEIGHT_RANGE_CM = (150, 210)
MAX_MINUTES_PER_GAME = 130  # 120 + stoppage time


def normalize_position(raw: str | None) -> str | None:
    """Map a raw lineup/player position to a canonical sub-position or broad group.

    Returns None for blanks. Broad-only labels ("Attack", "midfield") map to the group
    name, which :func:`position_group` passes through unchanged.
    """
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "missing":
        return None
    for canonical in SUB_POSITION_GROUP:
        if text.lower() == canonical.lower():
            return canonical
    return POSITION_ALIASES.get(text.lower())


def position_group(position: str | None) -> str | None:
    """Broad group (Goalkeeper/Defender/Midfield/Attack) for a normalized position."""
    if position is None:
        return None
    if position in SUB_POSITION_GROUP.values():
        return position
    return SUB_POSITION_GROUP.get(position)


def normalize_foot(raw: str | None) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "unknown"
    text = str(raw).strip().lower()
    return text if text in VALID_FEET else "unknown"


def clean_height(cm: float | None) -> float | None:
    """Heights outside a plausible adult range are treated as missing."""
    if cm is None or (isinstance(cm, float) and math.isnan(cm)):
        return None
    lo, hi = HEIGHT_RANGE_CM
    return float(cm) if lo <= cm <= hi else None


def transfer_season_label(season: int) -> str:
    """2023 -> '23/24', the format used by transfers.transfer_season."""
    return f"{season % 100:02d}/{(season + 1) % 100:02d}"


def season_label(season: int) -> str:
    """2023 -> '2023/24' for display."""
    return f"{season}/{(season + 1) % 100:02d}"


def _position_map_frame() -> pd.DataFrame:
    rows = [(k, k, v) for k, v in SUB_POSITION_GROUP.items()]
    rows += [(alias, normalize_position(alias), position_group(normalize_position(alias)))
             for alias in POSITION_ALIASES]
    df = pd.DataFrame(rows, columns=["raw_lower", "position", "position_group"])
    df["raw_lower"] = df["raw_lower"].str.lower()
    return df.drop_duplicates("raw_lower")


def register_clean_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Materialize ``clean_*`` tables from ``raw_*``; mirrors the Python rules above."""
    con.register("position_map_df", _position_map_frame())
    con.execute("CREATE OR REPLACE TABLE position_map AS SELECT * FROM position_map_df")
    con.unregister("position_map_df")
    lo, hi = HEIGHT_RANGE_CM

    con.execute(f"""
        CREATE OR REPLACE TABLE clean_players AS
        SELECT player_id, name,
               CASE WHEN year(date_of_birth) BETWEEN 1950 AND 2015
                    THEN CAST(date_of_birth AS DATE) END AS date_of_birth,
               CASE WHEN lower(trim(foot)) IN ('left', 'right', 'both')
                    THEN lower(trim(foot)) ELSE 'unknown' END AS foot,
               CASE WHEN height_in_cm BETWEEN {lo} AND {hi} THEN height_in_cm END AS height_cm,
               nullif(trim(country_of_citizenship), '') AS citizenship,
               pm.position AS profile_position,
               pm.position_group AS profile_position_group,
               CAST(contract_expiration_date AS DATE) AS contract_expiration_date,
               image_url
        FROM raw_players p
        LEFT JOIN position_map pm
          ON pm.raw_lower = lower(trim(coalesce(p.sub_position, p.position)))
    """)

    # One row per (game, player); minutes clipped to a physically possible range.
    con.execute(f"""
        CREATE OR REPLACE TABLE clean_appearances AS
        SELECT game_id, player_id, player_club_id AS club_id, date, competition_id,
               greatest(coalesce(yellow_cards, 0), 0) AS yellow_cards,
               greatest(coalesce(red_cards, 0), 0) AS red_cards,
               greatest(coalesce(goals, 0), 0) AS goals,
               greatest(coalesce(assists, 0), 0) AS assists,
               least(greatest(coalesce(minutes_played, 0), 0), {MAX_MINUTES_PER_GAME}) AS minutes
        FROM raw_appearances
        QUALIFY row_number() OVER (PARTITION BY game_id, player_id ORDER BY minutes_played DESC) = 1
    """)

    con.execute("""
        CREATE OR REPLACE TABLE clean_lineups AS
        SELECT l.game_id, l.player_id, l.club_id, l.date,
               l.type = 'starting_lineup' AS is_starter,
               coalesce(l.team_captain, 0) = 1 AS is_captain,
               pm.position, pm.position_group
        FROM raw_game_lineups l
        LEFT JOIN position_map pm ON pm.raw_lower = lower(trim(l.position))
        QUALIFY row_number() OVER (PARTITION BY l.game_id, l.player_id
                                   ORDER BY l.type = 'starting_lineup' DESC) = 1
    """)

    con.execute("""
        CREATE OR REPLACE TABLE clean_valuations AS
        SELECT player_id, date, market_value_in_eur AS value,
               player_club_domestic_competition_id AS league
        FROM raw_player_valuations
        WHERE market_value_in_eur > 0
        QUALIFY row_number() OVER (PARTITION BY player_id, date ORDER BY market_value_in_eur DESC) = 1
    """)

    # transfer_fee: 0 = free transfer, NULL = undisclosed (kept distinct on purpose).
    con.execute("""
        CREATE OR REPLACE TABLE clean_transfers AS
        SELECT player_id, transfer_date, transfer_season, from_club_id, to_club_id,
               CASE WHEN transfer_fee >= 0 THEN transfer_fee END AS fee
        FROM raw_transfers
        WHERE from_club_id IS DISTINCT FROM to_club_id
    """)

    con.execute("CREATE OR REPLACE TABLE clean_games AS SELECT * FROM raw_games")
    con.execute("CREATE OR REPLACE TABLE clean_clubs AS SELECT * FROM raw_clubs")
    con.execute("CREATE OR REPLACE TABLE clean_competitions AS SELECT * FROM raw_competitions")
