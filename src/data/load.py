"""Expose the raw Transfermarkt CSVs as typed DuckDB views.

Only the columns the pipeline needs are read, so test fixtures can be minimal. Every
date-stamped table can be truncated with ``as_of`` to simulate the data as it existed
on that day; the leakage tests rely on this.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# table -> (columns with DuckDB types, date column used for as_of truncation or None)
SCHEMAS: dict[str, tuple[dict[str, str], str | None]] = {
    "players": (
        {
            "player_id": "BIGINT",
            "name": "VARCHAR",
            "date_of_birth": "TIMESTAMP",
            "position": "VARCHAR",
            "sub_position": "VARCHAR",
            "foot": "VARCHAR",
            "height_in_cm": "DOUBLE",
            "country_of_citizenship": "VARCHAR",
            "contract_expiration_date": "TIMESTAMP",
            "image_url": "VARCHAR",
        },
        None,
    ),
    "clubs": ({"club_id": "BIGINT", "name": "VARCHAR", "domestic_competition_id": "VARCHAR"}, None),
    "competitions": (
        {
            "competition_id": "VARCHAR",
            "name": "VARCHAR",
            "type": "VARCHAR",
            "country_name": "VARCHAR",
        },
        None,
    ),
    "games": (
        {
            "game_id": "BIGINT",
            "competition_id": "VARCHAR",
            "season": "BIGINT",
            "date": "DATE",
            "home_club_id": "BIGINT",
            "away_club_id": "BIGINT",
            "home_club_goals": "BIGINT",
            "away_club_goals": "BIGINT",
            "home_club_position": "BIGINT",
            "away_club_position": "BIGINT",
            "competition_type": "VARCHAR",
        },
        "date",
    ),
    "appearances": (
        {
            "game_id": "BIGINT",
            "player_id": "BIGINT",
            "player_club_id": "BIGINT",
            "date": "DATE",
            "competition_id": "VARCHAR",
            "yellow_cards": "BIGINT",
            "red_cards": "BIGINT",
            "goals": "BIGINT",
            "assists": "BIGINT",
            "minutes_played": "BIGINT",
        },
        "date",
    ),
    "game_lineups": (
        {
            "game_id": "BIGINT",
            "player_id": "BIGINT",
            "club_id": "BIGINT",
            "date": "DATE",
            "type": "VARCHAR",
            "position": "VARCHAR",
            "team_captain": "BIGINT",
        },
        "date",
    ),
    "player_valuations": (
        {
            "player_id": "BIGINT",
            "date": "DATE",
            "market_value_in_eur": "BIGINT",
            "player_club_domestic_competition_id": "VARCHAR",
        },
        "date",
    ),
    "transfers": (
        {
            "player_id": "BIGINT",
            "transfer_date": "DATE",
            "transfer_season": "VARCHAR",
            "from_club_id": "BIGINT",
            "to_club_id": "BIGINT",
            "transfer_fee": "DOUBLE",
        },
        "transfer_date",
    ),
}


def connect(
    raw_dir: Path, as_of: str | None = None, con: duckdb.DuckDBPyConnection | None = None
) -> duckdb.DuckDBPyConnection:
    """Register one ``raw_<table>`` view per CSV in ``raw_dir``.

    ``as_of`` (ISO date) drops every dated row after that day: the dataset as it would
    have looked then. Undated tables (players, clubs, competitions) are static attributes.
    """
    con = con or duckdb.connect()
    for table, (columns, date_col) in SCHEMAS.items():
        path = Path(raw_dir) / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        types = ", ".join(f"'{c}': '{t}'" for c, t in columns.items())
        select = ", ".join(f'"{c}"' for c in columns)
        where = f"WHERE {date_col} <= DATE '{as_of}'" if as_of and date_col else ""
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_{table} AS
            SELECT {select}
            FROM read_csv_auto('{path.as_posix()}', header=true, types={{{types}}})
            {where}
        """)
    return con
