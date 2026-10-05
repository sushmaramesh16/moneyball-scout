"""A tiny hand-checkable Transfermarkt-shaped dataset for pipeline tests.

Two leagues (GB1: clubs 1, 2; ES1: clubs 3, 4), seasons 2013-2015. Each league-season has
six league games, the last on (S+1)-05-20 (= season end). Club 1 also plays one Champions
League game (S-10-01, vs club 3) and one FA Cup game ((S+1)-01-10, vs club 2).

Players
  101  club 1, England, b. 1995-06-01, CF, plays every club-1 game for 90' (8 apps, 720'),
       scores in league games 0, 2, 4 and assists in league game 1; captain in league games
  102  club 2, Brazil, b. 2000-01-01, CB, 80' in every club-2 league game (480')
  103  club 3, Spain, b. 1990-03-03, lineup position "midfield", 90' per league game (540')
  104  club 4, Spain, plays two league games (180') -> below the minutes threshold
Valuations: each player on S-12-15 and (S+1)-06-10 (the target, 21 days after season end).
Transfers per season into/out of club 1: one 3M signing on S-07-15 and one undisclosed sale.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

SEASONS = (2013, 2014, 2015)
LEAGUE_CLUBS = {"GB1": (1, 2), "ES1": (3, 4)}
LEAGUE_DAYS = [(0, 8, 15), (0, 9, 15), (0, 10, 15), (1, 2, 15), (1, 4, 15), (1, 5, 20)]

PLAYERS = [
    dict(
        player_id=101,
        name="Ada Forward",
        date_of_birth="1995-06-01",
        position="Attack",
        sub_position="Centre-Forward",
        foot="Right",
        height_in_cm=182,
        country_of_citizenship="England",
        contract_expiration_date="2030-06-30",
    ),
    dict(
        player_id=102,
        name="Bo Back",
        date_of_birth="2000-01-01",
        position="Defender",
        sub_position="Centre-Back",
        foot="left",
        height_in_cm=190,
        country_of_citizenship="Brazil",
        contract_expiration_date=None,
    ),
    dict(
        player_id=103,
        name="Cy Mid",
        date_of_birth="1990-03-03",
        position="Midfield",
        sub_position=None,
        foot=None,
        height_in_cm=None,
        country_of_citizenship="Spain",
        contract_expiration_date=None,
    ),
    dict(
        player_id=104,
        name="Di Keeper",
        date_of_birth="1998-01-01",
        position="Goalkeeper",
        sub_position="Goalkeeper",
        foot="right",
        height_in_cm=195,
        country_of_citizenship="Spain",
        contract_expiration_date=None,
    ),
]
LINEUP_POSITION = {101: "Centre-Forward", 102: "Centre-Back", 103: "midfield", 104: "Goalkeeper"}
VALUES = {
    101: (10_000_000, 12_000_000),
    102: (1_000_000, 2_000_000),
    103: (5_000_000, 4_000_000),
    104: (500_000, 500_000),
}
PLAYER_LEAGUE = {101: "GB1", 102: "GB1", 103: "ES1", 104: "ES1"}


def season_end(season: int) -> date:
    return date(season + 1, 5, 20)


def _tables() -> dict[str, list[dict]]:
    games, apps, lineups, vals, transfers = [], [], [], [], []
    gid = 0

    def add_game(comp, ctype, season, d, home, away, hg, ag, hp=None, ap=None):
        nonlocal gid
        gid += 1
        games.append(
            dict(
                game_id=gid,
                competition_id=comp,
                season=season,
                date=d,
                home_club_id=home,
                away_club_id=away,
                home_club_goals=hg,
                away_club_goals=ag,
                home_club_position=hp,
                away_club_position=ap,
                competition_type=ctype,
            )
        )
        return gid

    def play(g, d, comp, pid, club, minutes, goals=0, assists=0, captain=0):
        apps.append(
            dict(
                game_id=g,
                player_id=pid,
                player_club_id=club,
                date=d,
                competition_id=comp,
                yellow_cards=0,
                red_cards=0,
                goals=goals,
                assists=assists,
                minutes_played=minutes,
            )
        )
        lineups.append(
            dict(
                game_id=g,
                player_id=pid,
                club_id=club,
                date=d,
                type="starting_lineup",
                position=LINEUP_POSITION[pid],
                team_captain=captain,
            )
        )

    for s in SEASONS:
        for league, (a, b) in LEAGUE_CLUBS.items():
            for i, (dy, m, dd) in enumerate(LEAGUE_DAYS):
                d = date(s + dy, m, dd)
                home, away = (a, b) if i % 2 == 0 else (b, a)
                hg, ag = (2, 1) if i % 2 == 0 else (0, 0)
                hp, ap = (1, 2) if home == a else (2, 1)
                g = add_game(league, "domestic_league", s, d, home, away, hg, ag, hp, ap)
                if league == "GB1":
                    play(
                        g,
                        d,
                        league,
                        101,
                        1,
                        90,
                        goals=int(i % 2 == 0),
                        assists=int(i == 1),
                        captain=1,
                    )
                    play(g, d, league, 102, 2, 80)
                else:
                    play(g, d, league, 103, 3, 90)
                    if i < 2:
                        play(g, d, league, 104, 4, 90)
        d = date(s, 10, 1)
        g = add_game("CL", "international_cup", s, d, 1, 3, 1, 1)
        play(g, d, "CL", 101, 1, 90)
        d = date(s + 1, 1, 10)
        g = add_game("FAC", "domestic_cup", s, d, 1, 2, 1, 0)
        play(g, d, "FAC", 101, 1, 90)

        for pid, (dec, jun) in VALUES.items():
            vals.append(
                dict(
                    player_id=pid,
                    date=date(s, 12, 15),
                    market_value_in_eur=dec,
                    player_club_domestic_competition_id=PLAYER_LEAGUE[pid],
                )
            )
            vals.append(
                dict(
                    player_id=pid,
                    date=date(s + 1, 6, 10),
                    market_value_in_eur=jun,
                    player_club_domestic_competition_id=PLAYER_LEAGUE[pid],
                )
            )

        label = f"{s % 100:02d}/{(s + 1) % 100:02d}"
        transfers.append(
            dict(
                player_id=900 + s,
                transfer_date=date(s, 7, 15),
                transfer_season=label,
                from_club_id=50,
                to_club_id=1,
                transfer_fee=3_000_000.0,
            )
        )
        transfers.append(
            dict(
                player_id=950 + s,
                transfer_date=date(s, 8, 1),
                transfer_season=label,
                from_club_id=1,
                to_club_id=60,
                transfer_fee=None,
            )
        )

    clubs = [
        dict(club_id=c, name=f"Club {c}", domestic_competition_id=lg)
        for lg, cs in LEAGUE_CLUBS.items()
        for c in cs
    ]
    competitions = [
        dict(
            competition_id="GB1",
            name="premier-league",
            type="domestic_league",
            country_name="England",
        ),
        dict(competition_id="ES1", name="laliga", type="domestic_league", country_name="Spain"),
        dict(
            competition_id="CL",
            name="uefa-champions-league",
            type="international_cup",
            country_name=None,
        ),
        dict(competition_id="FAC", name="fa-cup", type="domestic_cup", country_name="England"),
    ]
    return dict(
        players=PLAYERS,
        clubs=clubs,
        competitions=competitions,
        games=games,
        appearances=apps,
        game_lineups=lineups,
        player_valuations=vals,
        transfers=transfers,
    )


def write_raw(directory: Path, extra: dict[str, list[dict]] | None = None) -> Path:
    """Write the fixture CSVs (plus any ``extra`` rows per table) into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    for name, rows in (extra or {}).items():
        tables[name] = tables[name] + rows
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(directory / f"{name}.csv", index=False)
    return directory
