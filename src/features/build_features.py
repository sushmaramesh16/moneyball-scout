"""Build the point-in-time player-season panel.

One row = one player in one season (2015/16 ... latest). Every feature is computed from
data dated on or before that season's final league matchday ("season end"); the target is
the valuation nearest season end. See ``src/features/feature_sets.py`` for which columns
each model may use.

Usage:
    python -m src.features.build_features            # -> data/processed/panel.parquet
    python -m src.features.build_features --as-of 2024-06-30 --out /tmp/panel_asof.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd

from src import config
from src.data import clean, load, validate
from src.features import feature_sets


def _sql_list(values) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


LEAGUES = _sql_list(config.LEAGUES)
EURO = _sql_list(config.EURO_COMPS)
TRANSFER_SEASON = (
    "lpad((season % 100)::VARCHAR, 2, '0') || '/' || lpad(((season + 1) % 100)::VARCHAR, 2, '0')"
)


def _league_seasons(con: duckdb.DuckDBPyConnection) -> None:
    """Start/end dates per league-season, and whether the season is over.

    A season is complete if the league has already started a later season, if it has at
    least 85% of the previous season's fixtures, or if it has had no games for 30+ days
    and its last one was in Mar-Jul (season end, not a winter break). This keeps seasons
    that were cut short (COVID 2019/20, Ukraine 2021/22) or shrank (Ligue 1 2023/24),
    and drops only seasons still in progress at the data cut-off.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE league_seasons AS
        WITH g AS (
            SELECT competition_id AS league, season, min(date) AS start_date,
                   max(date) AS end_date, count(*) AS n_games,
                   count(DISTINCT home_club_id) AS n_clubs
            FROM clean_games WHERE competition_id IN {LEAGUES}
            GROUP BY ALL
        ),
        data_end AS (SELECT max(date) AS d FROM clean_games)
        SELECT g.*, c.country_name, c.name AS league_name,
               lead(g.season) OVER w IS NOT NULL
               OR coalesce(g.n_games >= 0.85 * lag(g.n_games) OVER w, false)
               OR (g.end_date < (SELECT d FROM data_end) - INTERVAL 30 DAY
                   AND month(g.end_date) BETWEEN 3 AND 7) AS complete
        FROM g LEFT JOIN clean_competitions c ON c.competition_id = g.league
        WINDOW w AS (PARTITION BY g.league ORDER BY g.season)
    """)


def _appearances(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE app AS
        SELECT a.*, g.season,
               a.competition_id IN {LEAGUES} AS is_league,
               a.competition_id IN {EURO} AS is_euro,
               coalesce(g.competition_type = 'domestic_cup', false) AS is_cup
        FROM clean_appearances a JOIN clean_games g USING (game_id)
    """)


def _player_seasons(con: duckdb.DuckDBPyConnection) -> None:
    """Main league + club per player-season (most league minutes), then season stats."""
    con.execute("""
        CREATE OR REPLACE TABLE ps_main AS
        WITH lm AS (
            SELECT player_id, season, competition_id AS league, club_id,
                   sum(minutes) AS m, count(*) AS n
            FROM app WHERE is_league GROUP BY ALL
        )
        SELECT lm.player_id, lm.season, lm.league, lm.club_id,
               ls.start_date, ls.end_date, ls.complete, ls.country_name, ls.league_name,
               ls.n_clubs
        FROM lm JOIN league_seasons ls ON ls.league = lm.league AND ls.season = lm.season
        QUALIFY row_number() OVER (PARTITION BY lm.player_id, lm.season
                                   ORDER BY lm.m DESC, lm.n DESC, lm.club_id) = 1
    """)

    # All competitions with season = S, but never after the main league's final matchday.
    con.execute("""
        CREATE OR REPLACE TABLE ps_stats AS
        SELECT p.player_id, p.season,
               count(*) AS apps,
               sum(a.minutes) AS minutes,
               sum(a.goals) AS goals,
               sum(a.assists) AS assists,
               sum(a.yellow_cards) AS yellow_cards,
               sum(a.red_cards) AS red_cards,
               coalesce(sum(a.minutes) FILTER (WHERE a.is_league), 0) AS league_minutes,
               coalesce(sum(a.minutes) FILTER (WHERE a.is_league AND a.club_id = p.club_id), 0)
                   AS main_club_league_minutes,
               coalesce(sum(a.minutes) FILTER (WHERE a.is_euro), 0) AS euro_minutes,
               coalesce(sum(a.minutes) FILTER (WHERE a.is_cup), 0) AS cup_minutes,
               count(DISTINCT a.club_id) AS n_clubs_season,
               max(a.date) AS _max_app_date
        FROM ps_main p
        JOIN app a ON a.player_id = p.player_id AND a.season = p.season
                  AND a.date <= p.end_date
        GROUP BY ALL
    """)

    con.execute("""
        CREATE OR REPLACE TABLE ps_lineups AS
        WITH l AS (
            -- has_app: lineups also list games with no appearance row (e.g. cup and
            -- qualifier ties); starts/captaincy only count games the player appeared in.
            SELECT p.player_id, p.season, l.is_starter, l.is_captain, l.position, l.date,
                   a.game_id IS NOT NULL AS has_app
            FROM ps_main p
            JOIN clean_lineups l ON l.player_id = p.player_id AND l.date <= p.end_date
            JOIN clean_games g ON g.game_id = l.game_id AND g.season = p.season
            LEFT JOIN app a ON a.game_id = l.game_id AND a.player_id = l.player_id
        ),
        pos AS (
            SELECT player_id, season, position, count(*) AS n
            FROM l WHERE position IS NOT NULL GROUP BY player_id, season, position
            QUALIFY row_number() OVER (
                PARTITION BY player_id, season
                ORDER BY n DESC, position IN ('Attack', 'Defender', 'Midfield'), position) = 1
        )
        SELECT l.player_id, l.season,
               count(*) FILTER (WHERE l.is_starter AND l.has_app) AS starts,
               count(*) FILTER (WHERE l.is_captain AND l.has_app) AS captain_games,
               max(l.date) AS _max_lineup_date,
               any_value(pos.position) AS season_position
        FROM l LEFT JOIN pos USING (player_id, season)
        GROUP BY ALL
    """)

    con.execute("""
        CREATE OR REPLACE TABLE ps_all AS
        SELECT m.*, s.* EXCLUDE (player_id, season),
               coalesce(l.starts, 0) AS starts,
               coalesce(l.captain_games, 0) AS captain_games,
               l.season_position, l._max_lineup_date,
               s.goals + s.assists AS ga
        FROM ps_main m
        JOIN ps_stats s USING (player_id, season)
        LEFT JOIN ps_lineups l USING (player_id, season)
    """)


def _club_seasons(con: duckdb.DuckDBPyConnection) -> None:
    """Club strength, squad make-up and transfer activity for each club-season,
    all bounded by that league-season's final matchday."""
    con.execute(f"""
        CREATE OR REPLACE TABLE club_season AS
        WITH long AS (
            SELECT season, competition_id AS league, date, home_club_id AS club_id,
                   home_club_goals AS gf, away_club_goals AS ga, home_club_position AS pos
            FROM clean_games WHERE competition_id IN {LEAGUES}
            UNION ALL
            SELECT season, competition_id, date, away_club_id,
                   away_club_goals, home_club_goals, away_club_position
            FROM clean_games WHERE competition_id IN {LEAGUES}
        )
        SELECT c.club_id, c.season, c.league,
               count(*) AS club_games,
               avg(CASE WHEN gf > ga THEN 3 WHEN gf = ga THEN 1 ELSE 0 END) AS club_ppg,
               avg(gf - ga) AS club_gd_per_game,
               arg_max(pos, c.date) FILTER (WHERE pos IS NOT NULL) AS club_position
        FROM long c
        JOIN league_seasons ls ON ls.league = c.league AND ls.season = c.season
                              AND c.date <= ls.end_date
        GROUP BY ALL
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE club_europe AS
        SELECT cs.club_id, cs.season, count(*) > 0 AS club_in_europe
        FROM club_season cs
        JOIN league_seasons ls ON ls.league = cs.league AND ls.season = cs.season
        JOIN clean_games g ON g.season = cs.season AND g.competition_id IN {EURO}
                          AND g.date <= ls.end_date
                          AND cs.club_id IN (g.home_club_id, g.away_club_id)
        GROUP BY ALL
    """)

    con.execute("""
        CREATE OR REPLACE TABLE club_squad AS
        SELECT cs.club_id, cs.season,
               count(DISTINCT a.player_id) AS club_players_used,
               sum(a.minutes * date_diff('day', pl.date_of_birth, ls.end_date) / 365.25)
                   / nullif(sum(a.minutes) FILTER (WHERE pl.date_of_birth IS NOT NULL), 0)
                   AS club_avg_age,
               coalesce(sum(a.minutes) FILTER (WHERE pl.citizenship <> ls.country_name), 0)
                   / nullif(sum(a.minutes), 0) AS club_foreign_share
        FROM club_season cs
        JOIN league_seasons ls ON ls.league = cs.league AND ls.season = cs.season
        JOIN app a ON a.club_id = cs.club_id AND a.season = cs.season AND a.is_league
                  AND a.date <= ls.end_date
        LEFT JOIN clean_players pl ON pl.player_id = a.player_id
        GROUP BY ALL
    """)

    # Fees: 0 = free transfer, NULL = undisclosed. Undisclosed fees count as 0 euros but
    # lower club_fee_disclosed_share, so the model can tell secrecy from frugality.
    con.execute(f"""
        CREATE OR REPLACE TABLE club_transfers AS
        WITH moves AS (
            SELECT to_club_id AS club_id, 'in' AS dir, fee, transfer_date, transfer_season
            FROM clean_transfers
            UNION ALL
            SELECT from_club_id, 'out', fee, transfer_date, transfer_season
            FROM clean_transfers
        )
        SELECT cs.club_id, cs.season,
               coalesce(sum(m.fee) FILTER (WHERE m.dir = 'in'), 0) AS club_fees_in,
               coalesce(sum(m.fee) FILTER (WHERE m.dir = 'out'), 0) AS club_fees_out,
               count(*) AS club_n_transfers,
               count(m.fee) / count(*) AS club_fee_disclosed_share,
               max(m.transfer_date) AS _max_transfer_date
        FROM club_season cs
        JOIN league_seasons ls ON ls.league = cs.league AND ls.season = cs.season
        JOIN moves m ON m.club_id = cs.club_id
                    AND m.transfer_season = {TRANSFER_SEASON.replace("season", "cs.season")}
                    AND m.transfer_date <= ls.end_date
        GROUP BY ALL
    """)

    # The player's own fee into/out of his club that season. Subtracted from the club's
    # totals in the final table: a player's own fee is a market price for him (i.e. value
    # history), which the main model must not see.
    con.execute(f"""
        CREATE OR REPLACE TABLE own_fees AS
        SELECT p.player_id, p.season,
               coalesce(sum(t.fee) FILTER (WHERE t.to_club_id = p.club_id), 0) AS own_in,
               coalesce(sum(t.fee) FILTER (WHERE t.from_club_id = p.club_id), 0) AS own_out
        FROM ps_main p
        JOIN clean_transfers t
          ON t.player_id = p.player_id
         AND t.transfer_season = {TRANSFER_SEASON.replace("season", "p.season")}
         AND t.transfer_date <= p.end_date
        GROUP BY ALL
    """)


def _market_index(con: duckdb.DuckDBPyConnection) -> None:
    """Market level at the start of each season (1 July), global and per league.

    Population: players with >= MIN_MINUTES in the 14 leagues the season before (known
    at season start). Level: mean of their latest log valuation in the prior year.
    A fixed population matters: the raw valuation table covers far fewer fringe players
    in recent years, which would masquerade as inflation. The mean (not median) avoids
    jumps from Transfermarkt's round-number values. Defined one season past the last
    played season so backtest outcomes can be indexed.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE market_index AS
        WITH pop AS (
            SELECT player_id, season + 1 AS season, league,
                   make_date((season + 1)::INT, 7, 1) AS asof_date
            FROM ps_all WHERE minutes >= {config.MIN_MINUTES}
        ),
        latest AS (
            SELECT p.season, p.asof_date, p.league, p.player_id,
                   arg_max(v.value, v.date) AS mv
            FROM pop p
            JOIN clean_valuations v
              ON v.player_id = p.player_id AND v.date <= p.asof_date
             AND v.date > p.asof_date - INTERVAL {config.INDEX_LOOKBACK_DAYS} DAY
            WHERE p.asof_date <= (SELECT max(date) FROM clean_valuations) + INTERVAL 31 DAY
            GROUP BY ALL
        )
        SELECT season, asof_date, coalesce(league, '_ALL') AS league,
               avg(ln(mv)) AS idx, count(*) AS n_players
        FROM latest
        GROUP BY GROUPING SETS ((season, asof_date), (season, asof_date, league))
    """)


def _targets_and_history(con: duckdb.DuckDBPyConnection) -> None:
    w = config.TARGET_WINDOW_DAYS
    # Nearest valuation to season end; on equal distance prefer the later one.
    con.execute(f"""
        CREATE OR REPLACE TABLE targets AS
        SELECT p.player_id, p.season,
               arg_min(v.value, abs(date_diff('day', p.end_date, v.date)) * 2
                                - (v.date > p.end_date)::INT) AS target_value,
               arg_min(v.date, abs(date_diff('day', p.end_date, v.date)) * 2
                               - (v.date > p.end_date)::INT) AS target_date
        FROM ps_main p
        JOIN clean_valuations v ON v.player_id = p.player_id
             AND v.date BETWEEN p.end_date - INTERVAL {w} DAY AND p.end_date + INTERVAL {w} DAY
        GROUP BY ALL
    """)

    # Value history (ceiling model only): strictly before the target valuation AND on or
    # before season end.
    con.execute("""
        CREATE OR REPLACE TABLE val_hist AS
        WITH c AS (
            SELECT t.player_id, t.season, p.start_date,
                   least(p.end_date, t.target_date - 1) AS cutoff
            FROM targets t JOIN ps_main p USING (player_id, season)
        )
        SELECT c.player_id, c.season,
               arg_max(v.value, v.date) AS val_last,
               max(v.date) AS _val_hist_date,
               max(v.value) AS val_peak,
               arg_max(v.value, v.date) FILTER (WHERE v.date <= c.start_date) AS val_season_start,
               arg_max(v.value, v.date) FILTER (WHERE v.date <= c.cutoff - 365) AS val_12m,
               any_value(c.cutoff) AS _val_cutoff
        FROM c JOIN clean_valuations v ON v.player_id = c.player_id AND v.date <= c.cutoff
        GROUP BY ALL
    """)

    # Backtest helpers. Known at flag time (= target date): the change at the latest
    # update. Outcome: valuation ~12 months after the target date.
    fw = config.FUTURE_WINDOW_DAYS
    con.execute(f"""
        CREATE OR REPLACE TABLE backtest AS
        WITH prev AS (
            SELECT t.player_id, t.season, arg_max(v.value, v.date) AS prev_value
            FROM targets t JOIN clean_valuations v
              ON v.player_id = t.player_id AND v.date < t.target_date
            GROUP BY ALL
        ),
        fut AS (
            SELECT t.player_id, t.season,
                   arg_min(v.value, abs(date_diff('day', t.target_date + 365, v.date))) AS fv,
                   arg_min(v.date, abs(date_diff('day', t.target_date + 365, v.date))) AS fd
            FROM targets t JOIN clean_valuations v
              ON v.player_id = t.player_id
             AND v.date BETWEEN t.target_date + 365 - {fw} AND t.target_date + 365 + {fw}
            GROUP BY ALL
        )
        SELECT t.player_id, t.season,
               ln(t.target_value / prev.prev_value) AS bt_last_update_log_change,
               fut.fv AS outcome_future_value, fut.fd AS outcome_future_date
        FROM targets t
        LEFT JOIN prev USING (player_id, season)
        LEFT JOIN fut USING (player_id, season)
    """)


FINAL_SQL = """
SELECT
    cur.player_id, cur.season,
    cur.season || '/' || lpad(((cur.season + 1) % 100)::VARCHAR, 2, '0') AS season_label,
    pl.name, cur.club_id, cl.name AS club_name, cur.league, cur.league_name,
    pl.image_url, pl.contract_expiration_date,
    cur.start_date, cur.end_date, t.target_date, t.target_value,

    -- profile
    date_diff('day', pl.date_of_birth, cur.end_date) / 365.25 AS age,
    pl.height_cm, pl.foot, pl.citizenship,
    (coalesce(pl.citizenship = cur.country_name, false))::INT AS is_domestic,
    coalesce(cur.season_position, pl.profile_position) AS position,
    coalesce(pm.position_group, pl.profile_position_group) AS position_group,

    -- season
    cur.apps, cur.starts, cur.minutes, cur.goals, cur.assists,
    cur.yellow_cards, cur.red_cards,
    cur.goals * 90.0 / nullif(cur.minutes, 0) AS goals_per90,
    cur.assists * 90.0 / nullif(cur.minutes, 0) AS assists_per90,
    cur.ga * 90.0 / nullif(cur.minutes, 0) AS ga_per90,
    (cur.yellow_cards + 2 * cur.red_cards) * 90.0 / nullif(cur.minutes, 0) AS cards_per90,
    cur.league_minutes,
    least(cur.main_club_league_minutes / nullif(cs.club_games * 90.0, 0), 1.0)
        AS league_minutes_share,
    cur.starts / nullif(cur.apps, 0) AS start_share,
    cur.captain_games, cur.euro_minutes, cur.cup_minutes, cur.n_clubs_season,
    CASE WHEN prev.player_id IS NOT NULL THEN (prev.club_id <> cur.club_id)::INT END
        AS changed_club,
    CASE WHEN prev.player_id IS NOT NULL THEN (prev.league <> cur.league)::INT END
        AS changed_league,

    -- trend
    (prev.player_id IS NOT NULL)::INT AS prev_observed,
    prev.minutes AS prev_minutes, prev.apps AS prev_apps,
    prev.ga * 90.0 / nullif(prev.minutes, 0) AS prev_ga_per90,
    cur.minutes - prev.minutes AS minutes_delta,
    cur.ga * 90.0 / nullif(cur.minutes, 0) - prev.ga * 90.0 / nullif(prev.minutes, 0)
        AS ga_per90_delta,
    l3.l3_seasons, l3.l3_minutes, l3.l3_goals, l3.l3_assists,

    -- club
    cs.club_position, cs.club_position / cur.n_clubs AS club_position_pct,
    cs.club_ppg, cs.club_gd_per_game,
    coalesce(ce.club_in_europe, false)::INT AS club_in_europe,
    sq.club_players_used, sq.club_avg_age, sq.club_foreign_share,
    -- club spend excludes the player's own fee (see own_fees)
    ((coalesce(ct.club_fees_in, 0) - coalesce(own.own_in, 0))
     - (coalesce(ct.club_fees_out, 0) - coalesce(own.own_out, 0))) / exp(gi.idx)
        AS club_net_spend_rel,
    (coalesce(ct.club_fees_in, 0) - coalesce(own.own_in, 0)) / exp(gi.idx)
        AS club_fees_in_rel,
    (coalesce(ct.club_fees_out, 0) - coalesce(own.own_out, 0)) / exp(gi.idx)
        AS club_fees_out_rel,
    coalesce(ct.club_n_transfers, 0) AS club_n_transfers,
    ct.club_fee_disclosed_share,

    -- market
    gi.idx AS market_index, li.idx - gi.idx AS league_index_rel,
    ln(t.target_value) - gi.idx AS y,

    -- value history (ceiling model only)
    ln(vh.val_last) - gi.idx AS val_last_rel,
    ln(vh.val_season_start) - gi.idx AS val_season_start_rel,
    ln(vh.val_peak) - gi.idx AS val_peak_rel,
    ln(vh.val_12m) - gi.idx AS val_12m_rel,
    ln(vh.val_last) - ln(vh.val_12m) AS val_trend_12m,
    date_diff('day', vh._val_hist_date, cur.end_date) AS days_since_val_last,

    -- backtest (never features)
    bt.bt_last_update_log_change,
    gi_t.idx AS index_at_target, gi_f.idx AS index_at_future,
    bt.outcome_future_value, bt.outcome_future_date,
    ln(bt.outcome_future_value / t.target_value) AS outcome_log_change,
    ln(bt.outcome_future_value / t.target_value) - (gi_f.idx - gi_t.idx) AS outcome_rel_change,

    -- audit (point-in-time checks in src/data/validate.py)
    cur._max_app_date, cur._max_lineup_date, ct._max_transfer_date,
    vh._val_hist_date, vh._val_cutoff, gi.asof_date AS _index_asof_date

FROM ps_all cur
JOIN targets t USING (player_id, season)
JOIN clean_players pl ON pl.player_id = cur.player_id
LEFT JOIN clean_clubs cl ON cl.club_id = cur.club_id
LEFT JOIN (SELECT DISTINCT position, position_group FROM position_map) pm
       ON pm.position = cur.season_position
LEFT JOIN ps_all prev ON prev.player_id = cur.player_id AND prev.season = cur.season - 1
LEFT JOIN (
    SELECT a.player_id, a.season, count(*) AS l3_seasons, sum(b.minutes) AS l3_minutes,
           sum(b.goals) AS l3_goals, sum(b.assists) AS l3_assists
    FROM ps_all a JOIN ps_all b
      ON b.player_id = a.player_id AND b.season BETWEEN a.season - 2 AND a.season
    GROUP BY ALL
) l3 ON l3.player_id = cur.player_id AND l3.season = cur.season
LEFT JOIN club_season cs ON cs.club_id = cur.club_id AND cs.season = cur.season
LEFT JOIN club_europe ce ON ce.club_id = cur.club_id AND ce.season = cur.season
LEFT JOIN club_squad sq ON sq.club_id = cur.club_id AND sq.season = cur.season
LEFT JOIN club_transfers ct ON ct.club_id = cur.club_id AND ct.season = cur.season
LEFT JOIN own_fees own ON own.player_id = cur.player_id AND own.season = cur.season
JOIN market_index gi ON gi.season = cur.season AND gi.league = '_ALL'
LEFT JOIN market_index li ON li.season = cur.season AND li.league = cur.league
LEFT JOIN market_index gi_t ON gi_t.season = cur.season + 1 AND gi_t.league = '_ALL'
LEFT JOIN market_index gi_f ON gi_f.season = cur.season + 2 AND gi_f.league = '_ALL'
LEFT JOIN val_hist vh USING (player_id, season)
LEFT JOIN backtest bt USING (player_id, season)
WHERE cur.complete
  AND cur.season >= $first_season
  AND cur.minutes >= $min_minutes
  AND pl.date_of_birth IS NOT NULL
ORDER BY cur.season, cur.player_id
"""


def build_panel(
    raw_dir: Path = config.RAW_DIR,
    as_of: str | None = None,
    first_season: int = config.FIRST_PANEL_SEASON,
    min_minutes: int = config.MIN_MINUTES,
) -> pd.DataFrame:
    """Build the panel from raw CSVs. ``as_of`` simulates the data as it existed then."""
    con = load.connect(raw_dir, as_of=as_of)
    clean.register_clean_tables(con)
    _league_seasons(con)
    _appearances(con)
    _player_seasons(con)
    _club_seasons(con)
    _market_index(con)
    _targets_and_history(con)
    df = con.execute(FINAL_SQL, {"first_season": first_season, "min_minutes": min_minutes}).df()
    con.close()
    # Plain float64 for every numeric feature: DuckDB returns nullable Int32/Int64 with
    # pd.NA, which scikit-learn rejects.
    numeric = [c for c in feature_sets.CEILING_FEATURES if c not in feature_sets.CATEGORICAL]
    df[numeric] = df[numeric].astype("float64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--as-of", default=None, help="ISO date: truncate all dated data")
    parser.add_argument("--out", type=Path, default=config.PANEL_PATH)
    args = parser.parse_args()

    t0 = time.time()
    panel = build_panel(args.raw_dir, as_of=args.as_of)
    validate.assert_valid(panel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)

    summary = panel.groupby("season").agg(
        rows=("player_id", "size"),
        median_value=("target_value", "median"),
        market_index=("market_index", "first"),
    )
    print(summary.to_string())
    print(
        f"\n{len(panel):,} rows, {panel.shape[1]} columns -> {args.out} ({time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
