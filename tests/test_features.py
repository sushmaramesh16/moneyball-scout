"""Feature values on the synthetic dataset, checked against hand calculations."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import row

from src.data.validate import validate_panel

AGE_101_2014 = (date(2015, 5, 20) - date(1995, 6, 1)).days / 365.25


def test_panel_rows_and_eligibility(panel):
    keys = set(zip(panel.player_id, panel.season, strict=True))
    assert keys == {(p, s) for p in (101, 102, 103) for s in (2014, 2015)}
    # 104 played 180' (< 450) and 2013 is before first_season


def test_panel_passes_validation(panel):
    assert validate_panel(panel) == []


def test_season_stats(panel):
    r = row(panel, 101, 2014)
    assert pd.Timestamp(r.end_date) == pd.Timestamp("2015-05-20")
    assert r.apps == 8 and r.starts == 8 and r.captain_games == 6
    assert r.minutes == 720 and r.league_minutes == 540
    assert r.euro_minutes == 90 and r.cup_minutes == 90
    assert r.goals == 3 and r.assists == 1
    assert r.ga_per90 == pytest.approx(4 * 90 / 720)
    assert r.league_minutes_share == pytest.approx(1.0)
    assert row(panel, 102, 2014).league_minutes_share == pytest.approx(480 / 540)


def test_profile(panel):
    r = row(panel, 101, 2014)
    assert r.age == pytest.approx(AGE_101_2014)
    assert r.position == "Centre-Forward" and r.position_group == "Attack"
    assert r.foot == "right" and r.is_domestic == 1
    assert row(panel, 102, 2014).is_domestic == 0
    r103 = row(panel, 103, 2014)
    assert r103.position == "Midfield" and r103.position_group == "Midfield"
    assert r103.foot == "unknown" and np.isnan(r103.height_cm)


def test_trend_features(panel):
    r = row(panel, 101, 2014)
    assert r.prev_observed == 1 and r.prev_minutes == 720 and r.prev_apps == 8
    assert r.prev_ga_per90 == pytest.approx(0.5)
    assert r.minutes_delta == 0 and r.changed_club == 0 and r.changed_league == 0
    assert r.l3_seasons == 2 and r.l3_minutes == 1440  # 2012 not in data
    assert row(panel, 101, 2015).l3_seasons == 3


def test_club_features(panel):
    c1 = row(panel, 101, 2014)
    assert c1.club_position == 1 and c1.club_position_pct == pytest.approx(0.5)
    assert c1.club_ppg == pytest.approx((3 * 3 + 3 * 1) / 6)
    assert c1.club_gd_per_game == pytest.approx(0.5)
    assert c1.club_in_europe == 1 and c1.club_players_used == 1
    assert c1.club_avg_age == pytest.approx(AGE_101_2014)
    assert c1.club_foreign_share == 0
    assert c1.club_n_transfers == 2 and c1.club_fee_disclosed_share == pytest.approx(0.5)
    assert c1.club_fees_in_rel * np.exp(c1.market_index) == pytest.approx(3_000_000)
    assert c1.club_net_spend_rel == pytest.approx(c1.club_fees_in_rel)

    c2 = row(panel, 102, 2014)
    assert c2.club_position == 2 and c2.club_in_europe == 0
    assert c2.club_foreign_share == pytest.approx(1.0)
    assert c2.club_n_transfers == 0 and c2.club_net_spend_rel == 0
    assert np.isnan(c2.club_fee_disclosed_share), "no transfers -> share undefined, not 0"


def test_target_and_market_index(panel):
    r = row(panel, 101, 2014)
    assert pd.Timestamp(r.target_date) == pd.Timestamp("2015-06-10")
    assert r.target_value == 12_000_000
    # index(2014) = mean log June-2014 value of players with >= 450' in 2013
    idx = np.mean(np.log([12e6, 2e6, 4e6]))
    assert r.market_index == pytest.approx(idx)
    assert r.y == pytest.approx(np.log(12e6) - idx)
    assert r.league_index_rel == pytest.approx(np.mean(np.log([12e6, 2e6])) - idx)


def test_value_history_uses_only_pre_cutoff_valuations(panel):
    r = row(panel, 101, 2014)
    idx = r.market_index
    assert r.val_last_rel == pytest.approx(np.log(10e6) - idx)  # Dec 2014
    assert r.val_season_start_rel == pytest.approx(np.log(12e6) - idx)  # Jun 2014
    assert r.val_peak_rel == pytest.approx(np.log(12e6) - idx)
    assert r.val_12m_rel == pytest.approx(np.log(10e6) - idx)  # Dec 2013
    assert r.days_since_val_last == (date(2015, 5, 20) - date(2014, 12, 15)).days


def test_backtest_columns(panel):
    r = row(panel, 101, 2014)
    assert r.bt_last_update_log_change == pytest.approx(np.log(12 / 10))
    assert r.outcome_future_value == 12_000_000
    assert r.outcome_log_change == pytest.approx(0.0)
    assert r.outcome_rel_change == pytest.approx(-(r.index_at_future - r.index_at_target))
    assert pd.isna(row(panel, 101, 2015).outcome_future_value)  # no data a year later


def test_eu_eea_flag_respects_brexit(tmp_path):
    from synthetic import write_raw

    from src.features.build_features import build_panel

    # 101 is English: EU citizen for seasons ending before 2021-01-01 only.
    panel = build_panel(write_raw(tmp_path / "raw"), first_season=2014)
    assert row(panel, 101, 2014).is_eu_eea == 1
    assert row(panel, 102, 2014).is_eu_eea == 0  # Brazil
    assert row(panel, 103, 2014).is_eu_eea == 1  # Spain
