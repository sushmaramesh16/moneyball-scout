"""Leakage tests: features must depend only on data dated on or before season end.

Synthetic tests always run. The real-data test rebuilds the panel as the data looked on
2024-06-30 and requires every 2015/16-2023/24 feature to match the full build exactly.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import row
from synthetic import write_raw

from src import config
from src.data.validate import validate_panel
from src.features import feature_sets as fs
from src.features.build_features import build_panel

ALL_FEATURES = fs.CEILING_FEATURES


def assert_features_equal(a: pd.DataFrame, b: pd.DataFrame, features=ALL_FEATURES):
    """Same rows, same feature values (NaN == NaN)."""
    a = a.set_index(fs.ID_COLS).sort_index()
    b = b.set_index(fs.ID_COLS).sort_index()
    assert a.index.equals(b.index), "row sets differ"
    bad = []
    for col in features:
        x, y = a[col], b[col]
        if col in fs.CATEGORICAL:
            same = (x == y) | (x.isna() & y.isna())
        else:
            same = np.isclose(
                x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True
            )
        if not np.all(same):
            bad.append(col)
    assert not bad, f"features changed: {bad}"


# --- feature-set contract ----------------------------------------------------------


def test_main_model_has_no_value_history():
    assert not set(fs.VALUE_HISTORY) & set(fs.feature_columns("main"))
    assert not any(c.startswith("val_") for c in fs.feature_columns("main"))


def test_ceiling_is_main_plus_value_history():
    assert fs.feature_columns("ceiling") == fs.feature_columns("main") + fs.VALUE_HISTORY


@pytest.mark.parametrize("kind", ["main", "ceiling"])
def test_no_forbidden_columns(kind):
    cols = fs.feature_columns(kind)
    for col in cols:
        for bad in fs.FORBIDDEN_SUBSTRINGS:
            assert bad not in col, f"{col!r} matches forbidden pattern {bad!r}"
    non_features = set(
        fs.ID_COLS + fs.BACKTEST_COLS + fs.DISPLAY_COLS + [fs.TARGET, fs.TARGET_VALUE, fs.INDEX_COL]
    )
    assert not non_features & set(cols)
    assert len(cols) == len(set(cols))


def test_every_feature_exists_in_panel(panel):
    missing = set(fs.CEILING_FEATURES + fs.BACKTEST_COLS + fs.DISPLAY_COLS) - set(panel)
    assert not missing


# --- point-in-time behaviour on synthetic data -------------------------------------

POST_SEASON_2014 = {
    # Everything below is dated after the 2014/15 season end (2015-05-20).
    "games": [
        dict(
            game_id=9001,
            competition_id="CL",
            season=2014,
            date=date(2015, 6, 1),
            home_club_id=1,
            away_club_id=3,
            home_club_goals=5,
            away_club_goals=0,
            home_club_position=None,
            away_club_position=None,
            competition_type="international_cup",
        )
    ],
    "appearances": [
        dict(
            game_id=9001,
            player_id=101,
            player_club_id=1,
            date=date(2015, 6, 1),
            competition_id="CL",
            yellow_cards=1,
            red_cards=0,
            goals=3,
            assists=2,
            minutes_played=90,
        )
    ],
    "game_lineups": [
        dict(
            game_id=9001,
            player_id=101,
            club_id=1,
            date=date(2015, 6, 1),
            type="starting_lineup",
            position="Left Winger",
            team_captain=1,
        )
    ],
    "transfers": [
        dict(
            player_id=777,
            transfer_date=date(2015, 6, 5),
            transfer_season="14/15",
            from_club_id=50,
            to_club_id=1,
            transfer_fee=50_000_000.0,
        )
    ],
    # after season end but before the 2015-06-10 target: may change the target only
    "player_valuations": [
        dict(
            player_id=101,
            date=date(2015, 5, 25),
            market_value_in_eur=99_000_000,
            player_club_domestic_competition_id="GB1",
        )
    ],
}


def test_post_season_data_does_not_change_features(tmp_path, panel):
    perturbed = build_panel(write_raw(tmp_path / "perturbed", POST_SEASON_2014), first_season=2014)
    assert_features_equal(panel[panel.season == 2014], perturbed[perturbed.season == 2014])
    # the perturbation was live: the nearer post-season valuation became the target
    assert row(perturbed, 101, 2014).target_value == 99_000_000


def test_in_season_data_does_change_features(tmp_path, panel):
    """Negative control: proves the comparison above is not vacuous. (Dated 2015-03-01
    so that the June valuation, 21 days after season end, remains the target.)"""
    extra = {
        "player_valuations": [
            dict(
                player_id=101,
                date=date(2015, 3, 1),
                market_value_in_eur=99_000_000,
                player_club_domestic_competition_id="GB1",
            )
        ]
    }
    changed = build_panel(write_raw(tmp_path / "changed", extra), first_season=2014)
    before, after = row(panel, 101, 2014), row(changed, 101, 2014)
    assert after.target_value == before.target_value
    assert after.val_last_rel == pytest.approx(np.log(99e6) - after.market_index)
    assert after.val_last_rel != pytest.approx(before.val_last_rel)


def test_as_of_build_matches_full_build(raw_dir, panel):
    truncated = build_panel(raw_dir, as_of="2015-06-30", first_season=2014)
    assert set(truncated.season) == {2014}, "2015/16 had not started on 2015-06-30"
    assert_features_equal(panel[panel.season == 2014], truncated)


def test_unfinished_season_is_excluded(raw_dir):
    """Mid-season (after the winter break) the current season must not appear."""
    truncated = build_panel(raw_dir, as_of="2015-03-01", first_season=2014)
    assert truncated.empty


@pytest.mark.parametrize(
    "column, shift_days, message",
    [
        ("_max_app_date", 1, "_max_app_date <= end_date"),
        ("_max_transfer_date", 1, "_max_transfer_date <= end_date"),
        ("_val_hist_date", 30, "_val_hist_date < target_date"),
    ],
)
def test_validator_flags_injected_violations(panel, column, shift_days, message):
    bad = panel.copy()
    anchor = "target_date" if "target" in message else "end_date"
    bad.loc[bad.index[0], column] = pd.Timestamp(bad.loc[bad.index[0], anchor]) + pd.Timedelta(
        days=shift_days
    )
    assert any(message in e for e in validate_panel(bad))


# --- real data ---------------------------------------------------------------------

REAL_AS_OF = "2024-06-30"
needs_real_data = pytest.mark.skipif(
    not (config.RAW_DIR / "players.csv").exists(), reason="data/raw not present"
)


@pytest.fixture(scope="module")
def real_builds():
    return build_panel(config.RAW_DIR), build_panel(config.RAW_DIR, as_of=REAL_AS_OF)


@pytest.mark.realdata
@needs_real_data
def test_real_panel_is_valid(real_builds):
    full, _ = real_builds
    assert validate_panel(full) == []


@pytest.mark.realdata
@needs_real_data
def test_real_as_of_build_matches_full_build(real_builds):
    full, truncated = real_builds
    past = full[full.season <= 2023]
    shared = past.merge(truncated[fs.ID_COLS], on=fs.ID_COLS)
    assert len(shared) > 40_000
    assert_features_equal(shared, truncated.merge(shared[fs.ID_COLS], on=fs.ID_COLS))

    # Rows only in the full build must be explained by a target dated after the cut-off.
    dropped = past.merge(truncated[fs.ID_COLS], on=fs.ID_COLS, how="left", indicator=True)
    dropped = dropped[dropped["_merge"] == "left_only"]
    assert (pd.to_datetime(dropped.target_date) > pd.Timestamp(REAL_AS_OF)).all()
    assert set(truncated.season) <= set(range(config.FIRST_PANEL_SEASON, 2024))


@pytest.mark.realdata
@needs_real_data
def test_no_main_feature_is_a_proxy_for_the_target(real_builds):
    full, _ = real_builds
    numeric = [c for c in fs.feature_columns("main") if c not in fs.CATEGORICAL]
    corr = full[numeric].corrwith(full[fs.TARGET]).abs()
    assert corr.max() < 0.9, corr.sort_values().tail(3)


def _signing(player_id):
    return {
        "transfers": [
            dict(
                player_id=player_id,
                transfer_date=date(2014, 7, 20),
                transfer_season="14/15",
                from_club_id=50,
                to_club_id=1,
                transfer_fee=20_000_000.0,
            )
        ]
    }


def test_own_transfer_fee_excluded_from_club_spend(tmp_path, panel):
    """A player's own fee is a price for him (value history), so it must not reach the
    main model through his club's spend; another player's fee must."""
    own = build_panel(write_raw(tmp_path / "own", _signing(101)), first_season=2014)
    other = build_panel(write_raw(tmp_path / "other", _signing(555)), first_season=2014)
    base = row(panel, 101, 2014)
    for col in ("club_fees_in_rel", "club_net_spend_rel"):
        assert row(own, 101, 2014)[col] == pytest.approx(base[col])
        assert row(other, 101, 2014)[col] == pytest.approx(
            base[col] + 20e6 / np.exp(base.market_index)
        )
