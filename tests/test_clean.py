from datetime import date

import pytest
from synthetic import write_raw

from src.data import clean, load


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Centre-Forward", "Centre-Forward"),
        ("centre-back", "Centre-Back"),
        ("  Left Winger ", "Left Winger"),
        ("midfield", "Midfield"),
        ("Attack", "Attack"),
        ("Sweeper", "Centre-Back"),
        ("Missing", None),
        ("", None),
        (None, None),
        (float("nan"), None),
        ("Libero-ish", None),
    ],
)
def test_normalize_position(raw, expected):
    assert clean.normalize_position(raw) == expected


@pytest.mark.parametrize(
    "position, group",
    [
        ("Centre-Forward", "Attack"),
        ("Right-Back", "Defender"),
        ("Goalkeeper", "Goalkeeper"),
        ("Defensive Midfield", "Midfield"),
        ("Midfield", "Midfield"),
        (None, None),
    ],
)
def test_position_group(position, group):
    assert clean.position_group(position) == group


def test_every_sub_position_has_a_group():
    groups = set(clean.SUB_POSITION_GROUP.values())
    assert groups == {"Goalkeeper", "Defender", "Midfield", "Attack"}
    for alias in clean.POSITION_ALIASES:
        assert clean.position_group(clean.normalize_position(alias)) in groups


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Right", "right"),
        ("left", "left"),
        ("both", "both"),
        (None, "unknown"),
        ("", "unknown"),
        ("ambidextrous", "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_normalize_foot(raw, expected):
    assert clean.normalize_foot(raw) == expected


@pytest.mark.parametrize(
    "cm, expected",
    [
        (180, 180.0),
        (150, 150.0),
        (210, 210.0),
        (0, None),
        (19, None),
        (250, None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_clean_height(cm, expected):
    assert clean.clean_height(cm) == expected


def test_season_labels():
    assert clean.transfer_season_label(2023) == "23/24"
    assert clean.transfer_season_label(1999) == "99/00"
    assert clean.season_label(2009) == "2009/10"


@pytest.fixture
def dirty_con(tmp_path):
    extra = {
        # duplicate of an existing appearance with impossible minutes and negative goals
        "appearances": [
            dict(
                game_id=1,
                player_id=101,
                player_club_id=1,
                date=date(2013, 8, 15),
                competition_id="GB1",
                yellow_cards=0,
                red_cards=0,
                goals=-1,
                assists=0,
                minutes_played=200,
            )
        ],
        "player_valuations": [
            dict(
                player_id=102,
                date=date(2014, 3, 1),
                market_value_in_eur=0,
                player_club_domestic_competition_id="GB1",
            )
        ],
        "transfers": [
            dict(
                player_id=1,
                transfer_date=date(2014, 7, 1),
                transfer_season="14/15",
                from_club_id=1,
                to_club_id=2,
                transfer_fee=-5.0,
            ),
            dict(
                player_id=2,
                transfer_date=date(2014, 7, 1),
                transfer_season="14/15",
                from_club_id=2,
                to_club_id=2,
                transfer_fee=1.0,
            ),
        ],
    }
    con = load.connect(write_raw(tmp_path / "raw", extra))
    clean.register_clean_tables(con)
    return con


def test_appearances_deduplicated_and_clipped(dirty_con):
    rows = dirty_con.execute(
        "SELECT minutes, goals FROM clean_appearances WHERE game_id = 1 AND player_id = 101"
    ).fetchall()
    assert rows == [(clean.MAX_MINUTES_PER_GAME, 0)]


def test_zero_valuations_dropped(dirty_con):
    n = dirty_con.execute("SELECT count(*) FROM clean_valuations WHERE value <= 0").fetchone()[0]
    assert n == 0


def test_transfer_fees_cleaned(dirty_con):
    neg = dirty_con.execute("SELECT fee FROM clean_transfers WHERE player_id = 1").fetchone()
    assert neg == (None,), "negative fee becomes undisclosed (NULL), not zero"
    same_club = dirty_con.execute(
        "SELECT count(*) FROM clean_transfers WHERE player_id = 2"
    ).fetchone()[0]
    assert same_club == 0, "a 'move' to the same club is not a transfer"


def test_player_attributes_cleaned(dirty_con):
    df = (
        dirty_con.execute(
            "SELECT player_id, foot, height_cm, profile_position, profile_position_group "
            "FROM clean_players ORDER BY player_id"
        )
        .df()
        .set_index("player_id")
    )
    assert df.loc[101, "foot"] == "right"
    assert df.loc[103, "foot"] == "unknown"
    assert df.loc[101, "profile_position"] == "Centre-Forward"
    # no sub_position -> falls back to the broad position
    assert df.loc[103, "profile_position"] == "Midfield"
    assert df.loc[103, "profile_position_group"] == "Midfield"


def test_as_of_truncates_dated_tables(tmp_path):
    con = load.connect(write_raw(tmp_path / "raw"), as_of="2014-06-30")
    latest = con.execute("SELECT max(date) FROM raw_appearances").fetchone()[0]
    assert latest <= date(2014, 6, 30)
    # static tables are untouched
    assert con.execute("SELECT count(*) FROM raw_players").fetchone()[0] == 4
