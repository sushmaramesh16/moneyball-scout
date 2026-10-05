import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import write_raw  # noqa: E402

from src.features.build_features import build_panel  # noqa: E402


@pytest.fixture
def raw_dir(tmp_path):
    return write_raw(tmp_path / "raw")


@pytest.fixture
def panel(raw_dir):
    return build_panel(raw_dir, first_season=2014)


def row(df, player_id, season):
    match = df[(df.player_id == player_id) & (df.season == season)]
    assert len(match) == 1, f"expected one row for ({player_id}, {season}), got {len(match)}"
    return match.iloc[0]
