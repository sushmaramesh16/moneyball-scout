"""Streamlit smoke tests: every page renders without exceptions from the artifacts."""

import pytest
from streamlit.testing.v1 import AppTest

from src import config

pytestmark = pytest.mark.skipif(
    not (config.ARTIFACTS_DIR / "model.joblib").exists(), reason="deployment artifacts not built"
)

APP = str(config.ROOT / "app" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("SCOUT_API_URL", raising=False)  # use local artifacts
    return AppTest.from_file(APP, default_timeout=120).run()


def test_scout_page_defaults_to_top_debiased_gem(app):
    assert not app.exception
    assert app.title[0].value == "Scout a player"
    labels = [m.label for m in app.metric]
    assert labels[0].startswith("Actual value") and "de-biased" in labels[2]


@pytest.mark.parametrize(
    "page, title",
    [
        ("views/gems.py", "Undervalued Gems"),
        ("views/backtest.py", "Backtest: do flagged players' values actually rise?"),
        ("views/model.py", "How the model works"),
    ],
)
def test_pages_render(app, page, title):
    app.switch_page(page).run()
    assert not app.exception
    assert app.title[0].value == title


def test_gems_respects_default_floor_and_raw_toggle(app):
    app.switch_page("views/gems.py").run()
    debiased = app.dataframe[0].value
    assert (debiased["Actual (€M)"] >= config.VALUE_FLOOR / 1e6).all()
    assert debiased["Score"].is_monotonic_decreasing
    app.sidebar.radio[0].set_value("raw").run()
    assert not app.exception
    raw = app.dataframe[0].value
    assert raw["Score"].is_monotonic_decreasing
    assert raw["Player"].tolist() != debiased["Player"].tolist()  # re-ranked


def test_backtest_page_shows_headline_ci_and_missingness(app):
    app.switch_page("views/backtest.py").run()
    text = " ".join(m.value for m in app.markdown)
    assert "Headline: 2024/25" in text and "no value floor" in text
    assert "matched lift" in text.lower()
    assert "flagged players are missing more often" in text.lower()
    headline = app.dataframe[0].value
    assert headline["Matched lift [95% CI]"].str.contains(r"\[").any()
    assert {
        "Young regulars (U23 by minutes) · top 10%",
        "Mean reversion (biggest recent drop) · top 10%",
    } <= set(headline["Strategy"])
