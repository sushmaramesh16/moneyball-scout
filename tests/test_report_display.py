"""Report prompt/caching logic and plain-language feature values (no network)."""

import math

import pytest

from src.explain import report
from src.explain.shap_explain import display_value
from src.service import driver_summary


@pytest.mark.parametrize(
    "feature, value, mi, expected",
    [
        ("league_index_rel", math.log(0.5), None, "50% of market average"),
        ("club_fees_in_rel", 2.0, math.log(1e6), "€2.0M"),
        ("club_net_spend_rel", -0.5, math.log(1e6), "−€500K"),
        ("minutes", 1382.0, None, "1,382"),
        ("goals", 3.0, None, "3"),
        ("league", "GB1", None, "Premier League (England)"),
        ("is_eu_eea", 1.0, None, "yes"),
        ("start_share", 0.25, None, "25%"),
        ("age", 20.43, None, "20.4 years"),
        ("ga_per90", 0.4567, None, "0.46"),
        ("goals", float("nan"), None, "unknown"),
        ("position", "Centre-Back", None, "Centre-Back"),
    ],
)
def test_display_value(feature, value, mi, expected):
    assert display_value(feature, value, mi) == expected


def _contrib(feature, shap):
    return {"feature": feature, "shap": shap}


def test_driver_summary_classifies_context_performance_mixed():
    ctx = driver_summary([_contrib("age", 0.6), _contrib("goals", 0.1)])
    assert ctx["driver"] == "context"
    perf = driver_summary([_contrib("age", 0.1), _contrib("ga_per90", 0.5)])
    assert perf["driver"] == "performance"
    mixed = driver_summary([_contrib("age", 0.3), _contrib("minutes", 0.25)])
    assert mixed["driver"] == "mixed"


PLAYER = {
    "player_id": 7,
    "name": "Ada Forward",
    "age": 21.2,
    "position": "Centre-Forward",
    "club_name": "Club 1",
    "league_name": "Premier League (England)",
    "season_label": "2025/26",
    "actual_value": 1e6,
    "predicted_value": 2e6,
    "undervalued_score": 0.8,
    "undervalued_score_raw": 1.0,
    "predicted_range": {"middle_50": [1.5e6, 2.6e6], "middle_80": [1.1e6, 3.4e6]},
    "stats": {
        "apps": 30,
        "starts": 25,
        "minutes": 2300,
        "goals": 12,
        "assists": 4,
        "ga_per90": 0.63,
    },
}
EXPLANATION = {
    "contributions": [{"label": "Age", "display": "21.2 years", "effect_pct": 40.0}],
    "drivers": {"context_effect_pct": 40.0, "performance_effect_pct": 10.0, "driver": "context"},
}


def test_facts_contain_only_supplied_information():
    facts = report.build_facts(
        PLAYER, EXPLANATION, {"typical_error_pct": 34, "hit_rate": 0.36, "base_rate": 0.30}
    )
    for snippet in (
        "Ada Forward",
        "€1.0M",
        "€2.0M",
        "€1.5M–€2.6M",
        "12 goals",
        "Age: 21.2 years (+40%)",
        "driven by: context",
        "typical error about 34%",
    ):
        assert snippet in facts
    rules = report.SYSTEM_INSTRUCTION.lower()
    for rule in ("only the facts", "injuries", "do not invent numbers", "uncertainty"):
        assert rule in rules


def test_generate_requires_key():
    with pytest.raises(report.ReportUnavailable):
        report.generate(PLAYER, EXPLANATION)


def test_generate_caches_on_disk_and_rejects_empty(monkeypatch):
    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(report, "call_model", lambda f: calls.append(f) or "A grounded note. " * 4)
    first = report.generate(PLAYER, EXPLANATION)
    report._memory.clear()  # force a disk read
    second = report.generate(PLAYER, EXPLANATION)
    assert first["cached"] is False and second["cached"] is True and len(calls) == 1
    assert second["ai_generated"] is True

    monkeypatch.setattr(report, "call_model", lambda f: "")
    with pytest.raises(report.ReportError):
        report.generate(PLAYER, EXPLANATION, force=True)


def test_model_name_is_configurable(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert report.model_name() == report.DEFAULT_MODEL
    monkeypatch.setenv("GEMINI_MODEL", "gemini-x-flash-lite")
    assert report.model_name() == "gemini-x-flash-lite"
