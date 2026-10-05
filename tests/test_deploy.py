"""Deployment wiring: Community Cloud requirements, secrets fallback, cache fallback."""

import os
import stat
import sys

import pytest

from src import config
from src.explain import report

TRAINING_ONLY = {"duckdb", "xgboost", "mlflow", "optuna", "shap", "pytest", "ruff"}


def _pins(path):
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        assert not line.startswith("-r"), f"{path.name}: no includes (Community Cloud)"
        name, _, version = line.partition("==")
        assert version, f"{path.name}: {line!r} is not pinned exactly"
        pins[name.lower()] = version
    return pins


def test_app_requirements_match_deploy_pins_without_training_deps():
    app_pins = _pins(config.ROOT / "app" / "requirements.txt")
    deploy_pins = _pins(config.ROOT / "requirements-deploy.txt")
    for name, version in app_pins.items():
        assert deploy_pins.get(name) == version, f"{name}: {version} vs {deploy_pins.get(name)}"
    assert {"scikit-learn", "lightgbm", "streamlit", "google-genai"} <= set(app_pins)
    assert not TRAINING_ONLY & set(app_pins)


def test_packages_txt_installs_openmp_for_lightgbm():
    assert "libgomp1" in (config.ROOT / "packages.txt").read_text().split()


def test_streamlit_secrets_are_a_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    secrets = {"GEMINI_API_KEY": "from-secrets", "GEMINI_MODEL": "gemini-secret-model"}
    monkeypatch.setattr(report, "_streamlit_secret", secrets.get)
    assert report.is_available() and report.setting("GEMINI_API_KEY") == "from-secrets"
    assert report.model_name() == "gemini-secret-model"
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-model")  # environment wins
    assert report.model_name() == "gemini-env-model"


def test_streamlit_secret_lookup_is_none_outside_streamlit(monkeypatch):
    monkeypatch.undo()  # use the real lookup, not the conftest stub
    assert report._streamlit_secret("GEMINI_API_KEY") is None


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="permission bits not enforced"
)
def test_cache_dir_falls_back_to_temp_when_not_writable(monkeypatch, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read-only
    try:
        monkeypatch.setenv("REPORT_CACHE_DIR", str(locked / "reports"))
        cache = report._cache_dir()
        assert cache.name == "moneyball-reports" and locked not in cache.parents
        monkeypatch.setenv("REPORT_CACHE_DIR", str(tmp_path / "ok"))
        assert report._cache_dir() == tmp_path / "ok"
    finally:
        locked.chmod(stat.S_IRWXU)
