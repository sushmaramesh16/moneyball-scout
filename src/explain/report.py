"""AI scouting report: a short, grounded summary of a player from Gemini (optional).

The model only sees the facts we pass (season stats, valuation numbers, the SHAP drivers
and how reliable the model is) and is told not to add anything else. The feature is off
unless GEMINI_API_KEY is set: environment variable / .env locally, or Streamlit secrets
(.streamlit/secrets.toml, or the app's Secrets setting on Streamlit Community Cloud).

Settings (environment first, then Streamlit secrets):
    GEMINI_API_KEY      required to enable reports
    GEMINI_MODEL        default gemini-3.5-flash-lite (free tier)
    REPORT_CACHE_DIR    default .cache/reports, falling back to the system temp dir when
                        that is not writable (one JSON per player/model/prompt version)
    REPORT_MIN_INTERVAL seconds between API calls, default 4 (stays under free-tier RPM)
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from src import config

load_dotenv(config.ROOT / ".env", override=False)

DEFAULT_MODEL = (
    "gemini-3.5-flash-lite"  # current free-tier Flash-Lite (2.5 is retired for new keys)
)
PROMPT_VERSION = "v1"
DISCLAIMER = (
    "AI-generated summary of this app's statistics; it may be wrong and is not "
    "based on any information beyond the numbers shown here."
)

SYSTEM_INSTRUCTION = """You are a football data analyst writing a short scouting note.
Rules:
- Use ONLY the facts provided. Do not add injuries, transfers, rumours, personality,
  playing style, tactics, contract details, nationality or anything else not given.
- Do not invent numbers; only quote numbers that appear in the facts.
- Write 3 to 4 plain sentences, no headings, no bullet points, no markdown.
- Explain what drives the model's valuation (the listed drivers), and say plainly when
  the case rests mostly on context (age, club, league) rather than on-pitch output.
- State the uncertainty: give the likely value range and say the model is a statistical
  estimate whose typical error is large.
- Refer to the player by surname or as "the player"."""


class ReportUnavailable(RuntimeError):
    """No API key / SDK: the feature is switched off."""


class ReportRateLimited(RuntimeError):
    def __init__(self, retry_after: float):
        super().__init__(f"Rate limited; retry in about {retry_after:.0f}s")
        self.retry_after = retry_after


class ReportError(RuntimeError):
    """The model call failed or returned nothing usable."""


def _streamlit_secret(name: str) -> str | None:
    """st.secrets[name] when running inside a Streamlit app, else None. Never raises:
    a missing secrets.toml or a non-Streamlit process (the API) just means 'not set'."""
    try:
        from streamlit import runtime

        if not runtime.exists():
            return None
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def setting(name: str) -> str | None:
    """Environment variable first, then Streamlit secrets."""
    value = os.environ.get(name, "").strip()
    return value or _streamlit_secret(name)


def model_name() -> str:
    return setting("GEMINI_MODEL") or DEFAULT_MODEL


def is_available() -> bool:
    if not setting("GEMINI_API_KEY"):
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


# ---- prompt ---------------------------------------------------------------------------
def _eur(v: float) -> str:
    return f"€{v / 1e6:.1f}M" if v >= 1e6 else f"€{v / 1e3:.0f}K"


def build_facts(player: dict, explanation: dict, reliability: dict | None = None) -> str:
    """Plain-text fact sheet: the only information the model may use."""
    s = player["stats"]
    lines = [
        f"Player: {player['name']}, {player['age']:.0f}, {player['position']}, "
        f"{player['club_name']}, {player['league_name']}, season {player['season_label']}.",
        f"Current market value (Transfermarkt): {_eur(player['actual_value'])}.",
        f"Model's predicted value: {_eur(player['predicted_value'])}.",
    ]
    rng = player.get("predicted_range")
    if rng:
        lo50, hi50 = rng["middle_50"]
        lo80, hi80 = rng["middle_80"]
        lines.append(
            f"Likely range for a player like this: {_eur(lo50)}–{_eur(hi50)} "
            f"(middle 50%), {_eur(lo80)}–{_eur(hi80)} (middle 80%)."
        )
    lines.append(
        f"Undervalued score: {player['undervalued_score']:+.0%} versus league and age peers "
        f"({player['undervalued_score_raw']:+.0%} raw)."
    )
    stat_bits = [
        f"{s['apps']:.0f} appearances",
        f"{s['starts']:.0f} starts",
        f"{s['minutes']:,.0f} minutes",
        f"{s['goals']:.0f} goals",
        f"{s['assists']:.0f} assists",
    ]
    if s.get("ga_per90") is not None:
        stat_bits.append(f"{s['ga_per90']:.2f} goals+assists per 90")
    lines.append("Season stats (all competitions): " + ", ".join(stat_bits) + ".")
    lines.append("Main drivers of the predicted value (effect vs an average player):")
    for f in explanation["contributions"][:6]:
        lines.append(f"- {f['label']}: {f['display']} ({f['effect_pct']:+.0f}%)")
    d = explanation["drivers"]
    lines.append(
        f"Context factors (age, club, league) together: {d['context_effect_pct']:+.0f}%; "
        f"performance factors together: {d['performance_effect_pct']:+.0f}% "
        f"(overall the case is driven by: {d['driver']})."
    )
    if reliability:
        lines.append(
            f"Model reliability: typical error about {reliability['typical_error_pct']:.0f}% "
            f"of value; in a backtest, {reliability['hit_rate']:.0%} of top-flagged players "
            f"rose in value within a year versus {reliability['base_rate']:.0%} of all players."
        )
    return "\n".join(lines)


# ---- cache & throttle -----------------------------------------------------------------
class _Throttle:
    def __init__(self):
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        interval = float(os.environ.get("REPORT_MIN_INTERVAL", "4"))
        with self.lock:
            delay = self.last + interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.last = time.monotonic()


_throttle = _Throttle()
_memory: dict[str, dict] = {}


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def _cache_dir() -> Path:
    """REPORT_CACHE_DIR (or .cache/reports) if writable, else the system temp dir, so the
    disk cache also works on read-only or sandboxed hosts like Streamlit Community Cloud."""
    preferred = Path(os.environ.get("REPORT_CACHE_DIR") or config.ROOT / ".cache" / "reports")
    if _writable(preferred):
        return preferred
    return Path(tempfile.gettempdir()) / "moneyball-reports"


def _cache_key(player: dict) -> str:
    season = player["season_label"].replace("/", "-")
    safe_model = model_name().replace("/", "_")
    return f"{player['player_id']}_{season}_{safe_model}_{PROMPT_VERSION}"


def _cache_get(key: str) -> dict | None:
    if key in _memory:
        return _memory[key]
    path = _cache_dir() / f"{key}.json"
    if path.exists():
        try:
            _memory[key] = json.loads(path.read_text())
            return _memory[key]
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _cache_put(key: str, value: dict) -> None:
    _memory[key] = value
    try:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        (_cache_dir() / f"{key}.json").write_text(json.dumps(value, indent=1))
    except OSError:
        pass  # read-only disk: memory cache still works


# ---- model call -----------------------------------------------------------------------
def _retry_after(exc) -> float:
    """Seconds suggested by a 429 response (RetryInfo), default 30."""
    try:
        for detail in exc.details["error"]["details"]:
            delay = detail.get("retryDelay")
            if delay:
                return float(str(delay).rstrip("s"))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return 30.0


def call_model(facts: str) -> str:
    """One Gemini call. Separated out so tests can replace it."""
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=setting("GEMINI_API_KEY"))
    name = model_name()
    cfg = dict(system_instruction=SYSTEM_INSTRUCTION, temperature=0.3, max_output_tokens=1024)
    if name.startswith("gemini-2.5-flash"):
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    contents = "Facts:\n" + facts + "\n\nWrite the scouting note."
    for attempt in range(2):
        _throttle.wait()
        try:
            response = client.models.generate_content(
                model=name, contents=contents, config=types.GenerateContentConfig(**cfg)
            )
            return (response.text or "").strip()
        except errors.APIError as exc:
            if exc.code == 429:
                wait = _retry_after(exc)
                if attempt == 0 and wait <= 10:
                    time.sleep(wait)
                    continue
                raise ReportRateLimited(wait) from exc
            raise ReportError(f"Gemini API error {exc.code}: {exc.message}") from exc
    raise ReportRateLimited(30.0)


def generate(
    player: dict, explanation: dict, reliability: dict | None = None, force: bool = False
) -> dict:
    if not is_available():
        raise ReportUnavailable("Set GEMINI_API_KEY to enable scouting reports.")
    key = _cache_key(player)
    if not force and (hit := _cache_get(key)):
        return hit | {"cached": True}
    text = call_model(build_facts(player, explanation, reliability))
    if len(text) < 40:
        raise ReportError("The model returned an empty or truncated report.")
    result = {
        "player_id": int(player["player_id"]),
        "report": text,
        "model": model_name(),
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ai_generated": True,
        "disclaimer": DISCLAIMER,
    }
    _cache_put(key, result)
    return result | {"cached": False}
