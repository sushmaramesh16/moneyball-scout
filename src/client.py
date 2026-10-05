"""Backends for the Streamlit app.

- LocalBackend: loads the artifacts in-process (Hugging Face Spaces; no API needed).
- ApiBackend:   calls the FastAPI service (local / docker-compose), chosen when the
                SCOUT_API_URL environment variable is set.

Both return the same JSON-shaped dicts, so the UI does not care which one it has.
"""

from __future__ import annotations

import os

import httpx

from src.service import ScoutService, get_service


class LocalBackend:
    name = "local artifacts"

    def __init__(self, service: ScoutService | None = None):
        self.svc = service or get_service()

    def reports_enabled(self) -> bool:
        return False  # the LLM scouting-report module is added in the next stage

    def filter_options(self):
        return self.svc.filter_options()

    def search(self, query="", limit=20):
        return self.svc.search(query, limit)

    def player(self, player_id):
        return self.svc.player(player_id)

    def explain(self, player_id):
        return self.svc.explain(player_id)

    def undervalued(self, **filters):
        return self.svc.undervalued(**filters)

    def backtest_summary(self):
        return self.svc.backtest_summary()

    def global_importance(self, top=15):
        return self.svc.global_importance(top)

    def all_players(self):
        p = self.svc.players
        return p[["player_id", "name", "club_name", "league", "target_value"]].to_dict("records")


class ApiBackend:
    name = "FastAPI"

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def _get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        r = self.http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def reports_enabled(self) -> bool:
        return bool(self._get("/health").get("reports_enabled"))

    def filter_options(self):
        return self._get("/filters")

    def search(self, query="", limit=20):
        return self._get("/players", q=query, limit=limit)

    def player(self, player_id):
        return self._get(f"/players/{player_id}")

    def explain(self, player_id):
        return self._get(f"/explain/{player_id}")

    def undervalued(self, **filters):
        if filters.get("min_value") is None:
            filters["min_value"] = 0
        return self._get("/undervalued", **filters)

    def backtest_summary(self):
        return self._get("/backtest")

    def global_importance(self, top=15):
        return self._get("/model/importance", top=top)

    def all_players(self):
        rows = self._get("/players", limit=10_000)
        return [{**r, "target_value": r["actual_value"]} for r in rows]


def get_backend():
    url = os.environ.get("SCOUT_API_URL")
    return ApiBackend(url) if url else LocalBackend()
