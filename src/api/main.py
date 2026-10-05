"""Moneyball Scout API (local / Docker). Run with:

uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query

from src import config
from src.api import schemas
from src.explain import report as report_module
from src.service import PlayerNotFound, ScoutService, get_service

app = FastAPI(
    title="Moneyball Scout API",
    description="Predicted market values, undervalued players and SHAP explanations "
    "for players in 14 European leagues (Transfermarkt data, CC0).",
    version="1.0.0",
)

Service = Annotated[ScoutService, Depends(get_service)]


def _player_or_404(fn, player_id: int):
    try:
        return fn(player_id)
    except PlayerNotFound:
        raise HTTPException(
            404, f"No player {player_id} in the {config.LIVE_SEASON} list"
        ) from None


@app.get("/health", response_model=schemas.Health)
def health(svc: Service):
    return {
        "status": "ok",
        "season": svc.season_label,
        "n_players": len(svc.players),
        "reports_enabled": svc.reports_enabled(),
    }


@app.get("/filters")
def filters(svc: Service):
    return svc.filter_options()


@app.get("/players", response_model=list[schemas.PlayerSummary])
def search_players(svc: Service, q: str = "", limit: int = Query(20, ge=1, le=10_000)):
    return svc.search(q, limit)


@app.get("/players/{player_id}", response_model=schemas.PlayerDetail)
def player(svc: Service, player_id: int):
    return _player_or_404(svc.player, player_id)


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(svc: Service, req: schemas.PredictRequest):
    try:
        return svc.predict(req.features, req.actual_value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/undervalued", response_model=list[schemas.RankedPlayer])
def undervalued(
    svc: Service,
    position_group: str | None = None,
    position: str | None = None,
    league: str | None = None,
    min_age: float | None = Query(None, ge=0),
    max_age: float | None = Query(None, ge=0),
    min_value: float = Query(
        config.VALUE_FLOOR, ge=0, description="Minimum actual value in euros (0 = no floor)"
    ),
    score: Literal["debiased", "raw"] = "debiased",
    limit: int = Query(50, ge=1, le=500),
):
    return svc.undervalued(
        position_group, position, league, min_age, max_age, min_value, score, limit
    )


@app.get("/explain/{player_id}", response_model=schemas.Explanation)
def explain(svc: Service, player_id: int):
    return _player_or_404(svc.explain, player_id)


@app.get("/report/{player_id}", response_model=schemas.Report)
def report(svc: Service, player_id: int, refresh: bool = False):
    """AI-generated scouting note grounded in the player's stats and SHAP drivers.
    503 if no GEMINI_API_KEY is configured; 429 (with Retry-After) when rate limited."""
    _player_or_404(svc.player, player_id)
    try:
        return svc.scouting_report(player_id, force=refresh)
    except report_module.ReportUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except report_module.ReportRateLimited as exc:
        raise HTTPException(
            429,
            "The AI report service is busy; please try again shortly.",
            headers={"Retry-After": str(int(exc.retry_after))},
        ) from None
    except report_module.ReportError as exc:
        raise HTTPException(502, str(exc)) from None


@app.get("/backtest")
def backtest(svc: Service):
    return svc.backtest_summary()


@app.get("/model/importance")
def importance(svc: Service, top: int = Query(15, ge=1, le=60)):
    return svc.global_importance(top)
