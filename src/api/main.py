"""Moneyball Scout API (local / Docker). Run with:

uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query

from src import config
from src.api import schemas
from src.service import PlayerNotFound, ScoutService, get_service

app = FastAPI(
    title="Moneyball Scout API",
    description="Predicted market values, undervalued players and SHAP explanations "
    "for players in 14 European leagues (Transfermarkt data, CC0).",
    version="1.0.0",
)

Service = Annotated[ScoutService, Depends(get_service)]

REPORTS_ENABLED = False  # the LLM scouting-report module is added in the next stage


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
        "reports_enabled": REPORTS_ENABLED,
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


@app.get("/report/{player_id}")
def report(svc: Service, player_id: int):
    _player_or_404(svc.player, player_id)
    if not REPORTS_ENABLED:
        raise HTTPException(503, "Scouting reports are not enabled on this server.")


@app.get("/backtest")
def backtest(svc: Service):
    return svc.backtest_summary()


@app.get("/model/importance")
def importance(svc: Service, top: int = Query(15, ge=1, le=60)):
    return svc.global_importance(top)
