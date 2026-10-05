"""Request/response models for the Moneyball Scout API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: Literal["ok"]
    season: str
    n_players: int
    reports_enabled: bool


class PlayerSummary(BaseModel):
    player_id: int
    name: str
    club_name: str | None
    league: str
    position: str | None
    age: float
    actual_value: float
    predicted_value: float
    undervalued_score: float = Field(description="De-biased: vs league x age-band peers")
    undervalued_score_raw: float = Field(description="(predicted - actual) / actual")


class RankedPlayer(PlayerSummary):
    rank: int


class ValueRange(BaseModel):
    middle_50: list[float] = Field(description="[low, high] euros: middle 50% of outcomes")
    middle_80: list[float] = Field(description="[low, high] euros: middle 80% of outcomes")
    band: str
    source: str


class PlayerDetail(BaseModel):
    player_id: int
    name: str
    club_name: str | None
    league: str
    league_name: str | None
    position: str | None
    position_group: str | None
    age: float
    foot: str | None
    height_cm: float | None
    citizenship: str | None
    season_label: str
    actual_value: float
    valuation_date: str | None
    predicted_value: float
    predicted_range: ValueRange | None
    undervalued_score: float
    undervalued_score_raw: float
    contract_expiration_date: str | None = Field(
        description="Display only: not a model feature (no point-in-time history)"
    )
    stats: dict[str, float | int | None]


class Factor(BaseModel):
    feature: str
    label: str
    value: float | int | str | None
    display: str | None = Field(default=None, description="Value in plain terms")
    shap: float = Field(description="Contribution to log(value / market index)")
    effect_pct: float = Field(description="exp(shap) - 1, in percent")


class Drivers(BaseModel):
    context_shap: float
    performance_shap: float
    context_effect_pct: float
    performance_effect_pct: float
    driver: Literal["context", "performance", "mixed"]


class Explanation(BaseModel):
    player_id: int
    name: str
    base_value: float
    prediction: float
    market_index: float
    predicted_value: float
    actual_value: float
    contributions: list[Factor]
    top_factors: list[Factor]
    drivers: Drivers


class PredictRequest(BaseModel):
    """Any subset of the model's features; missing ones are treated as unknown (NaN),
    exactly as in training."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "age": 22.5,
                    "league": "NL1",
                    "position": "Centre-Forward",
                    "position_group": "Attack",
                    "minutes": 2400,
                    "goals": 14,
                    "assists": 6,
                    "ga_per90": 0.75,
                    "league_index_rel": -0.2,
                },
                "actual_value": 4_000_000,
            }
        }
    )

    features: dict[str, float | int | str | None]
    actual_value: float | None = Field(default=None, gt=0)


class PredictResponse(BaseModel):
    prediction: float
    predicted_value: float
    market_index: float
    predicted_range: ValueRange | None = None
    top_factors: list[Factor]
    actual_value: float | None = None
    undervalued_score_raw: float | None = None


class Report(BaseModel):
    player_id: int
    report: str
    model: str
    prompt_version: str
    generated_at: str
    ai_generated: Literal[True]
    disclaimer: str
    cached: bool
