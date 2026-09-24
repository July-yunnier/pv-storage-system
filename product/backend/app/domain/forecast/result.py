"""预测结果的 DTO。精度必须带口径（caliber），裸数字没有意义。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ForecastModel(BaseModel):
    name: str
    version: str


class AccuracyCaliber(BaseModel):
    normalization: str
    cap_kw: float
    includes_nighttime: bool
    spatial_scale: str = "site"
    baseline: str


class ForecastAccuracy(BaseModel):
    metric: str
    value: float
    caliber: AccuracyCaliber
    evaluated_on: str


class ForecastPoint(BaseModel):
    ts: datetime
    value: float


class TargetForecast(BaseModel):
    target: Literal["pv_power", "load_power", "price"]
    unit: str
    model: ForecastModel
    points: list[ForecastPoint]
    p10: list[float] | None = None
    p90: list[float] | None = None
    accuracy: ForecastAccuracy | None = None
    notes: str = ""


class ForecastResponse(BaseModel):
    site_id: str
    issued_at: datetime
    resolution_minutes: int
    results: list[TargetForecast]
