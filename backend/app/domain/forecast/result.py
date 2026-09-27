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


class ForecastProvenance(BaseModel):
    """预测溯源：数据从哪来、怎么校准的。供前端"计算路线"展示。"""

    provider: str | None = None  # 气象数据源（open_meteo / mock）
    fetched_at: datetime | None = None  # 气象数据拉取时间（UTC）
    factor: float | None = None  # 融合校准系数（实测/物理回测，仅 blend）
    window: str | None = None  # 使用的数据窗口描述


class TargetForecast(BaseModel):
    target: Literal["pv_power", "load_power", "price"]
    unit: str
    model: ForecastModel
    points: list[ForecastPoint]
    p10: list[float] | None = None
    p90: list[float] | None = None
    accuracy: ForecastAccuracy | None = None
    provenance: ForecastProvenance | None = None
    notes: str = ""


class ForecastResponse(BaseModel):
    site_id: str
    issued_at: datetime
    resolution_minutes: int
    results: list[TargetForecast]
