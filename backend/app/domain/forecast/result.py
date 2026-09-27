"""预测结果的 DTO。精度必须带口径（caliber），裸数字没有意义。"""

from __future__ import annotations

from datetime import date, datetime
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
    explain: "ForecastExplain | None" = None  # 仅 options.explain=true 时返回


class ExplainWeatherPoint(BaseModel):
    """气象预报原始点（逐小时）。"""

    ts: datetime
    ghi_wm2: float
    dni_wm2: float
    dhi_wm2: float
    temperature_c: float | None = None


class ExplainBacktest(BaseModel):
    """融合校准回测明细。"""

    dates: list[date]
    sum_measured_kwh: float
    sum_physics_kwh: float
    factor: float


class ForecastExplain(BaseModel):
    weather: list[ExplainWeatherPoint] | None = None
    backtest: ExplainBacktest | None = None


ForecastProvenance.model_rebuild()


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
