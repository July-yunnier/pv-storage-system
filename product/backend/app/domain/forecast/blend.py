"""物理预测 × 实测校准的融合（blend）。

职责分工：物理模型负责"明天的天气和昨天不同"（辐射预报驱动）；
实测校准负责"这个站和教科书不同"（组件积灰、朝向偏差、遮挡等系统性偏差）。

factor = Σ实测 / Σ物理回测（近 N 天白天槽位）
预测   = 物理预测 × factor
精度与不确定带来自回测残差，口径照常随结果返回。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, tzinfo

from app.domain.forecast.pv import _points_for_day
from app.domain.forecast.pv_physics import day_power_values
from app.domain.forecast.result import (
    AccuracyCaliber,
    ExplainBacktest,
    ExplainWeatherPoint,
    ForecastAccuracy,
    ForecastExplain,
    ForecastModel,
    ForecastProvenance,
    TargetForecast,
)
from app.domain.forecast.stats import (
    SLOT_MINUTES,
    SLOTS_PER_DAY,
    daily_series,
    history_dates,
    residual_quantiles,
    rmse,
)
from app.domain.models import PVConfig
from app.infrastructure.weather.base import WeatherPoint

DAYLIGHT_RATIO = 0.05  # 物理回测 > 5% 装机容量才算"白天槽位"，夜间不参与校准


def _calibration(
    weather_past: list[WeatherPoint],
    days_measured: dict[date, dict[int, float]],
    dates: list[date],
    pv: PVConfig,
    lat: float,
    lon: float,
    tz: tzinfo,
) -> tuple[float, list[float], float, float]:
    """返回 (校准系数, 回测残差列表, Σ实测, Σ物理回测)。"""
    sum_measured = 0.0
    sum_physics = 0.0
    for day in dates:
        physics = day_power_values(weather_past, pv, lat, lon, day, tz)
        for slot in range(SLOTS_PER_DAY):
            minute = slot * SLOT_MINUTES
            if minute not in days_measured[day]:
                continue
            if physics[slot] > DAYLIGHT_RATIO * pv.capacity_kwp:
                sum_measured += days_measured[day][minute]
                sum_physics += physics[slot]
    factor = sum_measured / sum_physics if sum_physics > 0 else 1.0

    residuals: list[float] = []
    for day in dates:
        physics = day_power_values(weather_past, pv, lat, lon, day, tz)
        for slot in range(SLOTS_PER_DAY):
            minute = slot * SLOT_MINUTES
            if minute not in days_measured[day]:
                continue
            pred = min(max(physics[slot] * factor, 0.0), pv.capacity_kwp)
            residuals.append(days_measured[day][minute] - pred)
    return factor, residuals, sum_measured, sum_physics


def blend_pv_forecast(
    weather_past: list[WeatherPoint],
    weather_future: list[WeatherPoint],
    measured_history: list[tuple[datetime, float]],
    pv: PVConfig,
    lat: float,
    lon: float,
    target_date: date,
    tz: tzinfo,
    n_days: int = 12,
    explain: bool = False,
) -> TargetForecast:
    """融合预测：物理预测 × 实测校准系数。历史不足 n_days 时抛 ValueError。"""
    days_measured = daily_series(measured_history, tz)
    dates = history_dates(days_measured, target_date, n_days, "pv_power")

    factor, residuals, sum_measured, sum_physics = _calibration(
        weather_past, days_measured, dates, pv, lat, lon, tz
    )

    physics_values = day_power_values(weather_future, pv, lat, lon, target_date, tz)
    values = [min(max(v * factor, 0.0), pv.capacity_kwp) for v in physics_values]

    accuracy = None
    p10: list[float] | None = None
    p90: list[float] | None = None
    if residuals:
        error = rmse(residuals)
        accuracy = ForecastAccuracy(
            metric="one_minus_rmse_over_cap",
            value=round(1 - error / pv.capacity_kwp, 4),
            caliber=AccuracyCaliber(
                normalization="cap",
                cap_kw=pv.capacity_kwp,
                includes_nighttime=True,
                baseline="pvlib_physics",
            ),
            evaluated_on=f"backtest:last_{len(dates)}d",
        )
        q10, q90 = residual_quantiles(residuals)
        p10 = [round(min(max(v + q10, 0.0), pv.capacity_kwp), 6) for v in values]
        p90 = [round(min(max(v + q90, 0.0), pv.capacity_kwp), 6) for v in values]

    return TargetForecast(
        target="pv_power",
        unit="kW",
        model=ForecastModel(name="pvlib_blend", version="v1"),
        points=_points_for_day(target_date, values, tz),
        p10=p10,
        p90=p90,
        accuracy=accuracy,
        provenance=ForecastProvenance(
            factor=round(factor, 4),
            window=f"回测{len(dates)}天 + 预报1天",
            explain=ForecastExplain(
                weather=[
                    ExplainWeatherPoint(
                        ts=p.ts,
                        ghi_wm2=p.ghi_wm2,
                        dni_wm2=p.dni_wm2,
                        dhi_wm2=p.dhi_wm2,
                        temperature_c=p.temperature_c,
                    )
                    for p in weather_future
                ],
                backtest=ExplainBacktest(
                    dates=dates,
                    sum_measured_kwh=round(sum_measured * SLOT_MINUTES / 60, 3),
                    sum_physics_kwh=round(sum_physics * SLOT_MINUTES / 60, 3),
                    factor=round(factor, 4),
                ),
            )
            if explain
            else None,
        ),
        notes=(
            f"融合预测：pvlib 物理链 × 实测校准系数 {factor:.3f}"
            f"（最近 {len(dates)} 天回测）；物理负责天气变化，校准负责站点偏差"
        ),
    )
