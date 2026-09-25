"""光伏超短期统计法：pred(t+h) = clip(最近 N 天同刻均值 + alpha × 当前偏差, 0, Cap)。

MVP 日前场景下目标日整体在 issued_at 之后，"当前偏差"不可用，偏差项为 0
（即目标日预测 = 同刻均值）；alpha/w_dev 保留在签名里，供日内滚动更新使用。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo
from statistics import mean

from app.domain.forecast.result import (
    AccuracyCaliber,
    ForecastAccuracy,
    ForecastModel,
    ForecastPoint,
    TargetForecast,
)
from app.domain.forecast.stats import (
    SLOT_MINUTES,
    SLOTS_PER_DAY,
    daily_series,
    history_dates,
    residual_quantiles,
    rmse,
    slot_means,
    walk_forward_residuals,
)


def predict_pv_diurnal(
    history: list[tuple[datetime, float]],
    target_date: date,
    cap_kw: float,
    tz: tzinfo,
    n_days: int = 12,
    alpha: float = 0.25,
    w_dev: int = 2,
) -> TargetForecast:
    days = daily_series(history, tz)
    dates = history_dates(days, target_date, n_days, "pv_power")
    values = [min(max(value, 0.0), cap_kw) for value in slot_means(days, dates)]

    residuals = walk_forward_residuals(days, dates, n_days, cap_kw)
    accuracy = None
    p10: list[float] | None = None
    p90: list[float] | None = None
    if residuals:
        error = rmse(residuals)
        accuracy = ForecastAccuracy(
            metric="one_minus_rmse_over_cap",
            value=round(1 - error / cap_kw, 4),
            caliber=AccuracyCaliber(
                normalization="cap",
                cap_kw=cap_kw,
                includes_nighttime=True,
                baseline="persistence",
            ),
            evaluated_on=f"backtest:last_{len(dates)}d",
        )
        q10, q90 = residual_quantiles(residuals)
        p10 = [round(min(max(value + q10, 0.0), cap_kw), 6) for value in values]
        p90 = [round(min(max(value + q90, 0.0), cap_kw), 6) for value in values]

    points = _points_for_day(target_date, values, tz)
    return TargetForecast(
        target="pv_power",
        unit="kW",
        model=ForecastModel(name="pv_diurnal_stats", version="v1"),
        points=points,
        p10=p10,
        p90=p90,
        accuracy=accuracy,
        notes=(
            f"同刻均值法，使用最近 {len(dates)} 天历史；"
            f"偏差延续参数 alpha={alpha}、w_dev={w_dev} 留待日内滚动更新场景"
        ),
    )


def _points_for_day(target_date: date, values: list[float], tz: tzinfo) -> list[ForecastPoint]:
    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    return [
        ForecastPoint(
            ts=day_start + timedelta(minutes=SLOT_MINUTES * slot),
            value=round(values[slot], 6),
        )
        for slot in range(SLOTS_PER_DAY)
    ]


def predict_pv_physics(
    weather: list,
    target_date: date,
    pv,
    lat: float,
    lon: float,
    tz: tzinfo,
    provider_name: str,
) -> TargetForecast:
    """纯物理预测：辐射预报 → pvlib 转换链。无需历史数据（冷启动可用）。"""
    from app.domain.forecast.pv_physics import day_power_values

    values = day_power_values(weather, pv, lat, lon, target_date, tz)
    return TargetForecast(
        target="pv_power",
        unit="kW",
        model=ForecastModel(name="pvlib_pvwatts", version="v1"),
        points=_points_for_day(target_date, values, tz),
        accuracy=None,
        notes=(
            f"物理模型：{provider_name} 辐射预报（逐小时线性插值到 15 分钟）"
            f"经 pvlib 转换链（Hay-Davies 转置 + Faiman 温度 + PVWatts 直流）；"
            "精度待实测回测后标注"
        ),
    )


def predict_pv_intraday(
    history: list[tuple[datetime, float]],
    issued_local: datetime,
    cap_kw: float,
    tz: tzinfo,
    n_days: int = 12,
    alpha: float = 0.25,
    w_dev: int = 2,
) -> TargetForecast:
    """日内滚动修正：用今日已发生实测的偏差，修正今日剩余时段的同刻均值基线。

    兄弟仓库超短期算法的正确定位：pred(t) = clip(同刻均值 + alpha × 当前偏差, 0, Cap)。
    """
    days = daily_series(history, tz)
    today = issued_local.date()
    base_dates = history_dates(days, today, n_days, "pv_power")
    base = slot_means(days, base_dates)
    today_measured = days.get(today, {})

    current_slot = issued_local.hour * 60 // SLOT_MINUTES
    recent = [
        (slot, today_measured[slot * SLOT_MINUTES])
        for slot in range(max(0, current_slot - w_dev + 1), current_slot + 1)
        if slot * SLOT_MINUTES in today_measured
    ]
    deviation = 0.0
    if recent:
        deviation = mean([value - base[slot] for slot, value in recent])

    values: list[float] = []
    for slot in range(current_slot + 1, SLOTS_PER_DAY):
        values.append(min(max(base[slot] + alpha * deviation, 0.0), cap_kw))

    day_start = datetime.combine(today, time.min, tzinfo=tz)
    points = [
        ForecastPoint(
            ts=day_start + timedelta(minutes=SLOT_MINUTES * (current_slot + 1 + index)),
            value=round(value, 6),
        )
        for index, value in enumerate(values)
    ]
    return TargetForecast(
        target="pv_power",
        unit="kW",
        model=ForecastModel(name="pv_intraday_deviation", version="v1"),
        points=points,
        accuracy=None,
        notes=(
            f"日内滚动修正：基线=最近 {len(base_dates)} 天同刻均值，"
            f"偏差={deviation:.2f} kW（最近 {len(recent)} 个实测槽位）按 alpha={alpha} 延续；"
            "仅覆盖今日剩余时段"
        ),
    )
