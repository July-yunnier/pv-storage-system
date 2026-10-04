"""负荷预测：同刻均值法，无容量上限，下限 0。精度归一化用均值功率。"""

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
from app.domain.forecast.load_advanced import OnlineResidualKalman
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


def predict_load_profile(
    history: list[tuple[datetime, float]],
    target_date: date,
    tz: tzinfo,
    n_days: int = 12,
) -> TargetForecast:
    days = daily_series(history, tz)
    dates = history_dates(days, target_date, n_days, "load_power")
    values = [max(value, 0.0) for value in slot_means(days, dates)]

    residuals = walk_forward_residuals(days, dates, n_days, cap=None)
    accuracy = None
    p10: list[float] | None = None
    p90: list[float] | None = None
    if residuals:
        actuals = [
            days[d][minute]
            for d in dates
            for minute in range(0, 24 * 60, SLOT_MINUTES)
            if minute in days[d]
        ]
        mean_power = mean(actuals) if actuals else 0.0
        error = rmse(residuals)
        accuracy = ForecastAccuracy(
            metric="one_minus_rmse_over_mean",
            value=round(1 - error / mean_power, 4) if mean_power > 0 else 0.0,
            caliber=AccuracyCaliber(
                normalization="mean_power",
                cap_kw=round(mean_power, 6),
                includes_nighttime=True,
                baseline="diurnal_persistence",
            ),
            evaluated_on=f"backtest:last_{len(dates)}d",
        )
        q10, q90 = residual_quantiles(residuals)
        p10 = [round(max(value + q10, 0.0), 6) for value in values]
        p90 = [round(max(value + q90, 0.0), 6) for value in values]

    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    points = [
        ForecastPoint(
            ts=day_start + timedelta(minutes=SLOT_MINUTES * slot),
            value=round(values[slot], 6),
        )
        for slot in range(SLOTS_PER_DAY)
    ]
    return TargetForecast(
        target="load_power",
        unit="kW",
        model=ForecastModel(name="load_diurnal_mean", version="v1"),
        points=points,
        p10=p10,
        p90=p90,
        accuracy=accuracy,
        notes=f"同刻均值法，使用最近 {len(dates)} 天历史",
    )


def predict_load_intraday(
    history: list[tuple[datetime, float]],
    issued_local: datetime,
    tz: tzinfo,
    n_days: int = 12,
    q: float = 1e-4,
    r: float = 1e-2,
    decay: float = 0.95,
) -> TargetForecast:
    """今日负荷滚动预测：同刻均值基线 + 已发生实测残差卡尔曼修正。"""
    days = daily_series(history, tz)
    today = issued_local.astimezone(tz).date()
    dates = history_dates(days, today, n_days, "load_power")
    baseline = [max(value, 0.0) for value in slot_means(days, dates)]
    today_measured = days.get(today, {})
    current_slot = (issued_local.hour * 60 + issued_local.minute) // SLOT_MINUTES

    kf = OnlineResidualKalman(q=q, r=r, decay=decay)
    observed_count = 0
    for slot in range(0, min(current_slot + 1, SLOTS_PER_DAY)):
        minute = slot * SLOT_MINUTES
        if minute not in today_measured:
            continue
        kf.update(today_measured[minute] - baseline[slot])
        observed_count += 1

    corrected = kf.correct_remaining(baseline, min(current_slot, SLOTS_PER_DAY - 1))
    day_start = datetime.combine(today, time.min, tzinfo=tz)
    points = [
        ForecastPoint(
            ts=day_start + timedelta(minutes=SLOT_MINUTES * slot),
            value=round(corrected[slot], 6),
        )
        for slot in range(current_slot + 1, SLOTS_PER_DAY)
    ]
    return TargetForecast(
        target="load_power",
        unit="kW",
        model=ForecastModel(name="load_intraday_kalman", version="v1"),
        points=points,
        notes=(
            f"日内负荷修正：最近 {len(dates)} 天同刻均值为基线，"
            f"使用今日 {observed_count} 个已发生槽位更新残差状态，"
            f"未来影响按 decay={decay} 衰减；仅覆盖今日剩余时段"
        ),
    )
