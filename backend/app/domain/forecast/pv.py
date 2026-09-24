"""光伏超短期统计法：pred(t+h) = clip(最近 N 天同刻均值 + alpha × 当前偏差, 0, Cap)。

MVP 日前场景下目标日整体在 issued_at 之后，"当前偏差"不可用，偏差项为 0
（即目标日预测 = 同刻均值）；alpha/w_dev 保留在签名里，供日内滚动更新使用。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo

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

    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    points = [
        ForecastPoint(
            ts=day_start + timedelta(minutes=SLOT_MINUTES * slot),
            value=round(values[slot], 6),
        )
        for slot in range(SLOTS_PER_DAY)
    ]
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
