"""同刻均值法的公共统计工具。stdlib only，96 点序列不需要 numpy/pandas。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, tzinfo
from statistics import mean, quantiles

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
MIN_HISTORY_DAYS = 3


def daily_series(
    history: list[tuple[datetime, float]], tz: tzinfo
) -> dict[date, dict[int, float]]:
    """把 (ts, value) 历史按站点时区整理成 {日期: {分钟序号: 值}}。"""
    days: dict[date, dict[int, float]] = {}
    for ts, value in history:
        local = ts.astimezone(tz)
        days.setdefault(local.date(), {})[local.hour * 60 + local.minute] = value
    return days


def history_dates(
    days: dict[date, dict[int, float]], target_date: date, n_days: int, label: str
) -> list[date]:
    dates = [d for d in sorted(days) if d < target_date][-n_days:]
    if len(dates) < MIN_HISTORY_DAYS:
        raise ValueError(
            f"{label} 历史不足：需要至少 {MIN_HISTORY_DAYS} 天，当前 {len(dates)} 天"
        )
    return dates


def slot_means(days: dict[date, dict[int, float]], dates: list[date]) -> list[float]:
    """同刻均值：96 个槽位，某槽位所有历史天都无数据时取 0。"""
    means: list[float] = []
    for slot in range(SLOTS_PER_DAY):
        minute = slot * SLOT_MINUTES
        values = [days[d][minute] for d in dates if minute in days[d]]
        means.append(mean(values) if values else 0.0)
    return means


def walk_forward_residuals(
    days: dict[date, dict[int, float]],
    dates: list[date],
    n_days: int,
    cap: float | None,
) -> list[float]:
    """对每个历史天用其前 n_days 天回测，返回 actual - predicted 残差（预测已裁剪）。"""
    residuals: list[float] = []
    for index, day in enumerate(dates):
        train = dates[max(0, index - n_days) : index]
        if not train:
            continue
        predicted = slot_means(days, train)
        for slot in range(SLOTS_PER_DAY):
            minute = slot * SLOT_MINUTES
            if minute not in days[day]:
                continue
            pred = predicted[slot]
            if cap is not None:
                pred = min(max(pred, 0.0), cap)
            else:
                pred = max(pred, 0.0)
            residuals.append(days[day][minute] - pred)
    return residuals


def rmse(residuals: list[float]) -> float:
    return (mean([r * r for r in residuals])) ** 0.5


def residual_quantiles(residuals: list[float]) -> tuple[float, float]:
    """返回残差的 q10/q90。"""
    if len(residuals) < 2:
        return 0.0, 0.0
    qs = quantiles(residuals, n=10)
    return qs[0], qs[-1]
