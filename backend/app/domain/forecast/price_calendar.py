"""中国电力日历：法定节假日、调休上班日、季节性尖峰月份。

电价预测的日历特征不是可选装饰——工商业电价的负荷侧响应、代理购电的
尖峰加价、节假日的深谷电价都挂在日历上。这里把"日历"收敛成纯函数，
线上推理与离线基准共用同一份定义，避免训练/推理不一致。

数据口径说明：
- 节假日来源为国务院办公厅年度放假通知，按年份硬编码；通知每年 11 月发布，
  新增年份请在此追加，缺失年份自动退化为"仅固定日 + 周末"。
- 调休上班日（周末补班）单独标出：这些天本质是工作日，不能按周末建模。
"""

from __future__ import annotations

from datetime import date, timedelta

# 法定节假日（含调休连休）。键为年份，值为 (开始日, 结束日) 闭区间。
# 不确定的年份请留空，宁可少标也不要错标——错标会把节假日效应污染到工作日。
HOLIDAY_RANGES: dict[int, list[tuple[str, str]]] = {
    2024: [
        ("2024-01-01", "2024-01-01"),
        ("2024-02-10", "2024-02-17"),
        ("2024-04-04", "2024-04-06"),
        ("2024-05-01", "2024-05-05"),
        ("2024-06-10", "2024-06-10"),
        ("2024-09-15", "2024-09-17"),
        ("2024-10-01", "2024-10-07"),
    ],
    2025: [
        ("2025-01-01", "2025-01-01"),
        ("2025-01-28", "2025-02-04"),
        ("2025-04-04", "2025-04-06"),
        ("2025-05-01", "2025-05-05"),
        ("2025-05-31", "2025-06-02"),
        ("2025-10-01", "2025-10-08"),
    ],
    2026: [
        ("2026-01-01", "2026-01-03"),
        ("2026-02-17", "2026-02-23"),
        ("2026-04-04", "2026-04-06"),
        ("2026-05-01", "2026-05-05"),
        ("2026-06-19", "2026-06-19"),
        ("2026-09-25", "2026-09-25"),
        ("2026-10-01", "2026-10-07"),
    ],
}

# 周末补班日（调休）。这些天按工作日建模。
MAKEUP_WORKDAYS: dict[int, tuple[str, ...]] = {
    2024: ("2024-02-04", "2024-02-18", "2024-04-07", "2024-04-28", "2024-05-11", "2024-09-14", "2024-09-29", "2024-10-12"),
    2025: ("2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11"),
    2026: (),
}

# 固定日节假日（无调休概念，仅用于跨年兜底）
FIXED_MONTH_DAYS: frozenset[tuple[int, int]] = frozenset(
    {(1, 1), (5, 1), (10, 1), (10, 2), (10, 3)}
)

# 电网代理购电的典型季节性尖峰月份（迎峰度夏/度冬）。用于"季节压力"特征，
# 不直接改价格——加价幅度由省级政策决定，模型只学"这个月通常更贵"。
SUMMER_PEAK_MONTHS = frozenset({7, 8})
WINTER_PEAK_MONTHS = frozenset({12, 1})


def _expand(ranges: list[tuple[str, str]]) -> set[date]:
    out: set[date] = set()
    for start_text, end_text in ranges:
        start = date.fromisoformat(start_text)
        end = date.fromisoformat(end_text)
        cursor = start
        while cursor <= end:
            out.add(cursor)
            cursor += timedelta(days=1)
    return out


_HOLIDAY_CACHE: dict[int, set[date]] = {}
_MAKEUP_CACHE: dict[int, set[date]] = {}


def holidays(year: int) -> set[date]:
    if year not in _HOLIDAY_CACHE:
        _HOLIDAY_CACHE[year] = _expand(HOLIDAY_RANGES.get(year, []))
    return _HOLIDAY_CACHE[year]


def makeup_workdays(year: int) -> set[date]:
    if year not in _MAKEUP_CACHE:
        _MAKEUP_CACHE[year] = {date.fromisoformat(d) for d in MAKEUP_WORKDAYS.get(year, ())}
    return _MAKEUP_CACHE[year]


def is_holiday(day: date) -> bool:
    if day in holidays(day.year) or day in holidays(day.year - 1) or day in holidays(day.year + 1):
        return True
    return (day.month, day.day) in FIXED_MONTH_DAYS


def is_makeup_workday(day: date) -> bool:
    return day in makeup_workdays(day.year)


def is_workday(day: date) -> bool:
    """真实营业日：调休补班算工作日，节假日不算（即使落在周一至周五）。"""
    if is_makeup_workday(day):
        return True
    if is_holiday(day):
        return False
    return day.weekday() < 5


def neighbor_flags(day: date) -> tuple[int, int]:
    """返回 (节前一日, 节后一日)。节前抢工、节后复产对负荷与电价有明显影响。"""
    return int(is_holiday(day + timedelta(days=1))), int(is_holiday(day - timedelta(days=1)))


def season_pressure(day: date) -> int:
    """迎峰度夏/度冬标记：-1 度冬，+1 度夏及以上，0 平季。"""
    if day.month in WINTER_PEAK_MONTHS:
        return -1
    if day.month in SUMMER_PEAK_MONTHS:
        return 1
    return 0
