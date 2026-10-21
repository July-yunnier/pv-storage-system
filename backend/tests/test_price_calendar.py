from __future__ import annotations

from datetime import date

from app.domain.forecast import price_calendar as cal


def test_holiday_and_makeup_workday():
    assert cal.is_holiday(date(2025, 10, 1))
    assert cal.is_holiday(date(2025, 10, 8))
    assert not cal.is_holiday(date(2025, 10, 9))
    # 调休补班日是工作日，即便落在周末
    assert cal.is_makeup_workday(date(2025, 9, 28))
    assert cal.is_workday(date(2025, 9, 28))
    # 节假日即使落在周三也不是工作日
    assert not cal.is_workday(date(2025, 10, 1))


def test_fixed_holiday_fallback_for_unknown_year():
    # 2029 没有硬编码，固定日仍要生效
    assert cal.is_holiday(date(2029, 10, 1))
    assert cal.is_holiday(date(2029, 1, 1))
    assert not cal.is_holiday(date(2029, 3, 15))


def test_neighbor_flags_and_season_pressure():
    pre, post = cal.neighbor_flags(date(2025, 9, 30))
    assert pre == 1 and post == 0
    assert cal.season_pressure(date(2025, 8, 1)) == 1
    assert cal.season_pressure(date(2025, 1, 15)) == -1
    assert cal.season_pressure(date(2025, 4, 15)) == 0
