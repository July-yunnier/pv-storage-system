from __future__ import annotations

from datetime import date

import pytest

from app.domain.shift_calendar import MES_LABELS, ShiftCalendar

# 2026-09-07 是周一，2026-09-12 是周六
MONDAY = date(2026, 9, 7)
SATURDAY = date(2026, 9, 12)


def test_default_calendar_matches_manifest_mapping() -> None:
    """默认两班制标签 id 与训练工件 manifest.mes_mapping 一致。"""
    assert MES_LABELS == {
        "day_shift": 1,
        "evening_shift": 2,
        "high_energy": 3,
        "maintenance": 4,
        "night_base": 5,
        "startup": 6,
        "weekend_base": 7,
    }


def test_default_workday_segments() -> None:
    cal = ShiftCalendar()
    labels = cal.labels_for_day(MONDAY)
    assert len(labels) == 96
    assert labels[0] == MES_LABELS["night_base"]  # 00:00
    assert labels[6 * 4] == MES_LABELS["startup"]  # 06:00 开机
    assert labels[8 * 4] == MES_LABELS["day_shift"]  # 08:00 白班
    assert labels[13 * 4] == MES_LABELS["day_shift"]  # 13:00 仍白班（默认无高耗能）
    assert labels[14 * 4] == MES_LABELS["day_shift"]
    assert labels[18 * 4] == MES_LABELS["evening_shift"]  # 18:00 晚班
    assert labels[22 * 4] == MES_LABELS["night_base"]  # 22:00 夜间


def test_weekend_all_weekend_base() -> None:
    cal = ShiftCalendar()
    assert set(cal.labels_for_day(SATURDAY)) == {MES_LABELS["weekend_base"]}


def test_high_energy_weekday() -> None:
    cal = ShiftCalendar(high_energy_weekdays=[2])  # 周三
    labels = cal.labels_for_day(date(2026, 9, 9))  # 周三
    assert labels[14 * 4] == MES_LABELS["high_energy"]
    assert labels[17 * 4 + 3] == MES_LABELS["high_energy"]
    assert labels[13 * 4] == MES_LABELS["day_shift"]  # 14 点前不受影响
    # 其他工作日不触发
    assert cal.labels_for_day(MONDAY)[14 * 4] == MES_LABELS["day_shift"]


def test_maintenance_date_overrides_day_shift() -> None:
    cal = ShiftCalendar(maintenance_dates=[MONDAY])
    labels = cal.labels_for_day(MONDAY)
    assert labels[10 * 4] == MES_LABELS["maintenance"]
    assert labels[6 * 4] == MES_LABELS["startup"]  # 检修日开机时段不变
    assert labels[20 * 4] == MES_LABELS["evening_shift"]  # 晚班不变
    # 周末配置 maintenance_date 不生效（周末优先）
    cal2 = ShiftCalendar(maintenance_dates=[SATURDAY])
    assert set(cal2.labels_for_day(SATURDAY)) == {MES_LABELS["weekend_base"]}


def test_custom_workdays() -> None:
    cal = ShiftCalendar(workdays=[0, 1, 2, 3, 4, 5])  # 周六也开工
    labels = cal.labels_for_day(SATURDAY)
    assert labels[10 * 4] == MES_LABELS["day_shift"]


def test_invalid_ordering_rejected() -> None:
    with pytest.raises(ValueError):
        ShiftCalendar(workday_start_hour=20, evening_shift_end_hour=18)
    with pytest.raises(ValueError):
        ShiftCalendar(workdays=[7])


def test_slot_out_of_range() -> None:
    with pytest.raises(ValueError):
        ShiftCalendar().label_for_slot(MONDAY, 96)
