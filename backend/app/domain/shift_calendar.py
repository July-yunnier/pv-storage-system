"""班次日历 → MES 排产标签映射。

标签体系与训练工件 manifest.mes_mapping 一致（0 保留给未知/新工厂）：
day_shift=1, evening_shift=2, high_energy=3, maintenance=4,
night_base=5, startup=6, weekend_base=7。

默认日历对应典型两班制：工作日 06-08 开机准备、08-18 白班
（高耗能工作日的 14-18 记 high_energy）、18-22 晚班、其余夜间基础负荷；
非工作日全天 weekend_base。站点可按真实排产覆盖边界与检修日。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator

MES_LABELS: dict[str, int] = {
    "day_shift": 1,
    "evening_shift": 2,
    "high_energy": 3,
    "maintenance": 4,
    "night_base": 5,
    "startup": 6,
    "weekend_base": 7,
}

SLOT_MINUTES = 15
SLOTS_PER_DAY = 96


class ShiftCalendar(BaseModel):
    """站点班次/排产日历；LoadConfig.mes_calendar 缺省时用默认值。"""

    startup_start_hour: int = Field(default=6, ge=0, le=23)
    workday_start_hour: int = Field(default=8, ge=0, le=23)
    high_energy_start_hour: int = Field(default=14, ge=0, le=23)
    day_shift_end_hour: int = Field(default=18, ge=1, le=24)
    evening_shift_end_hour: int = Field(default=22, ge=1, le=24)
    workdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # 周一=0
    high_energy_weekdays: list[int] = Field(default_factory=list)  # 高耗能工作日；空=无
    maintenance_dates: list[date] = Field(default_factory=list)  # 检修日：白班时段记 maintenance

    @model_validator(mode="after")
    def validate_ordering(self) -> "ShiftCalendar":
        if not (
            self.startup_start_hour
            < self.workday_start_hour
            <= self.high_energy_start_hour
            < self.day_shift_end_hour
            <= self.evening_shift_end_hour
        ):
            raise ValueError(
                "班次边界必须满足 startup < workday_start <= high_energy < day_end <= evening_end"
            )
        if any(d < 0 or d > 6 for d in self.workdays + self.high_energy_weekdays):
            raise ValueError("weekday 取值 0-6（周一=0）")
        return self

    def label_for_slot(self, day: date, slot: int) -> int:
        """单日单槽（0-95）的 MES 标签 id。"""
        if not 0 <= slot < SLOTS_PER_DAY:
            raise ValueError("slot out of range")
        hour = (slot * SLOT_MINUTES) // 60
        if day.weekday() not in self.workdays:
            return MES_LABELS["weekend_base"]
        if (
            day in self.maintenance_dates
            and self.workday_start_hour <= hour < self.day_shift_end_hour
        ):
            return MES_LABELS["maintenance"]
        if self.startup_start_hour <= hour < self.workday_start_hour:
            return MES_LABELS["startup"]
        if self.workday_start_hour <= hour < self.day_shift_end_hour:
            if hour >= self.high_energy_start_hour and day.weekday() in self.high_energy_weekdays:
                return MES_LABELS["high_energy"]
            return MES_LABELS["day_shift"]
        if self.day_shift_end_hour <= hour < self.evening_shift_end_hour:
            return MES_LABELS["evening_shift"]
        return MES_LABELS["night_base"]

    def labels_for_day(self, day: date) -> list[int]:
        return [self.label_for_slot(day, slot) for slot in range(SLOTS_PER_DAY)]

    def labels_for_range(self, start: date, days: int) -> list[int]:
        from datetime import timedelta

        return [
            label
            for offset in range(days)
            for label in self.labels_for_day(start + timedelta(days=offset))
        ]
