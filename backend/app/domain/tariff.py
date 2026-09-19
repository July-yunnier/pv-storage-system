from __future__ import annotations

from datetime import datetime, time

from app.domain.models import PricePoint, TariffRead, TariffTier


def _minutes(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def _contains(tier: TariffTier, minute_of_day: int) -> bool:
    for hour_range in tier.hours:
        start_text, end_text = hour_range.split("-")
        start = _minutes(start_text)
        end = _minutes(end_text)
        if start < end and start <= minute_of_day < end:
            return True
        if start > end and (minute_of_day >= start or minute_of_day < end):
            return True
        if start == end:
            raise ValueError(f"tariff range cannot have same start and end: {hour_range}")
    return False


class TouProvider:
    """将 TOU 电价模板转换为统一价格点。时间段使用左闭右开。"""

    def __init__(self, tariff: TariffRead) -> None:
        if tariff.mode != "tou":
            raise ValueError(f"TouProvider does not support mode: {tariff.mode}")
        self.tariff = tariff

    def tier_at(self, ts: datetime) -> TariffTier:
        minute = ts.hour * 60 + ts.minute
        matches = [tier for tier in self.tariff.tiers if _contains(tier, minute)]
        if len(matches) != 1:
            raise ValueError(
                f"tariff must match exactly one tier at {ts.isoformat()}, matched {len(matches)}"
            )
        return matches[0]

    def price_at(self, ts: datetime) -> PricePoint:
        tier = self.tier_at(ts)
        return PricePoint(
            ts=ts,
            price=round(tier.price + self.tariff.transmission_and_fund, 6),
            energy_price=tier.price,
            tier=tier.name,
        )

    def points(self, start: datetime, periods: int, resolution_minutes: int = 15) -> list[PricePoint]:
        from datetime import timedelta

        if periods <= 0:
            raise ValueError("periods must be positive")
        return [
            self.price_at(start + timedelta(minutes=index * resolution_minutes))
            for index in range(periods)
        ]

    def validate_day(self, resolution_minutes: int = 15) -> None:
        if 1440 % resolution_minutes != 0:
            raise ValueError("resolution_minutes must divide 24 hours")
        from datetime import datetime, timezone, timedelta

        start = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=8)))
        self.points(start, 1440 // resolution_minutes, resolution_minutes)
