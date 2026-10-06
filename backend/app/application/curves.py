"""曲线聚合：把 pv/load 测量和 TOU 电价对齐到站点时区的 15 分钟网格。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Iterable

from pydantic import BaseModel

from app.domain.models import MeasurementRead, TierName
from app.domain.tariff import TouProvider
from app.infrastructure.db import _zone

RESOLUTION_MINUTES = 15
MAX_SPAN_DAYS = 31


class CurvePoint(BaseModel):
    ts: datetime
    pv_power_kw: float | None = None
    load_power_kw: float | None = None
    net_load_kw: float | None = None
    price: float
    tier: TierName | None = None


class CurvesResponse(BaseModel):
    site_id: str
    start: date
    end: date
    resolution_minutes: int
    points: list[CurvePoint]


def _usable_value(point: MeasurementRead | None) -> float | None:
    if point is None or point.value is None:
        return None
    if point.quality in ("missing", "invalid"):
        return None
    return point.value


def build_curves(
    site_id: str,
    start: date,
    end: date,
    timezone_name: str,
    pv_points: Iterable[MeasurementRead],
    load_points: Iterable[MeasurementRead],
    provider: TouProvider,
) -> CurvesResponse:
    tz = _zone(timezone_name)
    pv_by_ts = {point.ts: point for point in pv_points}
    load_by_ts = {point.ts: point for point in load_points}

    points: list[CurvePoint] = []
    cursor = datetime.combine(start, time.min, tzinfo=tz)
    last = datetime.combine(end, time.min, tzinfo=tz)
    while cursor <= last:
        for slot in range(24 * 60 // RESOLUTION_MINUTES):
            ts = cursor + timedelta(minutes=RESOLUTION_MINUTES * slot)
            pv = _usable_value(pv_by_ts.get(ts))
            load = _usable_value(load_by_ts.get(ts))
            net_load = round(load - pv, 6) if pv is not None and load is not None else None
            price_point = provider.price_at(ts)
            points.append(
                CurvePoint(
                    ts=ts,
                    pv_power_kw=pv,
                    load_power_kw=load,
                    net_load_kw=net_load,
                    price=price_point.price,
                    tier=price_point.tier,
                )
            )
        cursor += timedelta(days=1)
    return CurvesResponse(
        site_id=site_id,
        start=start,
        end=end,
        resolution_minutes=RESOLUTION_MINUTES,
        points=points,
    )
