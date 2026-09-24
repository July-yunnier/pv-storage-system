"""数据质量报告：按站点时区当日应有的时间网格核对测量点完整性。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Literal

from pydantic import BaseModel

from app.infrastructure.db import _zone

DEFAULT_RESOLUTION_MINUTES = 15


class MissingRange(BaseModel):
    start: datetime
    end: datetime


class DataQualityReport(BaseModel):
    site_id: str
    data_type: str
    date: date
    resolution_minutes: int
    expected_points: int
    measured: int
    estimated: int
    invalid: int
    missing: int
    completeness: float
    missing_ranges: list[MissingRange]
    status: Literal["complete", "partial", "empty"]


def build_data_quality_report(
    site_id: str,
    data_type: str,
    target_date: date,
    timezone_name: str,
    rows: list[dict[str, Any]],
) -> DataQualityReport:
    """rows 为 get_measurement_rows_for_date 的返回（ts 已转站点时区）。"""
    tz = _zone(timezone_name)
    resolutions = {row["resolution_minutes"] for row in rows}
    resolution = min(resolutions) if resolutions else DEFAULT_RESOLUTION_MINUTES
    expected = 24 * 60 // resolution

    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    by_ts = {row["ts"]: row for row in rows}

    measured = estimated = invalid = 0
    missing_slots: list[datetime] = []
    for index in range(expected):
        slot = day_start + timedelta(minutes=resolution * index)
        row = by_ts.get(slot)
        if row is None or row["value"] is None or row["quality"] == "missing":
            missing_slots.append(slot)
            continue
        if row["quality"] == "measured":
            measured += 1
        elif row["quality"] == "estimated":
            estimated += 1
        elif row["quality"] == "invalid":
            invalid += 1
        else:
            missing_slots.append(slot)

    missing = len(missing_slots)
    if missing == 0:
        status: Literal["complete", "partial", "empty"] = "complete"
    elif measured == 0:
        status = "empty"
    else:
        status = "partial"

    return DataQualityReport(
        site_id=site_id,
        data_type=data_type,
        date=target_date,
        resolution_minutes=resolution,
        expected_points=expected,
        measured=measured,
        estimated=estimated,
        invalid=invalid,
        missing=missing,
        completeness=round((measured + estimated) / expected, 4),
        missing_ranges=_merge_ranges(missing_slots, resolution),
        status=status,
    )


def _merge_ranges(missing_slots: list[datetime], resolution: int) -> list[MissingRange]:
    ranges: list[MissingRange] = []
    step = timedelta(minutes=resolution)
    for slot in missing_slots:
        if ranges and slot - ranges[-1].end == step:
            ranges[-1].end = slot
        else:
            ranges.append(MissingRange(start=slot, end=slot))
    return ranges
