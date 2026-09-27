"""实时看板编排：当前快照 + 模拟数据源写入。不依赖 FastAPI。"""

from __future__ import annotations

from datetime import datetime, time, timedelta

from app.api import errors
from app.domain.live_sim import SLOT_MINUTES, simulate_live_points
from app.domain.models import (
    LiveLatestPoint,
    LivePriceNow,
    LiveSimulateResponse,
    LiveStatus,
    LiveTodayPlan,
    MeasurementPoint,
    MeasurementUpload,
)
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database, _zone

_POWER_TYPES = {"pv_power", "load_power", "grid_power"}


def _today_latest(
    db: Database, site_id: str, data_type: str, today, tz
) -> LiveLatestPoint | None:
    """今日最新非 None 点（昨日及更早视为无实时数据）。"""
    point = db.get_latest_measurement_point(site_id, data_type, str(tz))
    if point is None:
        return None
    ts, value = point
    if ts.date() != today:
        return None
    return LiveLatestPoint(ts=ts, value=value)


def _coverage(db: Database, site_id: str, data_type: str, today, tz) -> int:
    rows = db.get_measurement_rows_for_date(site_id, data_type, today, str(tz))
    return sum(1 for row in rows if row["value"] is not None)


def get_live(db: Database, site_id: str) -> LiveStatus:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    tz = _zone(site.location.timezone)
    now_local = datetime.now(tz)
    today = now_local.date()

    tariff = db.get_tariff(site.tariff_id or "T-DEFAULT")
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {site.tariff_id}")
    provider = TouProvider(tariff)
    day_start = datetime.combine(today, time.min, tzinfo=tz)
    points = provider.points(day_start, 24 * 60 // SLOT_MINUTES, SLOT_MINUTES)
    current_slot = min((now_local.hour * 60 + now_local.minute) // SLOT_MINUTES, len(points) - 1)
    now_point = points[current_slot]
    next_change_at = None
    next_tier = None
    for index in range(current_slot + 1, len(points)):
        if points[index].tier != now_point.tier:
            next_change_at = day_start + timedelta(minutes=SLOT_MINUTES * index)
            next_tier = points[index].tier
            break

    today_plan = None
    for run in db.list_dispatch_runs(site_id):
        if run.date == today:
            today_plan = LiveTodayPlan(
                run_id=run.run_id,
                expected_benefit_yuan=run.expected_benefit_yuan,
                created_at=run.created_at,
            )
            break

    expected = 24 * 60 // SLOT_MINUTES
    return LiveStatus(
        site_id=site_id,
        now_local=now_local,
        latest_pv=_today_latest(db, site_id, "pv_power", today, tz),
        latest_load=_today_latest(db, site_id, "load_power", today, tz),
        latest_soc=_today_latest(db, site_id, "storage_soc", today, tz),
        latest_grid=_today_latest(db, site_id, "grid_power", today, tz),
        price_now=LivePriceNow(
            price=now_point.price,
            tier=now_point.tier,
            next_change_at=next_change_at,
            next_tier=next_tier,
        ),
        today_plan=today_plan,
        pv_points_today=_coverage(db, site_id, "pv_power", today, tz),
        load_points_today=_coverage(db, site_id, "load_power", today, tz),
        expected_points_today=expected,
    )


def simulate_live(db: Database, site_id: str) -> LiveSimulateResponse:
    """生成（或补齐）今日 00:00 至当前时刻的模拟测量数据。幂等：重跑覆盖同日同时槽。"""
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    tz = _zone(site.location.timezone)
    now_local = datetime.now(tz)

    plan_points = None
    for run in db.list_dispatch_runs(site_id):
        if run.date == now_local.date():
            detail = db.get_dispatch_run(run.run_id)
            if detail is not None:
                plan_points = detail.schedule
            break

    generated = simulate_live_points(site, now_local, plan_points)
    inserted: dict[str, int] = {}
    for data_type, series in generated.items():
        if not series:
            inserted[data_type] = 0
            continue
        unit = "kW" if data_type in _POWER_TYPES else "fraction"
        upload = MeasurementUpload(
            data_type=data_type,  # type: ignore[arg-type]
            unit=unit,
            resolution_minutes=SLOT_MINUTES,
            source="simulator",
            points=[
                MeasurementPoint(ts=ts, value=value, quality="measured")
                for ts, value in series
            ],
        )
        inserted[data_type] = db.insert_measurements(site_id, upload)

    current_slot = (now_local.hour * 60 + now_local.minute) // SLOT_MINUTES
    return LiveSimulateResponse(
        site_id=site_id, inserted=inserted, up_to_slot=current_slot
    )
