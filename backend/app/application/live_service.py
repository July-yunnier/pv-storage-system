"""实时看板编排：当前快照 + 模拟数据源写入。不依赖 FastAPI。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

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
    """生成（或补齐）今日 00:00 至当前时刻的模拟测量数据。幂等：重跑覆盖同日同时槽。

    回放驱动：站点配置了 replay_config 时，光伏/负荷改用回放指针（cursor_date）
    指向的真实历史曲线（public_replay，逐日推进、窗口循环），而非合成日型；
    SOC/并网仍按当日计划执行推算。
    """
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

    pv_override: list[float] | None = None
    load_override: list[float] | None = None
    cfg = db.get_replay_config(site_id)
    if cfg is not None:
        today_iso = now_local.date().isoformat()
        window_start = date.fromisoformat(cfg["window_start"])
        window_end = date.fromisoformat(cfg["window_end"])
        cursor = date.fromisoformat(cfg["cursor_date"]) if cfg["cursor_date"] else None
        if cfg["cursor_for"] != today_iso:
            # 新的一天（或首次）：推进指针；跳过数据不完整的日子，窗口循环
            cursor = window_start if cursor is None else cursor + timedelta(days=1)
            for _ in range((window_end - window_start).days + 1):
                if cursor > window_end:
                    cursor = window_start
                pv_day = db.get_measurements_for_date(
                    site_id, "pv_power", cursor, site.location.timezone
                )
                if len(pv_day) >= 96:
                    break
                cursor += timedelta(days=1)
            else:
                cursor = None
            if cursor is not None:
                db.update_replay_cursor(site_id, cursor.isoformat(), today_iso)
        if cursor is not None:
            pv_day = db.get_measurements_for_date(
                site_id, "pv_power", cursor, site.location.timezone
            )
            load_day = db.get_measurements_for_date(
                site_id, "load_power", cursor, site.location.timezone
            )
            if len(pv_day) >= 96 and len(load_day) >= 96:
                pv_override = [p.value or 0.0 for p in pv_day[:96]]
                load_override = [p.value or 0.0 for p in load_day[:96]]

    generated = simulate_live_points(
        site, now_local, plan_points, pv_override, load_override
    )
    replay_on = pv_override is not None
    inserted: dict[str, int] = {}
    for data_type, series in generated.items():
        if not series:
            inserted[data_type] = 0
            continue
        unit = "kW" if data_type in _POWER_TYPES else "fraction"
        # 血统口径：回放模式只有光伏/负荷来自真实回放；SOC/并网始终是计划推算
        replayed = replay_on and data_type in ("pv_power", "load_power")
        upload = MeasurementUpload(
            data_type=data_type,  # type: ignore[arg-type]
            unit=unit,
            resolution_minutes=SLOT_MINUTES,
            source="public_replay" if replayed else "simulator",
            points=[
                MeasurementPoint(
                    ts=ts,
                    value=value,
                    quality="measured" if replayed else "estimated",
                )
                for ts, value in series
            ],
        )
        inserted[data_type] = db.insert_measurements(site_id, upload)

    current_slot = (now_local.hour * 60 + now_local.minute) // SLOT_MINUTES
    return LiveSimulateResponse(
        site_id=site_id, inserted=inserted, up_to_slot=current_slot
    )


def auto_extend_simulated(db: Database) -> list[str]:
    """模拟数据源自动续传：每分钟巡检，把"在用模拟数据"的站点补到当前时刻。

    只碰 simulator 源或配置了回放映射（replay_config）的站点——
    接真实 EMS/CSV 数据的站点永远不会被改写。
    """
    extended: list[str] = []
    for site in db.list_sites():
        latest = db.get_latest_measurement_with_source(site.site_id, "pv_power")
        is_sim = latest is not None and latest[2] == "simulator"
        is_replay = db.get_replay_config(site.site_id) is not None
        if not (is_sim or is_replay):
            continue
        tz = _zone(site.location.timezone)
        now_local = datetime.now(tz)
        # 回放站点首次推进时 latest 可能为空或停在历史日期，直接补到现在
        latest_local = latest[0].astimezone(tz) if latest is not None else None
        if latest_local is None or latest_local.date() < now_local.date() or (
            (now_local.hour * 60 + now_local.minute) // SLOT_MINUTES
            > (latest_local.hour * 60 + latest_local.minute) // SLOT_MINUTES
        ):
            try:
                simulate_live(db, site.site_id)
                extended.append(site.site_id)
            except Exception:
                continue  # 单站失败不影响其他站
    return extended
