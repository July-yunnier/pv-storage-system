"""日内滚动修正：实测进来 → 偏差超限 → 对剩余时段重调度 → 存修正版计划。

触发策略（防抖动）：
- 死区：最近 2 个槽位光伏实测 vs 原计划平均偏差 ≤ 15% 不动
- 冷却：距上次修正不足 30 分钟不自动修（force 除外）
- 最早 08:00（第 32 槽）后才允许修正

修正不覆盖原计划：每次修正存一个 revision，审计与对比都靠它。
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta

from app.api import errors
from app.application import forecast_service
from app.application.dispatch_service import _forecast_as_measurements
from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.forecast.pv import predict_pv_intraday
from app.domain.models import (
    DispatchPoint,
    RollingRevisionRead,
    RollingState,
    ScheduleResponse,
)
from app.domain.tariff import TouProvider
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database, _zone

DEADBAND = 0.15
LOOKBACK_SLOTS = 2
MIN_SLOT = 32  # 08:00
COOLDOWN = timedelta(minutes=30)
SLOT_MINUTES = 15
SLOTS_PER_DAY = 96


def _slot_index(ts: datetime) -> int:
    return (ts.hour * 60 + ts.minute) // SLOT_MINUTES


def _today_measurements(
    db: Database, site_id: str, data_type: str, today: date, tz
) -> dict[int, float]:
    points = db.get_measurements_for_date(site_id, data_type, today, str(tz))
    return {
        _slot_index(p.ts): p.value
        for p in points
        if p.value is not None and p.quality in ("measured", "estimated")
    }


def _base_run(db: Database, site_id: str, today: date):
    for run in db.list_dispatch_runs(site_id):
        if run.date == today:
            return db.get_dispatch_run(run.run_id)
    return None


def _deviation(
    measured: dict[int, float], base_schedule: list[DispatchPoint], cap_kw: float
) -> tuple[float | None, int]:
    """最近槽位的光伏实测 vs 原计划平均相对偏差；返回 (偏差, 最新槽位)。"""
    if not base_schedule:
        return None, -1
    latest_slot = max(measured.keys(), default=-1)
    if latest_slot < 0:
        return None, latest_slot
    diffs: list[float] = []
    by_slot = {_slot_index(p.ts): p for p in base_schedule}
    for slot in range(max(0, latest_slot - LOOKBACK_SLOTS + 1), latest_slot + 1):
        plan_pv = by_slot.get(slot)
        actual = measured.get(slot)
        if plan_pv is None or actual is None:
            continue
        denom = max(plan_pv.pv_power_kw, 0.2 * cap_kw)
        diffs.append(abs(actual - plan_pv.pv_power_kw) / denom)
    if not diffs:
        return None, latest_slot
    return sum(diffs) / len(diffs), latest_slot


def get_rolling_state(db: Database, site_id: str) -> RollingState:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    tz = _zone(site.location.timezone)
    now_local = datetime.now(tz)
    today = now_local.date()

    base = _base_run(db, site_id, today)
    revisions_raw = db.get_rolling_plans(site_id, today)
    revisions = [
        RollingRevisionRead(
            revision=r["revision"],
            created_at=r["created_at"],
            initial_soc=r["initial_soc"],
            expected_benefit_yuan=r["expected_benefit_yuan"],
            trigger_reason=r["trigger_reason"],
        )
        for r in revisions_raw
    ]
    latest_schedule = None
    if revisions_raw:
        latest_schedule = [
            DispatchPoint.model_validate(p)
            for p in json.loads(revisions_raw[-1]["schedule_json"])
        ]

    measured_pv = _today_measurements(db, site_id, "pv_power", today, tz)
    deviation, _ = _deviation(
        measured_pv, base.schedule if base else [], site.pv.capacity_kwp
    )

    cooldown_until = None
    if revisions:
        cooldown_until = revisions[-1].created_at + COOLDOWN
        if cooldown_until.tzinfo is None:
            cooldown_until = cooldown_until.replace(tzinfo=tz)

    return RollingState(
        site_id=site_id,
        date=today,
        base_run_id=base.run_id if base else None,
        base_expected_benefit_yuan=base.expected_benefit_yuan if base else None,
        revisions=revisions,
        latest_schedule=latest_schedule,
        latest_revision=revisions[-1].revision if revisions else 0,
        cooldown_until=cooldown_until,
        deviation_now=round(deviation, 4) if deviation is not None else None,
        last_trigger_reason=revisions[-1].trigger_reason if revisions else None,
    )


def maybe_rolling_adjust(db: Database, site_id: str, force: bool = False) -> RollingState:
    """新数据驱动的一次修正判定。返回最新状态；未触发时 revisions 不新增。"""
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    tz = _zone(site.location.timezone)
    now_local = datetime.now(tz)
    today = now_local.date()

    base = _base_run(db, site_id, today)
    if base is None or not base.schedule:
        return get_rolling_state(db, site_id)

    measured_pv = _today_measurements(db, site_id, "pv_power", today, tz)
    deviation, latest_slot = _deviation(measured_pv, base.schedule, site.pv.capacity_kwp)
    revisions = db.get_rolling_plans(site_id, today)

    cooldown_ok = True
    if revisions:
        last_created = revisions[-1]["created_at"]
        if last_created.tzinfo is None:
            last_created = last_created.replace(tzinfo=tz)
        cooldown_ok = now_local >= last_created + COOLDOWN

    should_adjust = (
        force
        or (
            deviation is not None
            and deviation > DEADBAND
            and latest_slot >= MIN_SLOT
            and cooldown_ok
        )
    )
    if not should_adjust:
        return get_rolling_state(db, site_id)

    # 剩余时段输入
    day_start = datetime.combine(today, time.min, tzinfo=tz)
    tariff = db.get_tariff(site.tariff_id or "T-DEFAULT")
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {site.tariff_id}")
    price_all = TouProvider(tariff).points(day_start, SLOTS_PER_DAY, SLOT_MINUTES)

    # 光伏：日内滚动修正预测（今日实测偏差延续到剩余时段）
    history_pv = forecast_service._history(
        db, site, "pv_power", today + timedelta(days=1), forecast_service.N_DAYS
    )
    intraday = predict_pv_intraday(
        history_pv, now_local, site.pv.capacity_kwp, tz, forecast_service.N_DAYS
    )
    pv_points = _forecast_as_measurements("pv_power", intraday.points)

    # 负荷：历史同刻均值法全天预测；电价：TOU 全日排期。都按日内预测的时间戳对齐
    load_all = forecast_service.forecast_series(db, site, "load_power", today)
    pv_ts = {p.ts for p in intraday.points}
    load_points = _forecast_as_measurements(
        "load_power", [p for p in load_all if p.ts in pv_ts]
    )
    price_points = [pp for pp in price_all if pp.ts in pv_ts]

    # 初始 SOC：实测优先，退回原计划该槽位的 SOC
    soc_measured = _today_measurements(db, site_id, "storage_soc", today, tz)
    if soc_measured:
        initial_soc = soc_measured[max(soc_measured.keys())]
    else:
        by_slot = {_slot_index(p.ts): p for p in base.schedule}
        initial_soc = by_slot.get(latest_slot, base.schedule[0]).soc_end
    # Noisy telemetry can fall slightly outside operating bounds. Keep the
    # physical measurement in the database; constrain the planning initial
    # state explicitly and report the correction in the revision reason.
    measured_initial_soc = initial_soc
    initial_soc = min(max(initial_soc, site.storage.soc_min), site.storage.soc_max)

    if not pv_points or not load_points:
        return get_rolling_state(db, site_id)
    if len(pv_points) != len(load_points) or len(pv_points) != len(price_points):
        # 时间戳对齐失败时以最短序列兜底
        n = min(len(pv_points), len(load_points), len(price_points))
        pv_points, load_points, price_points = (
            pv_points[:n],
            load_points[:n],
            price_points[:n],
        )

    result: ScheduleResponse = RuleDispatchEngine().run(
        DispatchInputs(
            site=site,
            target_date=today,
            initial_soc=initial_soc,
            pv_points=pv_points,
            load_points=load_points,
            price_points=price_points,
            resolution_minutes=SLOT_MINUTES,
        )
    )

    reason = (
        f"手动触发"
        if force
        else f"光伏偏差 {deviation:.0%}（槽位 {latest_slot}）"
    )
    if initial_soc != measured_initial_soc:
        reason += f"; SOC规划值从{measured_initial_soc:.4f}限制到{initial_soc:.4f}"
    revision_no = (revisions[-1]["revision"] if revisions else 0) + 1
    db.insert_rolling_plan(
        site_id=site_id,
        date=today.isoformat(),
        revision=revision_no,
        base_run_id=base.run_id or "",
        initial_soc=round(initial_soc, 4),
        schedule_json=json.dumps(
            [p.model_dump(mode="json") for p in result.schedule]
        ),
        expected_benefit_yuan=result.expected_benefit_yuan,
        trigger_reason=reason,
    )
    return get_rolling_state(db, site_id)
