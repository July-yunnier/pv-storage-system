"""实时数据源模拟器：为演示生成"正在发生"的今日测量数据。

确定性：同一站点+日期+计划 → 生成完全相同的数据（hash 伪噪声，无随机源）。
数据标记 source="simulator"，与真实接入（csv/api）明确区分。

真实项目接入时应由工厂 EMS/电表经 API 推送替换本模块。
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, time, timedelta, tzinfo

from app.domain.forecast.pv_physics import clearsky_series
from app.domain.models import DispatchPoint, SiteRead

SLOT_MINUTES = 15
SLOTS_PER_DAY = 96

# 工厂负荷日形（占峰值比例，逐小时）：夜班低、早班爬升、白班平台、晚班回落
LOAD_SHAPE = [
    0.18, 0.16, 0.15, 0.15, 0.16, 0.20,  # 00-05
    0.32, 0.55, 0.78, 0.86, 0.88, 0.87,  # 06-11
    0.85, 0.84, 0.85, 0.86, 0.84, 0.80,  # 12-17
    0.72, 0.62, 0.52, 0.42, 0.32, 0.24,  # 18-23
]


def _hash01(*material: str) -> float:
    """确定性 [0,1) 伪随机数。"""
    digest = hashlib.sha256("|".join(material).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _cloud_factor(site_id: str, day: str) -> float:
    """当日云量折扣：0.35（厚云）~ 0.95（晴），逐日确定。"""
    return 0.35 + 0.60 * _hash01("cloud", site_id, day)


def simulate_live_points(
    site: SiteRead,
    now_local: datetime,
    plan_points: list[DispatchPoint] | None = None,
) -> dict[str, list[tuple[datetime, float]]]:
    """生成今日 00:00 至当前时刻（含）的 15 分钟测量点。

    返回 {data_type: [(ts_local, value), ...]}，pv/load/grid 单位 kW，soc 为 0~1。
    """
    tz: tzinfo = now_local.tzinfo  # type: ignore[assignment]
    today = now_local.date()
    day_key = today.isoformat()
    current_slot = (now_local.hour * 60 + now_local.minute) // SLOT_MINUTES
    slots = range(0, current_slot + 1)

    clearsky = clearsky_series(site.pv, site.location.lat, site.location.lon, today, tz)
    cloud = _cloud_factor(site.site_id, day_key)

    plan_by_slot = {}
    if plan_points:
        for p in plan_points:
            plan_by_slot[(p.ts.hour * 60 + p.ts.minute) // SLOT_MINUTES] = p

    pv_points: list[tuple[datetime, float]] = []
    load_points: list[tuple[datetime, float]] = []
    soc_points: list[tuple[datetime, float]] = []
    grid_points: list[tuple[datetime, float]] = []
    day_start = datetime.combine(today, time.min, tzinfo=tz)

    for slot in slots:
        ts = day_start + timedelta(minutes=SLOT_MINUTES * slot)
        wiggle = 0.94 + 0.06 * _hash01("w", site.site_id, day_key, str(slot))
        hour = slot * SLOT_MINUTES // 60
        next_hour = min(hour + 1, 23)
        frac = (slot * SLOT_MINUTES % 60) / 60
        shape = LOAD_SHAPE[hour] * (1 - frac) + LOAD_SHAPE[next_hour] * frac

        pv_kw = round(clearsky[slot] * cloud * wiggle, 3)
        load_kw = round(site.load.peak_kw * shape * wiggle, 3)

        soc = plan_by_slot.get(slot).soc_end if slot in plan_by_slot else 0.5
        plan = plan_by_slot.get(slot)
        # 执行扰动：真实电站不会完美跟计划——SOC 缓漂 ±4%，光伏叠加云团扰动
        soc = min(max(soc + 0.04 * math.sin(slot * 0.7), 0.05), 0.98)
        pv_kw = round(pv_kw * (0.85 + 0.15 * math.sin(slot * 1.3 + 1.0)), 3)
        charge_kw = plan.power_kw if plan and plan.action == "charge" else 0.0
        discharge_kw = plan.power_kw if plan and plan.action == "discharge" else 0.0
        grid_kw = max(load_kw - pv_kw + charge_kw - discharge_kw, 0.0)

        pv_points.append((ts, pv_kw))
        load_points.append((ts, load_kw))
        soc_points.append((ts, round(soc, 4)))
        grid_points.append((ts, round(grid_kw, 3)))

    return {
        "pv_power": pv_points,
        "load_power": load_points,
        "storage_soc": soc_points,
        "grid_power": grid_points,
    }
