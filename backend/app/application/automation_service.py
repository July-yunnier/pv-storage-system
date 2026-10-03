"""计划自动化：日前调度（23:00 生成明日计划）+ 早间兜底（00:05 前补齐今日计划）。

生成失败的（如气象 API 不可用）不记录，下一分钟巡检重试。
自动生成与手动生成的区分：auto_plan_runs 审计表（list_dispatch_runs 会带 auto_generated 标记）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from app.api import errors
from app.application import dispatch_service
from app.domain.models import ScheduleRequest
from app.infrastructure.db import Database, _zone

DAY_AHEAD_HOUR = 23
BACKSTOP_MINUTE = 10  # 00:00~00:10 之间兜底


def _has_plan(db: Database, site_id: str, day: date) -> bool:
    return any(run.date == day for run in db.list_dispatch_runs(site_id))


def _latest_soc(db: Database, site_id: str, timezone_name: str) -> float:
    point = db.get_latest_measurement_point(site_id, "storage_soc", timezone_name)
    return point[1] if point is not None else 0.5


def _auto_generate(db: Database, site_id: str, day: date, kind: str) -> str | None:
    site = db.get_site(site_id)
    if site is None:
        return None
    initial_soc = _latest_soc(db, site_id, site.location.timezone)
    try:
        result = dispatch_service.run_schedule(
            db,
            ScheduleRequest(
                site_id=site_id,
                date=day,
                initial_soc=initial_soc,
                strategy="rule_v1",
                resolution_minutes=15,
                source="auto",
            ),
        )
    except errors.AppError:
        return None  # 失败不记录，下轮重试
    if result.run_id:
        db.insert_auto_plan_run(
            site_id=site_id, plan_date=day, run_id=result.run_id, kind=kind
        )
        return result.run_id
    return None


def ensure_daily_plans(db: Database, now_override: datetime | None = None) -> list[str]:
    """巡检入口：到点且缺计划则自动生成。返回本轮生成的 run_id 列表。

    只在"活"站点上生成：最新测量在 3 天内（历史回放站点的数据停在历史区间，
    不应触发面向明天的自动计划）。
    """
    generated: list[str] = []
    for site in db.list_sites():
        tz = _zone(site.location.timezone)
        now = now_override.astimezone(tz) if now_override else datetime.now(tz)
        latest = db.get_latest_measurement_ts(
            site.site_id, ["pv_power", "load_power"], site.location.timezone
        )
        if latest is None or (now - latest).days > 3:
            continue
        tomorrow = now.date() + timedelta(days=1)

        if now.hour == DAY_AHEAD_HOUR and not _has_plan(db, site.site_id, tomorrow):
            run_id = _auto_generate(db, site.site_id, tomorrow, "day_ahead")
            if run_id:
                generated.append(run_id)
        if (
            now.hour == 0
            and now.minute <= BACKSTOP_MINUTE
            and not _has_plan(db, site.site_id, now.date())
        ):
            run_id = _auto_generate(db, site.site_id, now.date(), "backstop")
            if run_id:
                generated.append(run_id)
    return generated
