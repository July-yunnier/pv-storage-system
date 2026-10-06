"""站点总览聚合：站点配置 + 当前 SOC + 数据健康 + 最新调度运行（含动作区间归并）。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel

from app.api import errors
from app.application.quality import build_data_quality_report
from app.domain.models import DispatchPoint
from app.infrastructure.db import Database

WINDOW_STEP_MINUTES = 15


class OverviewSite(BaseModel):
    name: str
    pv_capacity_kwp: float
    storage_pcs_kw: float
    storage_capacity_kwh: float
    soc_min: float
    soc_max: float


class CurrentSoc(BaseModel):
    ts: datetime
    value: float


class DataStatus(BaseModel):
    date: date
    pv_completeness: float
    load_completeness: float
    pv_status: Literal["complete", "partial", "empty"]
    load_status: Literal["complete", "partial", "empty"]


class ActionWindow(BaseModel):
    action: Literal["charge", "discharge"]
    start: str
    end: str
    max_power_kw: float


class LatestRun(BaseModel):
    run_id: str
    date: date
    strategy: str
    data_source: str
    expected_benefit_yuan: float
    optimized_cost_yuan: float
    equivalent_full_cycles: float
    created_at: datetime
    windows: list[ActionWindow]


class SiteOverview(BaseModel):
    site_id: str
    generated_at: datetime
    site: OverviewSite
    current_soc: CurrentSoc | None
    latest_measurement_date: date | None
    data_status: DataStatus | None
    latest_run: LatestRun | None


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def merge_dispatch_windows(schedule: list[DispatchPoint]) -> list[ActionWindow]:
    """归并连续同动作（charge/discharge）区间为窗口；idle 隔断且不出窗口。

    start = 首点 HH:mm，end = 末点 +15min 的 HH:mm（跨午夜记 24:00）。
    """
    windows: list[ActionWindow] = []
    current_action: str | None = None
    start_min = end_min = 0
    max_power = 0.0

    def flush() -> None:
        if current_action is not None:
            windows.append(
                ActionWindow(
                    action=current_action,  # type: ignore[arg-type]
                    start=_hhmm(start_min),
                    end=_hhmm(end_min),
                    max_power_kw=round(max_power, 6),
                )
            )

    for point in schedule:
        if point.action == "idle":
            flush()
            current_action = None
            continue
        minute = point.ts.hour * 60 + point.ts.minute
        if point.action == current_action:
            end_min = minute + WINDOW_STEP_MINUTES
            max_power = max(max_power, point.power_kw)
        else:
            flush()
            current_action = point.action
            start_min = minute
            end_min = minute + WINDOW_STEP_MINUTES
            max_power = point.power_kw
    flush()
    return windows


def get_overview(db: Database, site_id: str) -> SiteOverview:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    tz_name = site.location.timezone

    soc_point = db.get_latest_measurement_point(site_id, "storage_soc", tz_name)
    current_soc = (
        CurrentSoc(ts=soc_point[0], value=soc_point[1]) if soc_point is not None else None
    )

    latest_ts = db.get_latest_measurement_ts(site_id, ["pv_power", "load_power"], tz_name)
    latest_date = latest_ts.date() if latest_ts is not None else None
    data_status = None
    if latest_date is not None:
        pv_report = build_data_quality_report(
            site_id,
            "pv_power",
            latest_date,
            tz_name,
            db.get_measurement_rows_for_date(site_id, "pv_power", latest_date, tz_name),
        )
        load_report = build_data_quality_report(
            site_id,
            "load_power",
            latest_date,
            tz_name,
            db.get_measurement_rows_for_date(site_id, "load_power", latest_date, tz_name),
        )
        data_status = DataStatus(
            date=latest_date,
            pv_completeness=pv_report.completeness,
            load_completeness=load_report.completeness,
            pv_status=pv_report.status,
            load_status=load_report.status,
        )

    latest_run = None
    runs = db.list_dispatch_runs(site_id)
    if runs:
        detail = db.get_dispatch_run(runs[0].run_id)
        if detail is not None:
            latest_run = LatestRun(
                run_id=detail.run_id or runs[0].run_id,
                date=detail.date,
                strategy=detail.strategy,
                data_source=detail.data_source,
                expected_benefit_yuan=detail.expected_benefit_yuan,
                optimized_cost_yuan=detail.optimized_cost_yuan,
                equivalent_full_cycles=detail.equivalent_full_cycles,
                created_at=detail.created_at,
                windows=merge_dispatch_windows(detail.schedule),
            )

    return SiteOverview(
        site_id=site_id,
        generated_at=datetime.now(timezone.utc),
        site=OverviewSite(
            name=site.name,
            pv_capacity_kwp=site.pv.capacity_kwp,
            storage_pcs_kw=site.storage.pcs_kw,
            storage_capacity_kwh=site.storage.capacity_kwh,
            soc_min=site.storage.soc_min,
            soc_max=site.storage.soc_max,
        ),
        current_soc=current_soc,
        latest_measurement_date=latest_date,
        data_status=data_status,
        latest_run=latest_run,
    )
