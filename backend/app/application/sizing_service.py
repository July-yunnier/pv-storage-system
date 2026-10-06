"""Sizing 编排：找代表日 → 逐候选跑调度引擎 → 经济性 → 落库 sizing_runs。"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from uuid import uuid4

from app.api import errors
from app.domain.economics.cashflow import FinancialParams
from app.domain.models import SiteRead, SizingRequest
from app.domain.sizing.planner import (
    RepresentativeDay,
    evaluate_sizing,
    generate_candidates,
)
from app.domain.sizing.result import SizingResponse
from app.infrastructure.db import Database, _zone

MAX_GRID_COMBINATIONS = 100
LOOKBACK_DAYS = 60


def _financial_params(payload: SizingRequest) -> FinancialParams:
    base = FinancialParams()
    overrides = payload.financial.model_dump(exclude_none=True)
    return base.model_copy(update=overrides) if overrides else base


def _complete_day(
    db: Database, site: SiteRead, day: date, expected_points: int = 96
) -> RepresentativeDay | None:
    pv = db.get_measurements_for_date(site.site_id, "pv_power", day, site.location.timezone)
    load = db.get_measurements_for_date(site.site_id, "load_power", day, site.location.timezone)
    if len(pv) != expected_points or len(load) != expected_points:
        return None
    if any(point.value is None for point in pv) or any(point.value is None for point in load):
        return None
    return RepresentativeDay(
        date=day,
        pv_values=[point.value for point in pv],  # type: ignore[misc]
        load_values=[point.value for point in load],  # type: ignore[misc]
    )


def find_representative_days(
    db: Database, site: SiteRead, count: int, before: date | None = None
) -> list[RepresentativeDay]:
    """最近 count 个 pv/load 数据完整的自然日（从 before 前一天往前找）。"""
    tz = _zone(site.location.timezone)
    cursor = before or datetime.now(tz).date()
    days: list[RepresentativeDay] = []
    for _ in range(LOOKBACK_DAYS):
        cursor -= timedelta(days=1)
        day = _complete_day(db, site, cursor)
        if day is not None:
            days.append(day)
            if len(days) == count:
                break
    days.reverse()
    return days


def run_sizing(db: Database, payload: SizingRequest) -> SizingResponse:
    site = db.get_site(payload.site_id)
    if site is None:
        raise errors.not_found(f"site not found: {payload.site_id}")
    grid_size = len(payload.candidates.pcs_kw) * len(payload.candidates.duration_hours)
    if grid_size > MAX_GRID_COMBINATIONS:
        raise errors.validation_error(
            f"candidate grid has {grid_size} combinations, max {MAX_GRID_COMBINATIONS}"
        )
    tariff_id = site.tariff_id or "T-DEFAULT"
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {tariff_id}")

    days = find_representative_days(db, site, payload.representative_days)
    if len(days) < payload.representative_days:
        raise errors.data_quality_error(
            f"representative days insufficient: need {payload.representative_days} complete days "
            f"(pv/load 96 points each), found {len(days)} in the last {LOOKBACK_DAYS} days"
        )

    params = _financial_params(payload)
    try:
        candidates = generate_candidates(payload.candidates.pcs_kw, payload.candidates.duration_hours)
        result = evaluate_sizing(site, days, tariff, candidates, params)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc

    run_id = f"Z{uuid4().hex[:8].upper()}"
    digest = hashlib.sha256(
        json.dumps(
            {
                "site_id": payload.site_id,
                "pcs_kw": payload.candidates.pcs_kw,
                "duration_hours": payload.candidates.duration_hours,
                "representative_days": payload.representative_days,
                "representative_dates": [day.date.isoformat() for day in days],
                "financial": params.model_dump(),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]

    response = SizingResponse(**result.model_dump(), run_id=run_id)
    created_at = db.insert_sizing_run(
        run_id=run_id, site_id=site.site_id, input_digest=digest, result=response
    )
    return response.model_copy(update={"created_at": created_at})
