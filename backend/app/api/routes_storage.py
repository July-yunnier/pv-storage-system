from datetime import datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_db
from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.models import ScheduleRequest, ScheduleResponse
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database

router = APIRouter(prefix="/storage", tags=["storage"])


@router.post("/schedule", response_model=ScheduleResponse)
def create_schedule(
    payload: ScheduleRequest,
    db: Database = Depends(get_db),
) -> ScheduleResponse:
    site = db.get_site(payload.site_id)
    if site is None:
        raise HTTPException(status_code=404, detail=f"site not found: {payload.site_id}")
    if payload.resolution_minutes != 15:
        raise HTTPException(status_code=422, detail="MVP only supports 15-minute resolution")

    pv_points = db.get_measurements_for_date(
        payload.site_id, "pv_power", payload.date, site.location.timezone
    )
    load_points = db.get_measurements_for_date(
        payload.site_id, "load_power", payload.date, site.location.timezone
    )
    if not pv_points or not load_points:
        raise HTTPException(
            status_code=422,
            detail="pv_power and load_power measurements are required for the requested date",
        )
    if len(pv_points) != len(load_points):
        raise HTTPException(status_code=422, detail="pv/load measurement lengths do not match")
    expected_points = 24 * 60 // payload.resolution_minutes
    if len(pv_points) != expected_points:
        raise HTTPException(
            status_code=422,
            detail=f"a full day requires exactly {expected_points} points at {payload.resolution_minutes}-minute resolution",
        )

    tariff_id = site.tariff_id or "T-DEFAULT"
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise HTTPException(status_code=422, detail=f"tariff not found: {tariff_id}")
    try:
        provider = TouProvider(tariff)
        price_points = provider.points(
            pv_points[0].ts,
            periods=len(pv_points),
            resolution_minutes=payload.resolution_minutes,
        )
        return RuleDispatchEngine().run(
            DispatchInputs(
                site=site,
                target_date=payload.date,
                initial_soc=payload.initial_soc,
                pv_points=pv_points,
                load_points=load_points,
                price_points=price_points,
                resolution_minutes=payload.resolution_minutes,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
