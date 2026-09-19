from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_db
from app.domain.models import MeasurementRead, MeasurementUpload
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites/{site_id}/measurements", tags=["measurements"])


@router.post("")
def upload_measurements(
    site_id: str,
    payload: MeasurementUpload,
    db: Database = Depends(get_db),
) -> dict[str, int | str]:
    try:
        accepted = db.insert_measurements(site_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"site_id": site_id, "accepted": accepted}


@router.get("", response_model=list[MeasurementRead])
def list_measurements(
    site_id: str,
    data_type: str = Query(...),
    target_date: date = Query(..., alias="date"),
    db: Database = Depends(get_db),
) -> list[MeasurementRead]:
    site = db.get_site(site_id)
    if site is None:
        raise HTTPException(status_code=404, detail=f"site not found: {site_id}")
    return db.get_measurements_for_date(site_id, data_type, target_date, site.location.timezone)
