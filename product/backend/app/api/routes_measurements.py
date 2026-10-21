from datetime import date
from typing import get_args

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.api import errors
from app.api.dependencies import get_db
from app.application.ingestion import ingest_csv
from app.application.quality import DataQualityReport, build_data_quality_report
from app.domain.models import MeasurementRead, MeasurementType, MeasurementUpload
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites/{site_id}/measurements", tags=["measurements"])

quality_router = APIRouter(prefix="/sites/{site_id}/data-quality", tags=["measurements"])


@router.post("")
def upload_measurements(
    site_id: str,
    payload: MeasurementUpload,
    db: Database = Depends(get_db),
) -> dict[str, int | str]:
    try:
        accepted = db.insert_measurements(site_id, payload)
    except KeyError as exc:
        raise errors.not_found(str(exc)) from exc
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    return {"site_id": site_id, "accepted": accepted}


@router.post("/csv")
async def upload_measurements_csv(
    site_id: str,
    file: UploadFile = File(...),
    data_type: str = Form(...),
    resolution_minutes: int = Form(15, gt=0),
    db: Database = Depends(get_db),
) -> dict:
    if data_type not in get_args(MeasurementType):
        raise errors.validation_error(
            f"invalid data_type: {data_type!r}, expected one of {list(get_args(MeasurementType))}"
        )
    raw = await file.read()
    if not raw:
        raise errors.validation_error("csv file is empty")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise errors.validation_error("csv file must be UTF-8 encoded") from exc
    try:
        result = ingest_csv(db, site_id, data_type, content, resolution_minutes=resolution_minutes)  # type: ignore[arg-type]
    except KeyError as exc:
        raise errors.not_found(str(exc)) from exc
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    return {
        "site_id": result.site_id,
        "data_type": result.data_type,
        "total_rows": result.total_rows,
        "accepted": result.accepted,
        "rejected": [{"row": row.row, "reason": row.reason} for row in result.rejected],
    }


@router.get("", response_model=list[MeasurementRead])
def list_measurements(
    site_id: str,
    data_type: str = Query(...),
    target_date: date = Query(..., alias="date"),
    db: Database = Depends(get_db),
) -> list[MeasurementRead]:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    return db.get_measurements_for_date(site_id, data_type, target_date, site.location.timezone)


@quality_router.get("", response_model=DataQualityReport)
def get_data_quality(
    site_id: str,
    data_type: str = Query(...),
    target_date: date = Query(..., alias="date"),
    db: Database = Depends(get_db),
) -> DataQualityReport:
    if data_type not in get_args(MeasurementType):
        raise errors.validation_error(
            f"invalid data_type: {data_type!r}, expected one of {list(get_args(MeasurementType))}"
        )
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    rows = db.get_measurement_rows_for_date(site_id, data_type, target_date, site.location.timezone)
    return build_data_quality_report(site_id, data_type, target_date, site.location.timezone, rows)
