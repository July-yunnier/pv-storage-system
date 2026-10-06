from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, Depends, Query, status

from app.api import errors
from app.api.dependencies import get_db
from app.application.curves import MAX_SPAN_DAYS, CurvesResponse, build_curves
from app.application.overview_service import SiteOverview, get_overview
from app.domain.models import MeasurementRead, PricePoint, SiteCreate, SitePatch, SiteRead
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database, _zone

router = APIRouter(prefix="/sites", tags=["sites"])


@router.post("", response_model=SiteRead, status_code=status.HTTP_201_CREATED)
def create_site(payload: SiteCreate, db: Database = Depends(get_db)) -> SiteRead:
    if payload.tariff_id and db.get_tariff(payload.tariff_id) is None:
        raise errors.validation_error(f"tariff not found: {payload.tariff_id}")
    try:
        return db.create_site(payload)
    except ValueError as exc:
        raise errors.conflict(str(exc)) from exc


@router.get("", response_model=list[SiteRead])
def list_sites(db: Database = Depends(get_db)) -> list[SiteRead]:
    return db.list_sites()


@router.get("/{site_id}", response_model=SiteRead)
def get_site(site_id: str, db: Database = Depends(get_db)) -> SiteRead:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    return site


@router.get("/{site_id}/overview", response_model=SiteOverview)
def get_site_overview(site_id: str, db: Database = Depends(get_db)) -> SiteOverview:
    return get_overview(db, site_id)


@router.patch("/{site_id}", response_model=SiteRead)
def update_site(site_id: str, payload: SitePatch, db: Database = Depends(get_db)) -> SiteRead:
    if payload.tariff_id and db.get_tariff(payload.tariff_id) is None:
        raise errors.validation_error(f"tariff not found: {payload.tariff_id}")
    site = db.update_site(site_id, payload)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    return site


@router.get("/{site_id}/tariff-points", response_model=list[PricePoint])
def get_tariff_points(
    site_id: str,
    target_date: date = Query(..., alias="date"),
    db: Database = Depends(get_db),
) -> list[PricePoint]:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    tariff_id = site.tariff_id or "T-DEFAULT"
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {tariff_id}")
    try:
        provider = TouProvider(tariff)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    day_start = datetime.combine(target_date, time.min, tzinfo=_zone(site.location.timezone))
    return provider.points(day_start, periods=24 * 60 // 15, resolution_minutes=15)


@router.get("/{site_id}/curves", response_model=CurvesResponse)
def get_curves(
    site_id: str,
    start: date = Query(...),
    end: date = Query(...),
    db: Database = Depends(get_db),
) -> CurvesResponse:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    if start > end:
        raise errors.validation_error(f"start must not be after end: {start} > {end}")
    if (end - start).days + 1 > MAX_SPAN_DAYS:
        raise errors.validation_error(f"date range must not exceed {MAX_SPAN_DAYS} days")
    tariff_id = site.tariff_id or "T-DEFAULT"
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {tariff_id}")
    try:
        provider = TouProvider(tariff)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc

    pv_points: list[MeasurementRead] = []
    load_points: list[MeasurementRead] = []
    cursor = start
    while cursor <= end:
        pv_points.extend(
            db.get_measurements_for_date(site_id, "pv_power", cursor, site.location.timezone)
        )
        load_points.extend(
            db.get_measurements_for_date(site_id, "load_power", cursor, site.location.timezone)
        )
        cursor += timedelta(days=1)
    try:
        return build_curves(
            site_id, start, end, site.location.timezone, pv_points, load_points, provider
        )
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
