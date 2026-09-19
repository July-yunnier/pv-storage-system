from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_db
from app.domain.models import SiteCreate, SitePatch, SiteRead
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites", tags=["sites"])


@router.post("", response_model=SiteRead, status_code=status.HTTP_201_CREATED)
def create_site(payload: SiteCreate, db: Database = Depends(get_db)) -> SiteRead:
    if payload.tariff_id and db.get_tariff(payload.tariff_id) is None:
        raise HTTPException(status_code=422, detail=f"tariff not found: {payload.tariff_id}")
    try:
        return db.create_site(payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("", response_model=list[SiteRead])
def list_sites(db: Database = Depends(get_db)) -> list[SiteRead]:
    return db.list_sites()


@router.get("/{site_id}", response_model=SiteRead)
def get_site(site_id: str, db: Database = Depends(get_db)) -> SiteRead:
    site = db.get_site(site_id)
    if site is None:
        raise HTTPException(status_code=404, detail=f"site not found: {site_id}")
    return site


@router.patch("/{site_id}", response_model=SiteRead)
def update_site(site_id: str, payload: SitePatch, db: Database = Depends(get_db)) -> SiteRead:
    if payload.tariff_id and db.get_tariff(payload.tariff_id) is None:
        raise HTTPException(status_code=422, detail=f"tariff not found: {payload.tariff_id}")
    site = db.update_site(site_id, payload)
    if site is None:
        raise HTTPException(status_code=404, detail=f"site not found: {site_id}")
    return site
