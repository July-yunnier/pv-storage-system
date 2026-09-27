"""实时看板接口：当前快照 + 模拟数据源。"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_db
from app.application import live_service
from app.domain.models import LiveSimulateResponse, LiveStatus
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites", tags=["live"])


@router.get("/{site_id}/live", response_model=LiveStatus)
def get_live(site_id: str, db: Database = Depends(get_db)) -> LiveStatus:
    return live_service.get_live(db, site_id)


@router.post("/{site_id}/simulate-live", response_model=LiveSimulateResponse)
def simulate_live(site_id: str, db: Database = Depends(get_db)) -> LiveSimulateResponse:
    return live_service.simulate_live(db, site_id)
