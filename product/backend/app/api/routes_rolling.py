"""日内滚动修正接口。"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_db
from app.application import rolling_service
from app.domain.models import RollingState
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites", tags=["rolling"])


@router.get("/{site_id}/rolling", response_model=RollingState)
def get_rolling(site_id: str, db: Database = Depends(get_db)) -> RollingState:
    return rolling_service.get_rolling_state(db, site_id)


@router.post("/{site_id}/rolling/run", response_model=RollingState)
def run_rolling(site_id: str, db: Database = Depends(get_db)) -> RollingState:
    return rolling_service.maybe_rolling_adjust(db, site_id, force=True)
