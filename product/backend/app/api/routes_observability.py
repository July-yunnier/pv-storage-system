"""观测体系接口：预测精度对账查询。"""

from fastapi import APIRouter, Depends, Query

from app.api import errors
from app.api.dependencies import get_db
from app.application import observability_service
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites", tags=["observability"])


@router.get("/{site_id}/accuracy")
def get_accuracy(
    site_id: str,
    target: str | None = Query(default=None, pattern="^(pv_power|load_power)$"),
    days: int = Query(default=30, ge=1, le=365),
    db: Database = Depends(get_db),
) -> dict:
    if db.get_site(site_id) is None:
        raise errors.not_found(f"site not found: {site_id}")
    observability_service.reconcile_site(db, site_id)  # 先补对账再读，读到的总是最新的
    return observability_service.accuracy_summary(db, site_id, target, days)


@router.post("/{site_id}/accuracy/reconcile")
def reconcile_accuracy(site_id: str, db: Database = Depends(get_db)) -> dict:
    if db.get_site(site_id) is None:
        raise errors.not_found(f"site not found: {site_id}")
    done = observability_service.reconcile_site(db, site_id)
    return {"reconciled": done}
