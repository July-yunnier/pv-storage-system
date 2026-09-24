from fastapi import APIRouter, Depends

from app.api import errors
from app.api.dependencies import get_db
from app.application import dispatch_service
from app.domain.models import DispatchRunDetail, DispatchRunSummary, ScheduleRequest, ScheduleResponse
from app.infrastructure.db import Database

router = APIRouter(prefix="/storage", tags=["storage"])

runs_router = APIRouter(prefix="/sites/{site_id}/dispatch-runs", tags=["storage"])


@router.post("/schedule", response_model=ScheduleResponse)
def create_schedule(
    payload: ScheduleRequest,
    db: Database = Depends(get_db),
) -> ScheduleResponse:
    return dispatch_service.run_schedule(db, payload)


@runs_router.get("", response_model=list[DispatchRunSummary])
def list_dispatch_runs(site_id: str, db: Database = Depends(get_db)) -> list[DispatchRunSummary]:
    if db.get_site(site_id) is None:
        raise errors.not_found(f"site not found: {site_id}")
    return db.list_dispatch_runs(site_id)


@runs_router.get("/{run_id}", response_model=DispatchRunDetail)
def get_dispatch_run(
    site_id: str,
    run_id: str,
    db: Database = Depends(get_db),
) -> DispatchRunDetail:
    run = db.get_dispatch_run(run_id)
    if run is None or run.site_id != site_id:
        raise errors.not_found(f"dispatch run not found: {run_id}")
    return run
