from fastapi import APIRouter, Depends

from app.api import errors
from app.api.dependencies import get_db
from app.application import sizing_service
from app.domain.models import SizingRequest
from app.domain.sizing.result import SizingResponse, SizingRunDetail, SizingRunSummary
from app.infrastructure.db import Database

router = APIRouter(prefix="/storage", tags=["sizing"])

runs_router = APIRouter(prefix="/sites/{site_id}/sizing-runs", tags=["sizing"])


@router.post("/sizing", response_model=SizingResponse)
def create_sizing(payload: SizingRequest, db: Database = Depends(get_db)) -> SizingResponse:
    return sizing_service.run_sizing(db, payload)


@runs_router.get("", response_model=list[SizingRunSummary])
def list_sizing_runs(site_id: str, db: Database = Depends(get_db)) -> list[SizingRunSummary]:
    if db.get_site(site_id) is None:
        raise errors.not_found(f"site not found: {site_id}")
    return db.list_sizing_runs(site_id)


@runs_router.get("/{run_id}", response_model=SizingRunDetail)
def get_sizing_run(
    site_id: str,
    run_id: str,
    db: Database = Depends(get_db),
) -> SizingRunDetail:
    run = db.get_sizing_run(run_id)
    if run is None or run.site_id != site_id:
        raise errors.not_found(f"sizing run not found: {run_id}")
    return run
