"""历史滚动回放实验接口。"""

from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api import errors
from app.api.dependencies import get_db
from app.application import replay_service
from app.infrastructure.db import Database

router = APIRouter(prefix="/sites", tags=["replay"])
runs_router = APIRouter(prefix="/replay-runs", tags=["replay"])

EXPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "replay"


class ReplayRequest(BaseModel):
    test_days: int = Field(default=30, ge=7, le=120)
    pv_methods: list[str] | None = None
    strategies: list[str] | None = None


@router.post("/{site_id}/replay")
def run_replay(site_id: str, payload: ReplayRequest, db: Database = Depends(get_db)) -> dict:
    try:
        return replay_service.run_replay(
            db,
            site_id,
            test_days=payload.test_days,
            pv_methods=payload.pv_methods,
            strategies=payload.strategies,
            export_dir=EXPORT_DIR,
        )
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc


@router.get("/{site_id}/replay-runs")
def list_replay_runs(site_id: str, db: Database = Depends(get_db)) -> list[dict]:
    if db.get_site(site_id) is None:
        raise errors.not_found(f"site not found: {site_id}")
    return db.list_replay_runs(site_id)


@runs_router.get("/{run_id}")
def get_replay_run(run_id: str, db: Database = Depends(get_db)) -> dict:
    run = db.get_replay_run(run_id)
    if run is None:
        raise errors.not_found(f"replay run not found: {run_id}")
    return {**run, "day_results": db.get_replay_day_results(run_id)}
