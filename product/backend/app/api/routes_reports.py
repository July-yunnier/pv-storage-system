from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.dependencies import get_db
from app.application import report_service
from app.domain.models import ReportCreate, ReportJob, ReportJobStatusResponse
from app.infrastructure.db import Database

router = APIRouter(prefix="/reports", tags=["reports"])


def _status_response(job: ReportJob) -> ReportJobStatusResponse:
    return ReportJobStatusResponse(
        report_id=job.report_id,
        site_id=job.site_id,
        report_type=job.report_type,
        status=job.status,
        error=job.error,
        created_at=job.created_at,
        finished_at=job.finished_at,
        download_url=f"/api/v1/reports/{job.report_id}/download",
    )


@router.post("", response_model=ReportJobStatusResponse)
def create_report(payload: ReportCreate, db: Database = Depends(get_db)) -> ReportJobStatusResponse:
    job = report_service.create_report(db, payload.site_id, payload.sizing_run_id)
    return _status_response(job)


@router.get("/{report_id}", response_model=ReportJobStatusResponse)
def get_report(report_id: str, db: Database = Depends(get_db)) -> ReportJobStatusResponse:
    return _status_response(report_service.get_report(db, report_id))


@router.get("/{report_id}/download")
def download_report(report_id: str, db: Database = Depends(get_db)) -> FileResponse:
    path = report_service.report_file_path(db, report_id)
    return FileResponse(path, media_type="application/pdf", filename=f"{report_id}.pdf")
