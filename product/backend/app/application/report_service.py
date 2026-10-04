"""报告任务编排：创建 report_jobs 记录并同步生成 PDF，状态全程落库。"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.api import errors
from app.application.report_pdf import build_sizing_pdf
from app.domain.models import ReportJob
from app.infrastructure.db import Database

REPORT_TYPE_SIZING = "sizing_v1"


def reports_dir(db: Database) -> Path:
    if db.path == ":memory:":
        return Path("data/reports")
    return Path(db.path).parent / "reports"


def create_report(db: Database, site_id: str, sizing_run_id: str) -> ReportJob:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    run = db.get_sizing_run(sizing_run_id)
    if run is None or run.site_id != site_id:
        raise errors.not_found(f"sizing run not found: {sizing_run_id}")

    report_id = f"RP{uuid4().hex[:8].upper()}"
    db.insert_report_job(
        report_id=report_id,
        site_id=site_id,
        report_type=REPORT_TYPE_SIZING,
        sizing_run_id=sizing_run_id,
    )
    db.update_report_job(report_id, status="running")
    try:
        out_dir = reports_dir(db)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{report_id}.pdf"
        tariff = db.get_tariff(site.tariff_id or "T-DEFAULT")
        build_sizing_pdf(site, tariff, run, path)
    except Exception as exc:  # 失败也要落可读 error，不抛 500
        db.update_report_job(report_id, status="failed", error=str(exc), finished=True)
    else:
        db.update_report_job(report_id, status="succeeded", file_path=str(path), finished=True)
    job = db.get_report_job(report_id)
    assert job is not None
    return job


def get_report(db: Database, report_id: str) -> ReportJob:
    job = db.get_report_job(report_id)
    if job is None:
        raise errors.not_found(f"report not found: {report_id}")
    return job


def report_file_path(db: Database, report_id: str) -> Path:
    job = get_report(db, report_id)
    if job.status in ("pending", "running"):
        raise errors.conflict(f"report is not finished yet: {job.status}")
    if job.status == "failed":
        raise errors.validation_error(f"report generation failed: {job.error}")
    if not job.file_path or not Path(job.file_path).exists():
        raise errors.not_found("report file is missing")
    return Path(job.file_path)
