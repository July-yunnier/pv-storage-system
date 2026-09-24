from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.application import report_service
from tests.conftest import CN_TZ, day_points

START = datetime(2026, 9, 17, tzinfo=CN_TZ)


@pytest.fixture()
def sizing_run_id(site_client: TestClient) -> str:
    for offset in range(3):
        day = START + timedelta(days=offset)
        for data_type, value in [("pv_power", 500), ("load_power", 600)]:
            assert site_client.post(
                "/api/v1/sites/S-API/measurements",
                json={
                    "data_type": data_type,
                    "unit": "kW",
                    "resolution_minutes": 15,
                    "points": day_points(day, value),
                },
            ).status_code == 200
    response = site_client.post(
        "/api/v1/storage/sizing",
        json={
            "site_id": "S-API",
            "candidates": {"pcs_kw": [200, 500], "duration_hours": [1, 2]},
            "representative_days": 3,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


def create_report(client: TestClient, run_id: str) -> dict:
    response = client.post(
        "/api/v1/reports", json={"site_id": "S-API", "sizing_run_id": run_id}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_report_lifecycle_and_download(site_client: TestClient, sizing_run_id: str) -> None:
    body = create_report(site_client, sizing_run_id)
    assert body["status"] == "succeeded"
    assert body["report_id"].startswith("RP")
    assert body["error"] is None
    assert body["finished_at"]

    fetched = site_client.get(f"/api/v1/reports/{body['report_id']}").json()
    assert fetched["status"] == "succeeded"
    assert fetched["download_url"] == f"/api/v1/reports/{body['report_id']}/download"

    download = site_client.get(fetched["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/pdf")
    assert download.content[:5] == b"%PDF-"
    assert len(download.content) > 2000


def test_report_unknown_sizing_run_404(site_client: TestClient) -> None:
    response = site_client.post(
        "/api/v1/reports", json={"site_id": "S-API", "sizing_run_id": "ZDEADBEEF"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_report_unknown_report_id_404(site_client: TestClient) -> None:
    assert site_client.get("/api/v1/reports/RPNOPE00").status_code == 404
    assert site_client.get("/api/v1/reports/RPNOPE00/download").status_code == 404


def test_report_failure_stores_error(
    site_client: TestClient, sizing_run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("pdf engine exploded")

    monkeypatch.setattr(report_service, "build_sizing_pdf", boom)
    body = create_report(site_client, sizing_run_id)
    assert body["status"] == "failed"
    assert "pdf engine exploded" in body["error"]

    download = site_client.get(f"/api/v1/reports/{body['report_id']}/download")
    assert download.status_code == 422
    assert "failed" in download.json()["error"]["message"]


def test_report_download_pending_409(site_client: TestClient, sizing_run_id: str) -> None:
    # 直接落一条 pending 任务，验证 409 分支
    db = site_client.app.state.db
    db.insert_report_job(
        report_id="RPPENDTEST", site_id="S-API", report_type="sizing_v1", sizing_run_id=sizing_run_id
    )
    response = site_client.get("/api/v1/reports/RPPENDTEST/download")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONFLICT"
