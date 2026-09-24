from datetime import datetime

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ, day_points


def test_health(site_client: TestClient) -> None:
    assert site_client.get("/api/v1/health").json()["status"] == "ok"


def test_schedule_flow(site_client: TestClient) -> None:
    start = datetime(2026, 9, 20, tzinfo=CN_TZ)
    for data_type, value in [("pv_power", 0), ("load_power", 800)]:
        response = site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(start, value),
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["accepted"] == 96

    response = site_client.post(
        "/api/v1/storage/schedule",
        json={"site_id": "S-API", "date": "2026-09-20", "initial_soc": 0.5},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["schedule"]) == 96
    assert body["expected_benefit_yuan"] > 0
