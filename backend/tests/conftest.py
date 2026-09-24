from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

CN_TZ = timezone(timedelta(hours=8))

SITE_PAYLOAD = {
    "site_id": "S-API",
    "name": "API 测试站点",
    "location": {"lat": 32.2, "lon": 118.7, "timezone": "Asia/Shanghai"},
    "pv": {"capacity_kwp": 1000},
    "load": {"peak_kw": 800},
    "storage": {
        "pcs_kw": 500,
        "capacity_kwh": 1000,
        "round_trip_eff": 0.88,
        "soc_min": 0.10,
        "soc_max": 0.95,
    },
}


@pytest.fixture()
def client(tmp_path) -> TestClient:
    return TestClient(create_app(str(tmp_path / "test.db")))


@pytest.fixture()
def site_client(client: TestClient) -> TestClient:
    response = client.post("/api/v1/sites", json=SITE_PAYLOAD)
    assert response.status_code == 201, response.text
    return client


def day_points(start: datetime, value: float, count: int = 96) -> list[dict]:
    return [
        {"ts": (start + timedelta(minutes=15 * index)).isoformat(), "value": value}
        for index in range(count)
    ]
