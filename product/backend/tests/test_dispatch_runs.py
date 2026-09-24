import json
import sqlite3
from datetime import datetime

from fastapi.testclient import TestClient

from app.domain.models import SiteCreate
from app.infrastructure.db import Database
from tests.conftest import CN_TZ, SITE_PAYLOAD, day_points

SCHEDULE_URL = "/api/v1/storage/schedule"
RUNS_URL = "/api/v1/sites/S-API/dispatch-runs"
START = datetime(2026, 9, 20, tzinfo=CN_TZ)


def seed_day(client: TestClient, load_value: float = 800) -> None:
    for data_type, value in [("pv_power", 0), ("load_power", load_value)]:
        response = client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(START, value),
            },
        )
        assert response.status_code == 200, response.text


def run_schedule(client: TestClient) -> dict:
    response = client.post(
        SCHEDULE_URL, json={"site_id": "S-API", "date": "2026-09-20", "initial_soc": 0.5}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_schedule_response_carries_run_id_and_persists(site_client: TestClient) -> None:
    seed_day(site_client)
    body = run_schedule(site_client)
    run_id = body["run_id"]
    assert run_id and run_id.startswith("R")

    listing = site_client.get(RUNS_URL)
    assert listing.status_code == 200
    runs = listing.json()
    assert [run["run_id"] for run in runs] == [run_id]
    summary = runs[0]
    assert summary["site_id"] == "S-API"
    assert summary["date"] == "2026-09-20"
    assert summary["strategy"] == "rule_v1"
    assert summary["initial_soc"] == 0.5
    assert summary["expected_benefit_yuan"] == body["expected_benefit_yuan"]
    assert summary["created_at"]

    detail = site_client.get(f"{RUNS_URL}/{run_id}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert len(detail_body["schedule"]) == 96
    assert detail_body["run_id"] == run_id
    assert detail_body["input_digest"]
    assert detail_body["expected_benefit_yuan"] == body["expected_benefit_yuan"]


def test_same_input_produces_same_digest(site_client: TestClient) -> None:
    seed_day(site_client)
    first = run_schedule(site_client)
    second = run_schedule(site_client)
    assert first["run_id"] != second["run_id"]
    first_detail = site_client.get(f"{RUNS_URL}/{first['run_id']}").json()
    second_detail = site_client.get(f"{RUNS_URL}/{second['run_id']}").json()
    assert first_detail["input_digest"] == second_detail["input_digest"]
    # 结果数值完全一致
    assert first["expected_benefit_yuan"] == second["expected_benefit_yuan"]


def test_dispatch_run_404_for_unknown_run(site_client: TestClient) -> None:
    response = site_client.get(f"{RUNS_URL}/RDEADBEEF")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_dispatch_run_404_for_other_site(site_client: TestClient) -> None:
    seed_day(site_client)
    body = run_schedule(site_client)
    other = dict(SITE_PAYLOAD, site_id="S-OTHER", name="另一个站点")
    assert site_client.post("/api/v1/sites", json=other).status_code == 201
    response = site_client.get(f"/api/v1/sites/S-OTHER/dispatch-runs/{body['run_id']}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
    # 另一个站点的运行列表为空
    assert site_client.get("/api/v1/sites/S-OTHER/dispatch-runs").json() == []


def test_dispatch_runs_unknown_site_404(site_client: TestClient) -> None:
    response = site_client.get("/api/v1/sites/S-NOPE/dispatch-runs")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_schedule_with_missing_point_returns_422(site_client: TestClient) -> None:
    pv_points = day_points(START, 0)
    pv_points[10] = {"ts": pv_points[10]["ts"], "value": None, "quality": "missing"}
    response = site_client.post(
        "/api/v1/sites/S-API/measurements",
        json={"data_type": "pv_power", "unit": "kW", "resolution_minutes": 15, "points": pv_points},
    )
    assert response.status_code == 200
    seed_response = site_client.post(
        "/api/v1/sites/S-API/measurements",
        json={
            "data_type": "load_power",
            "unit": "kW",
            "resolution_minutes": 15,
            "points": day_points(START, 800),
        },
    )
    assert seed_response.status_code == 200
    response = site_client.post(
        SCHEDULE_URL,
        json={"site_id": "S-API", "date": "2026-09-20", "initial_soc": 0.5, "source": "measurements"},
    )
    assert response.status_code == 422
    assert "缺失点" in response.json()["error"]["message"]


def test_schedule_with_incomplete_day_returns_422(site_client: TestClient) -> None:
    for data_type, value in [("pv_power", 0), ("load_power", 800)]:
        response = site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(START, value, count=90),
            },
        )
        assert response.status_code == 200
    response = site_client.post(
        SCHEDULE_URL,
        json={"site_id": "S-API", "date": "2026-09-20", "initial_soc": 0.5, "source": "measurements"},
    )
    assert response.status_code == 422
    assert "96" in response.json()["error"]["message"]


def test_demand_limit_roundtrip_and_schedule(site_client: TestClient) -> None:
    payload = dict(SITE_PAYLOAD, site_id="S-DEMAND", demand_limit_kw=600)
    assert site_client.post("/api/v1/sites", json=payload).status_code == 201
    site = site_client.get("/api/v1/sites/S-DEMAND").json()
    assert site["demand_limit_kw"] == 600
    for data_type, value in [("pv_power", 0), ("load_power", 400)]:
        assert site_client.post(
            "/api/v1/sites/S-DEMAND/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(START, value),
            },
        ).status_code == 200
    response = site_client.post(
        SCHEDULE_URL, json={"site_id": "S-DEMAND", "date": "2026-09-20", "initial_soc": 0.5}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "demand_limit_kw=600.0" in body["constraints_applied"]
    assert all(point["grid_import_kw"] <= 600 for point in body["schedule"])


def test_legacy_site_config_without_demand_limit(tmp_path) -> None:
    db = Database(str(tmp_path / "legacy.db"))
    db.create_site(
        SiteCreate(
            site_id="S-LEG",
            name="存量站点",
            location={"lat": 32.0, "lon": 118.0},
            pv={"capacity_kwp": 100},
            storage={"pcs_kw": 50, "capacity_kwh": 100},
        )
    )
    # 模拟 I3 之前写入的 config_json（没有 demand_limit_kw 键）
    conn = sqlite3.connect(db.path)
    row = conn.execute(
        "SELECT config_json FROM sites WHERE site_id = 'S-LEG'"
    ).fetchone()
    config = json.loads(row[0])
    del config["demand_limit_kw"]
    conn.execute(
        "UPDATE sites SET config_json = ? WHERE site_id = 'S-LEG'",
        (json.dumps(config, ensure_ascii=False),),
    )
    conn.commit()
    conn.close()
    site = db.get_site("S-LEG")
    assert site is not None
    assert site.demand_limit_kw is None
