from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from app.application.overview_service import merge_dispatch_windows
from app.domain.models import DispatchPoint
from tests.conftest import CN_TZ, day_points

OVERVIEW_URL = "/api/v1/sites/S-API/overview"
DAY1 = datetime(2026, 9, 18, tzinfo=CN_TZ)
DAY2 = datetime(2026, 9, 20, tzinfo=CN_TZ)


def upload(client: TestClient, data_type: str, points: list[dict]) -> None:
    response = client.post(
        "/api/v1/sites/S-API/measurements",
        json={"data_type": data_type, "unit": "kW", "resolution_minutes": 15, "points": points},
    )
    assert response.status_code == 200, response.text


def make_point(ts: datetime, action: str, power: float = 100.0) -> DispatchPoint:
    return DispatchPoint(
        ts=ts,
        action=action,
        power_kw=power if action != "idle" else 0.0,
        soc_start=0.5,
        soc_end=0.5,
        pv_power_kw=0,
        load_power_kw=800,
        price=0.5,
        tier="flat",
        grid_import_kw=0,
        grid_export_kw=0,
        pv_curtailment_kw=0,
    )


def test_merge_windows_merge_single_and_idle_breaks() -> None:
    base = datetime(2026, 9, 20, tzinfo=CN_TZ)
    schedule = [
        # 00:00-00:45 连续充电（功率峰值 200）
        make_point(base, "charge", 150),
        make_point(base + timedelta(minutes=15), "charge", 200),
        make_point(base + timedelta(minutes=30), "charge", 100),
        # 00:45 idle 隔断
        make_point(base + timedelta(minutes=45), "idle"),
        # 01:00 单点放电
        make_point(base + timedelta(minutes=60), "discharge", 300),
        # 01:15 直接换动作（无 idle）也应开新窗口
        make_point(base + timedelta(minutes=75), "charge", 50),
    ]
    windows = merge_dispatch_windows(schedule)
    assert [w.model_dump() for w in windows] == [
        {"action": "charge", "start": "00:00", "end": "00:45", "max_power_kw": 200},
        {"action": "discharge", "start": "01:00", "end": "01:15", "max_power_kw": 300},
        {"action": "charge", "start": "01:15", "end": "01:30", "max_power_kw": 50},
    ]


def test_merge_windows_all_idle_empty() -> None:
    base = datetime(2026, 9, 20, tzinfo=CN_TZ)
    assert merge_dispatch_windows([make_point(base, "idle")]) == []


def test_overview_empty_site(site_client: TestClient) -> None:
    body = site_client.get(OVERVIEW_URL).json()
    assert body["site_id"] == "S-API"
    assert body["generated_at"]
    assert body["site"]["pv_capacity_kwp"] == 1000
    assert body["site"]["storage_pcs_kw"] == 500
    assert body["site"]["soc_min"] == 0.1
    assert body["current_soc"] is None
    assert body["latest_measurement_date"] is None
    assert body["data_status"] is None
    assert body["latest_run"] is None


def test_overview_data_status_anchors_latest_date(site_client: TestClient) -> None:
    # 较早一天完整、较新一天 pv 只有半天 → 锚定较新一天
    upload(site_client, "pv_power", day_points(DAY1, 500))
    upload(site_client, "load_power", day_points(DAY1, 600))
    upload(site_client, "pv_power", day_points(DAY2, 500, count=48))
    body = site_client.get(OVERVIEW_URL).json()
    assert body["latest_measurement_date"] == "2026-09-20"
    status = body["data_status"]
    assert status["date"] == "2026-09-20"
    assert status["pv_completeness"] == 0.5
    assert status["pv_status"] == "partial"
    assert status["load_completeness"] == 0.0
    assert status["load_status"] == "empty"


def test_overview_current_soc_latest_point(site_client: TestClient) -> None:
    base = datetime(2026, 9, 20, tzinfo=CN_TZ)
    upload(
        site_client,
        "storage_soc",
        [
            {"ts": base.isoformat(), "value": 0.4},
            {"ts": (base + timedelta(minutes=15)).isoformat(), "value": 0.62},
        ],
    )
    body = site_client.get(OVERVIEW_URL).json()
    assert body["current_soc"]["value"] == 0.62
    assert body["current_soc"]["ts"].endswith("+08:00")


def test_overview_latest_run_with_windows(site_client: TestClient) -> None:
    upload(site_client, "pv_power", day_points(DAY1, 0))
    upload(site_client, "load_power", day_points(DAY1, 800))
    run = site_client.post(
        "/api/v1/storage/schedule",
        json={"site_id": "S-API", "date": "2026-09-18", "initial_soc": 0.5},
    ).json()
    body = site_client.get(OVERVIEW_URL).json()
    latest = body["latest_run"]
    assert latest["run_id"] == run["run_id"]
    assert latest["date"] == "2026-09-18"
    assert latest["strategy"] == "rule_v1"
    assert latest["data_source"] == "measurements"
    assert latest["expected_benefit_yuan"] == run["expected_benefit_yuan"]
    assert latest["optimized_cost_yuan"] == run["optimized_cost_yuan"]
    assert latest["equivalent_full_cycles"] == run["equivalent_full_cycles"]
    assert latest["created_at"]
    windows = latest["windows"]
    assert windows
    assert {w["action"] for w in windows} <= {"charge", "discharge"}
    for window in windows:
        assert window["start"] < window["end"]
        assert window["max_power_kw"] > 0
    # 归并结果与明细 schedule 手算一致
    detail = site_client.get(f"/api/v1/sites/S-API/dispatch-runs/{run['run_id']}").json()
    actions = [p["action"] for p in detail["schedule"]]
    expected_windows = sum(
        1 for i, a in enumerate(actions) if a != "idle" and (i == 0 or actions[i - 1] != a)
    )
    assert len(windows) == expected_windows


def test_overview_unknown_site_404(site_client: TestClient) -> None:
    response = site_client.get("/api/v1/sites/S-NOPE/overview")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
