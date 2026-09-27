"""实时看板与模拟数据源、预测 explain 的测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

CN_TZ = timezone(timedelta(hours=8))
NOW_LOCAL = datetime.now(CN_TZ)


def test_simulate_live_then_snapshot(site_client: TestClient) -> None:
    response = site_client.post("/api/v1/sites/S-API/simulate-live")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["site_id"] == "S-API"
    assert body["inserted"]["pv_power"] > 0
    assert body["up_to_slot"] == (NOW_LOCAL.hour * 60 + NOW_LOCAL.minute) // 15

    live = site_client.get("/api/v1/sites/S-API/live")
    assert live.status_code == 200, live.text
    status = live.json()
    assert status["expected_points_today"] == 96
    assert status["pv_points_today"] > 0
    assert status["latest_pv"] is not None
    assert status["latest_pv"]["value"] >= 0
    assert status["latest_soc"] is not None
    assert 0 <= status["latest_soc"]["value"] <= 1
    assert status["price_now"]["price"] > 0
    # 模拟器标记便于与真实接入区分
    curves = site_client.get(
        "/api/v1/sites/S-API/curves",
        params={"start": NOW_LOCAL.date().isoformat(), "end": NOW_LOCAL.date().isoformat()},
    )
    assert curves.status_code == 200


def test_simulate_live_is_idempotent(site_client: TestClient) -> None:
    first = site_client.post("/api/v1/sites/S-API/simulate-live").json()
    second = site_client.post("/api/v1/sites/S-API/simulate-live").json()
    assert first["inserted"]["pv_power"] == second["inserted"]["pv_power"]


def test_live_unknown_site(site_client: TestClient) -> None:
    response = site_client.get("/api/v1/sites/S-NOPE/live")
    assert response.status_code == 404


def _seed_pv_history(client: TestClient) -> None:
    """造 3 天历史光伏实测，使 auto 走融合路径。"""
    from tests.conftest import day_points

    for offset in range(3, 0, -1):
        day = (NOW_LOCAL.date() - timedelta(days=offset)).isoformat()
        start = datetime.fromisoformat(f"{day}T00:00:00+08:00")
        client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": "pv_power",
                "points": day_points(start, 300.0),
            },
        )


def test_forecast_explain_returns_weather_and_backtest(site_client: TestClient) -> None:
    _seed_pv_history(site_client)
    response = site_client.post(
        "/api/v1/forecast",
        json={
            "site_id": "S-API",
            "targets": ["pv_power"],
            "horizon_hours": 24,
            "resolution_minutes": 15,
            "options": {"explain": True},
        },
    )
    assert response.status_code == 200, response.text
    target = response.json()["results"][0]
    provenance = target["provenance"]
    assert provenance["provider"] == "mock_weather"
    assert provenance["explain"] is not None
    assert len(provenance["explain"]["weather"]) > 0
    backtest = provenance["explain"]["backtest"]
    assert backtest["sum_physics_kwh"] > 0
    assert backtest["factor"] > 0
    assert len(backtest["dates"]) >= 3


def test_forecast_explain_off_by_default(site_client: TestClient) -> None:
    response = site_client.post(
        "/api/v1/forecast",
        json={
            "site_id": "S-API",
            "targets": ["pv_power"],
            "horizon_hours": 24,
            "resolution_minutes": 15,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["provenance"]["explain"] is None
