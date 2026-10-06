from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ, day_points

SCHEDULE_URL = "/api/v1/storage/schedule"
HISTORY_START = datetime(2026, 9, 14, tzinfo=CN_TZ)


def seed_history(client: TestClient, days: int) -> None:
    for offset in range(days):
        day = HISTORY_START + timedelta(days=offset)
        for data_type, value in [("pv_power", 500), ("load_power", 600)]:
            response = client.post(
                "/api/v1/sites/S-API/measurements",
                json={
                    "data_type": data_type,
                    "unit": "kW",
                    "resolution_minutes": 15,
                    "points": day_points(day, value),
                },
            )
            assert response.status_code == 200, response.text


def post_schedule(client: TestClient, date: str, **extra) -> object:
    return client.post(
        SCHEDULE_URL, json={"site_id": "S-API", "date": date, "initial_soc": 0.5, **extra}
    )


def test_auto_uses_measurements_when_day_complete(site_client: TestClient) -> None:
    seed_history(site_client, days=4)
    response = post_schedule(site_client, "2026-09-16")  # 默认 source=auto
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data_source"] == "measurements"
    assert len(body["schedule"]) == 96
    listing = site_client.get("/api/v1/sites/S-API/dispatch-runs").json()
    assert listing[0]["data_source"] == "measurements"


def test_auto_falls_back_to_forecast_when_day_missing(site_client: TestClient) -> None:
    seed_history(site_client, days=4)
    response = post_schedule(site_client, "2026-09-18")  # 当天无数据，历史够 3 天
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["data_source"] == "forecast"
    assert len(body["schedule"]) == 96
    detail = site_client.get(f"/api/v1/sites/S-API/dispatch-runs/{body['run_id']}").json()
    assert detail["data_source"] == "forecast"


def test_auto_falls_back_when_day_has_null_values(site_client: TestClient) -> None:
    seed_history(site_client, days=4)
    # 当天 96 点但有 NULL 值 → 不算完整，回退预测
    target = HISTORY_START + timedelta(days=4)
    points = day_points(target, 500)
    points[10] = {"ts": points[10]["ts"], "value": None, "quality": "missing"}
    site_client.post(
        "/api/v1/sites/S-API/measurements",
        json={"data_type": "pv_power", "unit": "kW", "resolution_minutes": 15, "points": points},
    )
    site_client.post(
        "/api/v1/sites/S-API/measurements",
        json={
            "data_type": "load_power",
            "unit": "kW",
            "resolution_minutes": 15,
            "points": day_points(target, 600),
        },
    )
    response = post_schedule(site_client, "2026-09-18")
    assert response.status_code == 200, response.text
    assert response.json()["data_source"] == "forecast"


def test_auto_422_when_no_data_and_no_history(site_client: TestClient) -> None:
    seed_history(site_client, days=2)
    response = post_schedule(site_client, "2026-09-18")
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "DATA_QUALITY_ERROR"
    assert "实测数据不足且预测历史不足" in error["message"]


def test_explicit_measurements_still_strict(site_client: TestClient) -> None:
    seed_history(site_client, days=4)
    response = post_schedule(site_client, "2026-09-18", source="measurements")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_explicit_forecast_unchanged(site_client: TestClient) -> None:
    seed_history(site_client, days=4)
    # 09-17 当天有完整实测，显式 forecast 仍走预测（其前有 3 天历史）
    response = post_schedule(site_client, "2026-09-17", source="forecast")
    assert response.status_code == 200, response.text
    assert response.json()["data_source"] == "forecast"


def test_explicit_ai_curves_dispatch_without_training_history(site_client: TestClient) -> None:
    day = datetime(2026, 9, 18, tzinfo=CN_TZ)
    loads, pv = day_points(day, 600), day_points(day, 200)
    prices = [{"ts": p["ts"], "price": .3 if i < 48 else .9,
               "energy_price": .3 if i < 48 else .9, "tier": "flat"} for i, p in enumerate(loads)]
    response = post_schedule(site_client, "2026-09-18", source="forecast", price_model="ml",
                             predicted_loads=loads, predicted_pv=pv, predicted_prices=prices)
    assert response.status_code == 200, response.text
    assert len(response.json()["schedule"]) == 96
    bad = [{**p, "ts": (day+timedelta(days=1, minutes=i*15)).isoformat()} for i, p in enumerate(loads)]
    response = post_schedule(site_client, "2026-09-18", source="forecast", price_model="ml",
                             predicted_loads=bad, predicted_pv=pv, predicted_prices=prices)
    assert response.status_code == 422


def test_ml_dispatch_rejects_foreign_currency(site_client: TestClient) -> None:
    response = post_schedule(site_client, "2026-09-18", source="forecast", price_model="ml", price_unit="USD/MWh")
    assert response.status_code == 422
