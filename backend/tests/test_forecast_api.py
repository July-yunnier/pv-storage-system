from datetime import datetime, time, timedelta

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ, day_points


def _today() -> datetime:
    return datetime.combine(datetime.now(CN_TZ).date(), time.min, tzinfo=CN_TZ)


# 历史窗口随当前日期滚动：mock 气象只覆盖今天前后，写死日期会随时间漂移失效
HISTORY_START = _today() - timedelta(days=4)  # 4 天历史：今天-4 ~ 今天-1
TARGET_DATE = _today().date().isoformat()
ISSUED_AT = (_today() - timedelta(days=1)).strftime("%Y-%m-%dT18:00:00+08:00")


def seed_history(client: TestClient, days: int = 4) -> None:
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


def post_forecast(client: TestClient, **overrides) -> object:
    payload = {
        "site_id": "S-API",
        "targets": ["pv_power", "load_power", "price"],
        "horizon_hours": 24,
        "resolution_minutes": 15,
        "issued_at": ISSUED_AT,
        # 统计法路径的契约测试固定 pv_model，避免 auto 路由到 blend/physics
        "options": {"pv_model": "statistical"},
    }
    payload.update(overrides)
    return client.post("/api/v1/forecast", json=payload)


def test_forecast_three_targets_contract(site_client: TestClient) -> None:
    seed_history(site_client)
    response = post_forecast(site_client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["site_id"] == "S-API"
    assert body["resolution_minutes"] == 15
    assert body["issued_at"].startswith(ISSUED_AT[:16])
    assert [r["target"] for r in body["results"]] == ["pv_power", "load_power", "price"]

    for result in body["results"]:
        assert len(result["points"]) == 96
        assert result["points"][0]["ts"] == f"{TARGET_DATE}T00:00:00+08:00"
        assert result["model"]["name"]
        assert result["notes"]
    pv, load, price = body["results"]
    assert pv["unit"] == "kW"
    assert all(p["value"] == 500 for p in pv["points"])
    assert pv["accuracy"]["caliber"]["normalization"] == "cap"
    assert pv["p10"] and pv["p90"]
    assert all(p["value"] == 600 for p in load["points"])
    assert load["accuracy"]["caliber"]["normalization"] == "mean_power"
    assert price["unit"] == "CNY/kWh"
    assert price["accuracy"] is None
    assert price["p10"] == price["p90"] == [p["value"] for p in price["points"]]


def test_forecast_horizon_48_hours(site_client: TestClient) -> None:
    seed_history(site_client)
    response = post_forecast(site_client, horizon_hours=48, targets=["price"])
    assert response.status_code == 200, response.text
    points = response.json()["results"][0]["points"]
    assert len(points) == 192
    assert points[0]["ts"] == f"{TARGET_DATE}T00:00:00+08:00"
    day_after = (_today() + timedelta(days=1)).date().isoformat()
    assert points[-1]["ts"] == f"{day_after}T23:45:00+08:00"


def test_forecast_insufficient_history_422(site_client: TestClient) -> None:
    seed_history(site_client, days=2)
    response = post_forecast(site_client)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "DATA_QUALITY_ERROR"
    assert "至少 3 天" in error["message"]


def test_forecast_invalid_horizon_and_resolution(site_client: TestClient) -> None:
    seed_history(site_client)
    assert post_forecast(site_client, horizon_hours=25).status_code == 422
    assert post_forecast(site_client, horizon_hours=96).status_code == 422
    assert post_forecast(site_client, resolution_minutes=30).status_code == 422


def test_forecast_unknown_site_404(site_client: TestClient) -> None:
    response = post_forecast(site_client, site_id="S-NOPE")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_forecast_without_uncertainty(site_client: TestClient) -> None:
    seed_history(site_client)
    response = post_forecast(
        site_client,
        targets=["pv_power"],
        options={"include_uncertainty": False, "pv_model": "statistical"},
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["p10"] is None and result["p90"] is None
    assert result["accuracy"] is not None


def test_schedule_with_forecast_source(site_client: TestClient) -> None:
    seed_history(site_client)
    response = site_client.post(
        "/api/v1/storage/schedule",
        json={
            "site_id": "S-API",
            "date": TARGET_DATE,
            "initial_soc": 0.5,
            "source": "forecast",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["schedule"]) == 96
    assert body["data_source"] == "forecast"
    assert body["run_id"].startswith("R")

    listing = site_client.get("/api/v1/sites/S-API/dispatch-runs").json()
    assert listing[0]["data_source"] == "forecast"
    detail = site_client.get(
        f"/api/v1/sites/S-API/dispatch-runs/{body['run_id']}"
    ).json()
    assert detail["data_source"] == "forecast"
    assert len(detail["schedule"]) == 96


def test_schedule_measurements_source_default(site_client: TestClient) -> None:
    seed_history(site_client, days=1)
    # measurements 源需要当天数据
    for data_type, value in [("pv_power", 500), ("load_power", 600)]:
        day = _today()
        site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(day, value),
            },
        )
    response = site_client.post(
        "/api/v1/storage/schedule",
        json={"site_id": "S-API", "date": TARGET_DATE, "initial_soc": 0.5},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data_source"] == "measurements"
    listing = site_client.get("/api/v1/sites/S-API/dispatch-runs").json()
    assert listing[0]["data_source"] == "measurements"


def test_schedule_forecast_source_insufficient_history(site_client: TestClient) -> None:
    seed_history(site_client, days=2)
    response = site_client.post(
        "/api/v1/storage/schedule",
        json={
            "site_id": "S-API",
            "date": TARGET_DATE,
            "initial_soc": 0.5,
            "source": "forecast",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DATA_QUALITY_ERROR"
