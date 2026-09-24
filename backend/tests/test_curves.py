from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ, day_points

CURVES_URL = "/api/v1/sites/S-API/curves"
START = datetime(2026, 9, 18, tzinfo=CN_TZ)


def upload(client: TestClient, data_type: str, points: list[dict], day: datetime) -> None:
    response = client.post(
        "/api/v1/sites/S-API/measurements",
        json={"data_type": data_type, "unit": "kW", "resolution_minutes": 15, "points": points},
    )
    assert response.status_code == 200, response.text


def test_curves_three_day_span(site_client: TestClient) -> None:
    for day_offset in range(3):
        day = START + timedelta(days=day_offset)
        upload(site_client, "pv_power", day_points(day, 100), day)
        upload(site_client, "load_power", day_points(day, 600), day)
    response = site_client.get(CURVES_URL, params={"start": "2026-09-18", "end": "2026-09-20"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["site_id"] == "S-API"
    assert body["start"] == "2026-09-18"
    assert body["end"] == "2026-09-20"
    assert body["resolution_minutes"] == 15
    assert len(body["points"]) == 288
    assert body["points"][0]["ts"] == "2026-09-18T00:00:00+08:00"
    assert body["points"][-1]["ts"] == "2026-09-20T23:45:00+08:00"
    first = body["points"][0]
    assert first["pv_power_kw"] == 100
    assert first["load_power_kw"] == 600
    assert first["net_load_kw"] == 500


def test_curves_missing_side_is_null(site_client: TestClient) -> None:
    # 只有 pv 有数据：pv 有值、load 为 null、net_load 为 null（不补 0）
    upload(site_client, "pv_power", day_points(START, 100), START)
    body = site_client.get(
        CURVES_URL, params={"start": "2026-09-18", "end": "2026-09-18"}
    ).json()
    point = body["points"][0]
    assert point["pv_power_kw"] == 100
    assert point["load_power_kw"] is None
    assert point["net_load_kw"] is None


def test_curves_missing_and_invalid_quality_become_null(site_client: TestClient) -> None:
    points = day_points(START, 100)
    points[4] = {"ts": points[4]["ts"], "value": 100, "quality": "invalid"}
    points[5] = {"ts": points[5]["ts"], "value": None, "quality": "missing"}
    points[6] = {"ts": points[6]["ts"], "value": 100, "quality": "estimated"}
    upload(site_client, "pv_power", points, START)
    upload(site_client, "load_power", day_points(START, 600), START)
    body = site_client.get(
        CURVES_URL, params={"start": "2026-09-18", "end": "2026-09-18"}
    ).json()
    assert body["points"][4]["pv_power_kw"] is None
    assert body["points"][4]["net_load_kw"] is None
    assert body["points"][5]["pv_power_kw"] is None
    # estimated 正常返回
    assert body["points"][6]["pv_power_kw"] == 100
    assert body["points"][6]["net_load_kw"] == 500


def test_curves_price_tiers(site_client: TestClient) -> None:
    body = site_client.get(
        CURVES_URL, params={"start": "2026-09-18", "end": "2026-09-18"}
    ).json()
    points = body["points"]
    assert len(points) == 96
    # T-DEFAULT：23:00-07:00 低谷、07:00-13:00 平段、17:00-19:00 尖峰
    assert points[0]["tier"] == "valley"
    assert points[0]["price"] == round(0.35 + 0.21, 6)
    assert points[27]["ts"] == "2026-09-18T06:45:00+08:00"
    assert points[27]["tier"] == "valley"
    assert points[28]["ts"] == "2026-09-18T07:00:00+08:00"
    assert points[28]["tier"] == "flat"
    assert points[28]["price"] == round(0.65 + 0.21, 6)
    assert points[68]["ts"] == "2026-09-18T17:00:00+08:00"
    assert points[68]["tier"] == "sharp"
    assert points[68]["price"] == round(1.25 + 0.21, 6)
    assert points[92]["tier"] == "valley"


def test_curves_span_too_long_422(site_client: TestClient) -> None:
    response = site_client.get(CURVES_URL, params={"start": "2026-09-01", "end": "2026-10-02"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    # 恰好 31 天允许
    ok = site_client.get(CURVES_URL, params={"start": "2026-09-01", "end": "2026-10-01"})
    assert ok.status_code == 200
    assert len(ok.json()["points"]) == 31 * 96


def test_curves_start_after_end_422(site_client: TestClient) -> None:
    response = site_client.get(CURVES_URL, params={"start": "2026-09-20", "end": "2026-09-18"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_curves_unknown_site_404(site_client: TestClient) -> None:
    response = site_client.get(
        "/api/v1/sites/S-NOPE/curves", params={"start": "2026-09-18", "end": "2026-09-20"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
