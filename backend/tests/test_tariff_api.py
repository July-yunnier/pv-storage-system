from fastapi.testclient import TestClient


def test_get_tariff(site_client: TestClient) -> None:
    response = site_client.get("/api/v1/tariffs/T-DEFAULT")
    assert response.status_code == 200
    body = response.json()
    assert body["tariff_id"] == "T-DEFAULT"
    assert body["mode"] == "tou"
    assert len(body["tiers"]) == 4


def test_get_tariff_404(site_client: TestClient) -> None:
    response = site_client.get("/api/v1/tariffs/T-NOPE")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "NOT_FOUND"
    assert error["request_id"].startswith("req_")


def test_tariff_points(site_client: TestClient) -> None:
    response = site_client.get(
        "/api/v1/sites/S-API/tariff-points", params={"date": "2026-09-20"}
    )
    assert response.status_code == 200, response.text
    points = response.json()
    assert len(points) == 96
    assert points[0]["ts"] == "2026-09-20T00:00:00+08:00"
    assert points[-1]["ts"] == "2026-09-20T23:45:00+08:00"
    assert points[0]["tier"] == "valley"
    assert points[0]["price"] == round(0.35 + 0.21, 6)
    # 17:00 为尖峰起点（第 68 个 15 分钟点）
    assert points[68]["ts"] == "2026-09-20T17:00:00+08:00"
    assert points[68]["tier"] == "sharp"
    assert points[68]["price"] == round(1.25 + 0.21, 6)


def test_tariff_points_unknown_site_404(site_client: TestClient) -> None:
    response = site_client.get(
        "/api/v1/sites/S-NOPE/tariff-points", params={"date": "2026-09-20"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
