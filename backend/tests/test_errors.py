from fastapi.testclient import TestClient


def test_unknown_site_returns_structured_404(site_client: TestClient) -> None:
    response = site_client.get("/api/v1/sites/S-DOES-NOT-EXIST")
    assert response.status_code == 404
    assert "X-Request-ID" in response.headers
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == "NOT_FOUND"
    assert "S-DOES-NOT-EXIST" in error["message"]
    assert error["request_id"].startswith("req_")
    assert error["request_id"] == response.headers["X-Request-ID"]


def test_invalid_payload_returns_structured_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sites",
        json={"name": "", "location": {"lat": 200, "lon": 118.7}},
    )
    assert response.status_code == 422
    assert "X-Request-ID" in response.headers
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["request_id"].startswith("req_")
    fields = {detail["field"] for detail in error["details"]}
    assert "name" in fields
    assert "location.lat" in fields
    assert all(detail["reason"] for detail in error["details"])


def test_duplicate_site_returns_structured_409(site_client: TestClient) -> None:
    from tests.conftest import SITE_PAYLOAD

    response = site_client.post("/api/v1/sites", json=SITE_PAYLOAD)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "CONFLICT"
    assert error["request_id"].startswith("req_")


def test_schedule_on_missing_site_returns_404(site_client: TestClient) -> None:
    response = site_client.post(
        "/api/v1/storage/schedule",
        json={"site_id": "S-NOPE", "date": "2026-09-20"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
