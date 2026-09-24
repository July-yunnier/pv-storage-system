from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ, day_points

QUALITY_URL = "/api/v1/sites/S-API/data-quality"
START = datetime(2026, 9, 20, tzinfo=CN_TZ)


def upload_points(client: TestClient, points: list[dict], data_type: str = "pv_power") -> None:
    response = client.post(
        f"/api/v1/sites/S-API/measurements",
        json={"data_type": data_type, "unit": "kW", "resolution_minutes": 15, "points": points},
    )
    assert response.status_code == 200, response.text


def get_quality(client: TestClient, data_type: str = "pv_power", date: str = "2026-09-20") -> dict:
    response = client.get(QUALITY_URL, params={"data_type": data_type, "date": date})
    assert response.status_code == 200, response.text
    return response.json()


def test_quality_complete(site_client: TestClient) -> None:
    upload_points(site_client, day_points(START, 100))
    body = get_quality(site_client)
    assert body["status"] == "complete"
    assert body["expected_points"] == 96
    assert body["resolution_minutes"] == 15
    assert body["measured"] == 96
    assert body["missing"] == 0
    assert body["completeness"] == 1.0
    assert body["missing_ranges"] == []


def test_quality_partial_with_merged_missing_ranges(site_client: TestClient) -> None:
    points = day_points(START, 100)
    # 挖掉两段：连续 3 点（10,11,12 → 02:30~03:00）和单独 1 点（20 → 05:00）
    dropped = [points[i] for i in (10, 11, 12, 20)]
    kept = [point for i, point in enumerate(points) if i not in (10, 11, 12, 20)]
    upload_points(site_client, kept)
    body = get_quality(site_client)
    assert body["status"] == "partial"
    assert body["measured"] == 92
    assert body["missing"] == 4
    assert body["completeness"] == round(92 / 96, 4)
    assert body["missing_ranges"] == [
        {"start": dropped[0]["ts"], "end": dropped[2]["ts"]},
        {"start": dropped[3]["ts"], "end": dropped[3]["ts"]},
    ]


def test_quality_empty(site_client: TestClient) -> None:
    body = get_quality(site_client)
    assert body["status"] == "empty"
    assert body["expected_points"] == 96
    assert body["resolution_minutes"] == 15
    assert body["measured"] == 0
    assert body["missing"] == 96
    assert body["completeness"] == 0.0
    assert len(body["missing_ranges"]) == 1
    assert body["missing_ranges"][0]["start"] == "2026-09-20T00:00:00+08:00"
    assert body["missing_ranges"][0]["end"] == "2026-09-20T23:45:00+08:00"


def test_quality_null_value_counts_as_missing(site_client: TestClient) -> None:
    points = day_points(START, 100)
    points[5] = {"ts": points[5]["ts"], "value": None, "quality": "missing"}
    points[6] = {"ts": points[6]["ts"], "value": 200, "quality": "invalid"}
    points[7] = {"ts": points[7]["ts"], "value": 200, "quality": "estimated"}
    upload_points(site_client, points)
    body = get_quality(site_client)
    assert body["measured"] == 93
    assert body["estimated"] == 1
    assert body["invalid"] == 1
    assert body["missing"] == 1
    assert body["missing_ranges"] == [
        {"start": points[5]["ts"], "end": points[5]["ts"]}
    ]
    assert body["status"] == "partial"


def test_quality_measured_zero_with_gaps_is_empty(site_client: TestClient) -> None:
    # missing==0 时优先 complete；measured==0 且仍有缺失点才判 empty
    points = [
        {"ts": point["ts"], "value": point["value"], "quality": "estimated"}
        for point in day_points(START, 100)
    ]
    upload_points(site_client, points[:-1])
    body = get_quality(site_client)
    assert body["measured"] == 0
    assert body["estimated"] == 95
    assert body["missing"] == 1
    assert body["status"] == "empty"


def test_quality_all_estimated_no_gaps_is_complete(site_client: TestClient) -> None:
    points = [
        {"ts": point["ts"], "value": point["value"], "quality": "estimated"}
        for point in day_points(START, 100)
    ]
    upload_points(site_client, points)
    body = get_quality(site_client)
    assert body["measured"] == 0
    assert body["missing"] == 0
    assert body["status"] == "complete"


def test_quality_unknown_site_404(site_client: TestClient) -> None:
    response = site_client.get(
        "/api/v1/sites/S-NOPE/data-quality",
        params={"data_type": "pv_power", "date": "2026-09-20"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_quality_invalid_data_type_422(site_client: TestClient) -> None:
    response = site_client.get(
        QUALITY_URL, params={"data_type": "sunshine", "date": "2026-09-20"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
