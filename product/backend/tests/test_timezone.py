from fastapi.testclient import TestClient


def upload_point(client: TestClient, ts: str, value: float = 100.0) -> None:
    response = client.post(
        "/api/v1/sites/S-API/measurements",
        json={
            "data_type": "pv_power",
            "unit": "kW",
            "resolution_minutes": 15,
            "points": [{"ts": ts, "value": value}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["accepted"] == 1


def query_day(client: TestClient, target_date: str) -> list[dict]:
    response = client.get(
        "/api/v1/sites/S-API/measurements",
        params={"data_type": "pv_power", "date": target_date},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_same_instant_with_different_offsets_deduplicates(site_client: TestClient) -> None:
    upload_point(site_client, "2026-09-20T12:00:00+08:00", value=100)
    upload_point(site_client, "2026-09-20T13:00:00+09:00", value=120)
    points = query_day(site_client, "2026-09-20")
    assert len(points) == 1
    # 后上传的等价时间戳通过 ON CONFLICT 覆盖
    assert points[0]["value"] == 120
    assert points[0]["ts"] == "2026-09-20T12:00:00+08:00"


def test_site_date_query_finds_points_uploaded_with_other_offsets(
    site_client: TestClient,
) -> None:
    # 2026-09-19T18:00:00Z 即站点时区 2026-09-20T02:00，属于 9 月 20 日
    upload_point(site_client, "2026-09-19T18:00:00+00:00")
    # 2026-09-21T00:30:00+09:00 即站点时区 2026-09-20T23:30，也属于 9 月 20 日
    upload_point(site_client, "2026-09-21T00:30:00+09:00")
    # 2026-09-20T16:00:00Z 即站点时区 2026-09-21T00:00，属于 9 月 21 日
    upload_point(site_client, "2026-09-20T16:00:00+00:00")

    points = query_day(site_client, "2026-09-20")
    assert [point["ts"] for point in points] == [
        "2026-09-20T02:00:00+08:00",
        "2026-09-20T23:30:00+08:00",
    ]

    next_day = query_day(site_client, "2026-09-21")
    assert [point["ts"] for point in next_day] == ["2026-09-21T00:00:00+08:00"]


def test_returned_timestamps_carry_site_timezone_offset(site_client: TestClient) -> None:
    upload_point(site_client, "2026-09-20T05:00:00+00:00")
    points = query_day(site_client, "2026-09-20")
    assert len(points) == 1
    assert points[0]["ts"].endswith("+08:00")
