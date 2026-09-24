from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from tests.conftest import CN_TZ

CSV_URL = "/api/v1/sites/S-API/measurements/csv"


def make_csv(start: datetime, count: int, value: float = 100.0) -> str:
    lines = ["ts,value"]
    for index in range(count):
        ts = (start + timedelta(minutes=15 * index)).isoformat()
        lines.append(f"{ts},{value}")
    return "\n".join(lines)


def post_csv(client: TestClient, content: str | bytes, data_type: str = "pv_power", site_id: str = "S-API"):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return client.post(
        f"/api/v1/sites/{site_id}/measurements/csv",
        files={"file": ("data.csv", content, "text/csv")},
        data={"data_type": data_type},
    )


def test_csv_upload_happy_path(site_client: TestClient) -> None:
    start = datetime(2026, 9, 20, tzinfo=CN_TZ)
    # 带 BOM 和多余空格也应被容忍
    content = "\ufeff ts , value \n" + "\n".join(
        f"  {(start + timedelta(minutes=15 * i)).isoformat()} , {50 + i} " for i in range(96)
    )
    response = post_csv(site_client, content)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "site_id": "S-API",
        "data_type": "pv_power",
        "total_rows": 96,
        "accepted": 96,
        "rejected": [],
    }
    points = site_client.get(
        "/api/v1/sites/S-API/measurements",
        params={"data_type": "pv_power", "date": "2026-09-20"},
    ).json()
    assert len(points) == 96
    assert points[0]["value"] == 50
    assert points[0]["source"] == "csv"


def test_csv_upload_partial_rejection(site_client: TestClient) -> None:
    start = datetime(2026, 9, 20, tzinfo=CN_TZ)
    lines = ["ts,value", f"{start.isoformat()},10", "not-a-timestamp,20", f"{start.isoformat()},30"]
    lines.append(f"{(start + timedelta(minutes=15)).isoformat()},abc")
    lines.append(f"{(start + timedelta(minutes=30)).isoformat()},-5")
    lines.append(f"{(start + timedelta(minutes=45)).isoformat()},")
    response = post_csv(site_client, "\n".join(lines))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_rows"] == 6
    assert body["accepted"] == 2  # 第 2 行正常 + 空值缺失点
    reasons = {item["row"]: item["reason"] for item in body["rejected"]}
    assert set(reasons) == {3, 4, 5, 6}
    assert "ISO8601" in reasons[3]
    assert "duplicate ts" in reasons[4]
    assert "numeric" in reasons[5]
    assert "negative" in reasons[6]


def test_csv_empty_file_rejected(site_client: TestClient) -> None:
    response = post_csv(site_client, b"")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_csv_header_only_rejected(site_client: TestClient) -> None:
    response = post_csv(site_client, "ts,value\n")
    assert response.status_code == 422
    assert "no data rows" in response.json()["error"]["message"]


def test_csv_wrong_header_rejected(site_client: TestClient) -> None:
    response = post_csv(site_client, "time,kwh\n2026-09-20T00:00:00+08:00,1\n")
    assert response.status_code == 422
    assert "header" in response.json()["error"]["message"]


def test_csv_all_rows_invalid_rejected(site_client: TestClient) -> None:
    response = post_csv(site_client, "ts,value\nbad,1\nworse,2\n")
    assert response.status_code == 422
    assert "all 2 data rows are invalid" in response.json()["error"]["message"]


def test_csv_invalid_data_type(site_client: TestClient) -> None:
    response = post_csv(site_client, "ts,value\n2026-09-20T00:00:00+08:00,1\n", data_type="sunshine")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_csv_unknown_site_404(site_client: TestClient) -> None:
    response = post_csv(
        site_client, "ts,value\n2026-09-20T00:00:00+08:00,1\n", site_id="S-NOPE"
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_csv_soc_out_of_range_row_rejected(site_client: TestClient) -> None:
    response = post_csv(
        site_client,
        "ts,value\n2026-09-20T00:00:00+08:00,1.5\n2026-09-20T00:15:00+08:00,0.5\n",
        data_type="storage_soc",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] == 1
    assert body["rejected"] == [{"row": 2, "reason": "storage_soc value must be between 0 and 1"}]
