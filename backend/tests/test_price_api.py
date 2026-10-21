"""电价中心 API 集成测试：从无历史（政策表）到有历史（统计画像）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

CN_TZ = timezone(timedelta(hours=8))


def _spot_payload(hours: int = 72, base: float = 0.62) -> dict:
    end = datetime.now(CN_TZ).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours)
    points = []
    for index in range(hours):
        ts = start + timedelta(hours=index)
        # 造一个日内双峰：午间与傍晚贵，深夜便宜
        hour = ts.hour
        shape = 0.18 if hour in (11, 19) else (-0.12 if hour < 7 else 0.0)
        points.append(
            {
                "ts": ts.isoformat(),
                "price_cny_per_kwh": round(base + shape + 0.01 * (index % 5), 4),
                "quality": "measured",
            }
        )
    return {"source": "pytest", "points": points}


def test_price_forecast_falls_back_to_tariff_without_history(site_client):
    response = site_client.get("/api/v1/sites/S-API/price/forecast", params={"days": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "tou_policy"
    assert body["horizon_hours"] == 24
    assert len(body["points"]) == 96
    # 政策表是确定量：区间退化为一点
    assert body["points"][0]["p10"] == body["points"][0]["p50"]
    assert body["summary"]["peak_valley_spread"] >= 0


def test_price_upload_rejects_unsorted_points(site_client):
    payload = _spot_payload(48)
    payload["points"] = list(reversed(payload["points"]))
    response = site_client.post("/api/v1/sites/S-API/price/spot-prices", json=payload)
    assert response.status_code == 422


def test_price_forecast_uses_history_when_available(site_client):
    upload = site_client.post(
        "/api/v1/sites/S-API/price/spot-prices", json=_spot_payload(168)
    )
    assert upload.status_code == 200, upload.text
    assert upload.json()["accepted"] == 168

    history = site_client.get("/api/v1/sites/S-API/price/history", params={"days": 30})
    assert history.status_code == 200
    assert history.json()["stats"]["count"] > 0

    response = site_client.get(
        "/api/v1/sites/S-API/price/forecast", params={"days": 1, "model": "auto"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "price_diurnal_quantile"
    assert len(body["points"]) == 96
    assert body["uncertainty"] is True
    for point in body["points"]:
        assert point["p10"] <= point["p50"] <= point["p90"]
    assert body["summary"]["arbitrage"] is not None
    assert body["summary"]["arbitrage"]["net_spread"] is not None


def test_price_forecast_rejects_unknown_model_and_bad_resolution(site_client):
    bad = site_client.get(
        "/api/v1/sites/S-API/price/forecast", params={"model": "not-a-model"}
    )
    assert bad.status_code == 422
    bad_res = site_client.get(
        "/api/v1/sites/S-API/price/forecast", params={"resolution_minutes": 20}
    )
    assert bad_res.status_code == 422


def test_price_forecast_unknown_site(site_client):
    response = site_client.get("/api/v1/sites/NOPE/price/forecast")
    assert response.status_code == 404


@pytest.mark.parametrize("days", [1, 3])
def test_price_forecast_horizon_length(site_client, days):
    site_client.post("/api/v1/sites/S-API/price/spot-prices", json=_spot_payload(168))
    response = site_client.get(
        "/api/v1/sites/S-API/price/forecast", params={"days": days, "model": "statistical"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["points"]) == days * 96
    assert body["horizon_hours"] == days * 24
