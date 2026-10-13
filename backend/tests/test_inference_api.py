"""推理网关路由测试：网关客户端打桩，验证路由/错误映射/默认值光伏链不受影响。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.infrastructure import pv_inference_gateway as gateway

SAMPLE_RESULT = {
    "model_id": "state_grid_1",
    "model_sha256": "22c65db50b85",
    "issued_at": "2020-05-26T07:45:00+00:00",
    "units": "kW",
    "forecasts": [
        {"horizon_minutes": 15, "target_at": "2020-05-26T07:45:00+08:00",
         "predicted_kw": 123.4, "p10_kw": 100.0, "p50_kw": 123.4, "p90_kw": 150.0},
        {"horizon_minutes": 60, "target_at": "2020-05-26T08:30:00+08:00",
         "predicted_kw": 456.7, "p10_kw": 400.0, "p50_kw": 456.7, "p90_kw": 520.0},
        {"horizon_minutes": 240, "target_at": "2020-05-26T11:45:00+08:00",
         "predicted_kw": 789.0, "p10_kw": 700.0, "p50_kw": 789.0, "p90_kw": 880.0},
    ],
}


@pytest.fixture()
def stub_gateway(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def fake_predict(model_id, request):
        calls["predict"] = (model_id, request)
        return SAMPLE_RESULT

    monkeypatch.setattr(gateway, "predict", fake_predict)
    monkeypatch.setattr(gateway, "list_models",
                        lambda: [{"model_id": "state_grid_1", "cap_kw": 50000,
                                  "sha256": "22c65db50b85", "loaded": False,
                                  "horizons_minutes": [15, 60, 240]}])
    monkeypatch.setattr(gateway, "health",
                        lambda: {"status": "ok", "python": "3.12.13",
                                 "models_loaded": [], "models_registered": 10})
    monkeypatch.setattr(gateway, "unload",
                        lambda model_id: {"model_id": model_id, "unloaded": True,
                                          "models_loaded": []})
    return calls


def test_inference_models(client: TestClient, stub_gateway) -> None:
    response = client.get("/api/v1/inference/models")
    assert response.status_code == 200, response.text
    models = response.json()["models"]
    assert models[0]["model_id"] == "state_grid_1"
    assert models[0]["sha256"]


def test_inference_health(client: TestClient, stub_gateway) -> None:
    response = client.get("/api/v1/inference/health")
    assert response.status_code == 200, response.text
    assert response.json()["models_registered"] == 10


def test_inference_predict_routes_request(client: TestClient, stub_gateway) -> None:
    body = {
        "issued_at": "2020-05-26T07:45:00Z",
        "observations": [
            {"ts": "2020-05-26T00:00:00+08:00", "power_kw": 0.0, "tsi": 0.0,
             "dni": 1.0, "ghi": 0.0, "temperature": 19.1}
        ],
    }
    response = client.post("/api/v1/inference/pv/state_grid_1", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert [f["horizon_minutes"] for f in result["forecasts"]] == [15, 60, 240]
    model_id, forwarded = stub_gateway["predict"]
    assert model_id == "state_grid_1"
    assert forwarded["issued_at"] == body["issued_at"]
    assert forwarded["observations"] == body["observations"]


def test_inference_predict_requires_body(client: TestClient, stub_gateway) -> None:
    response = client.post("/api/v1/inference/pv/state_grid_1",
                           json={"issued_at": "2020-05-26T07:45:00Z"})
    assert response.status_code == 422


def test_inference_unload(client: TestClient, stub_gateway) -> None:
    response = client.post("/api/v1/inference/models/state_grid_1/unload")
    assert response.status_code == 200, response.text
    assert response.json()["unloaded"] is True


def test_sidecar_down_maps_to_502(client: TestClient, monkeypatch) -> None:
    def down(model_id, request):
        raise gateway.InferenceGatewayUnavailable("PV 推理侧车不可达")

    monkeypatch.setattr(gateway, "predict", down)
    response = client.post(
        "/api/v1/inference/pv/state_grid_1",
        json={"issued_at": "2020-05-26T07:45:00Z", "observations": [{"ts": "t"}]},
    )
    assert response.status_code == 502
    assert "侧车" in response.text


def test_contract_violation_maps_to_422(client: TestClient, monkeypatch) -> None:
    def bad(model_id, request):
        raise ValueError("观测最后时间戳必须恰等于 issued_at")

    monkeypatch.setattr(gateway, "predict", bad)
    response = client.post(
        "/api/v1/inference/pv/state_grid_1",
        json={"issued_at": "2020-05-26T07:45:00Z", "observations": [{"ts": "t"}]},
    )
    assert response.status_code == 422
    assert "issued_at" in response.text


def test_default_pv_forecast_chain_untouched(site_client: TestClient) -> None:
    """默认光伏链路（auto）不经过推理网关，不装 sklearn/lightgbm 也照常工作。"""
    response = site_client.post(
        "/api/v1/forecast",
        json={"site_id": "S-API", "targets": ["pv_power"], "horizon_hours": 24},
    )
    assert response.status_code == 200, response.text
    assert len(response.json()["results"][0]["points"]) == 96
