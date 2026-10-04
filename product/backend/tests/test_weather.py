"""天气 Provider 与接口测试：mock 确定性、契约字段、Provider 失败转 502。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.infrastructure.weather.mock import MockWeatherProvider


class TestMockWeatherProvider:
    def test_deterministic_shape(self) -> None:
        provider = MockWeatherProvider()
        points = provider.irradiance_forecast(
            lat=32.06, lon=118.79, timezone_name="Asia/Shanghai", days=2
        )
        assert len(points) == 48
        # 夜间为 0，正午有值，能量守恒：dni 与 dhi 不超过 ghi
        assert points[0].ghi_wm2 == 0.0
        assert points[12].ghi_wm2 > 0
        for point in points:
            assert point.ghi_wm2 >= point.dni_wm2 >= 0
            assert point.dhi_wm2 >= 0
        # 时区带偏移
        assert points[0].ts.utcoffset() is not None

    def test_reproducible(self) -> None:
        provider = MockWeatherProvider()
        kwargs = dict(lat=32.06, lon=118.79, timezone_name="Asia/Shanghai", days=1)
        first = provider.irradiance_forecast(**kwargs)
        second = provider.irradiance_forecast(**kwargs)
        assert [p.ghi_wm2 for p in first] == [p.ghi_wm2 for p in second]


@pytest.fixture()
def mock_weather(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEATHER_PROVIDER", "mock")


def test_weather_endpoint_contract(site_client: TestClient, mock_weather: None) -> None:
    response = site_client.get("/api/v1/sites/S-API/weather?days=2")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == {"name": "mock_weather", "version": "v1"}
    assert body["resolution_minutes"] == 60
    assert body["days"] == 2
    assert len(body["points"]) == 48
    point = body["points"][12]
    assert {"ts", "ghi_wm2", "dni_wm2", "dhi_wm2", "temperature_c", "wind_speed_ms"} <= set(point)


def test_weather_site_not_found(client: TestClient, mock_weather: None) -> None:
    response = client.get("/api/v1/sites/S-NOPE/weather")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_weather_days_validation(site_client: TestClient, mock_weather: None) -> None:
    response = site_client.get("/api/v1/sites/S-API/weather?days=99")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_weather_provider_failure_is_502(
    site_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingProvider:
        name = "open_meteo"
        version = "v1"

        def irradiance_forecast(self, **kwargs):  # type: ignore[no-untyped-def]
            from app.api.errors import provider_error

            raise provider_error("open-meteo unavailable: connection refused")

    from app.api import routes_weather

    monkeypatch.setattr(routes_weather, "get_weather_provider", lambda: FailingProvider())
    response = site_client.get("/api/v1/sites/S-API/weather")
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "PROVIDER_ERROR"
    assert "request_id" in body["error"]
