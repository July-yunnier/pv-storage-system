"""物理链 / 融合 / 日内修正测试。日期一律相对今天生成，避免测试随时间腐烂。

气象源由 conftest 全局 mock（确定性晴空钟形），不打外网。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi.testclient import TestClient

from app.domain.forecast.blend import blend_pv_forecast
from app.domain.forecast.pv import predict_pv_intraday
from app.domain.models import PVConfig
from app.infrastructure.weather.mock import MockWeatherProvider
from tests.conftest import CN_TZ, day_points

PV = PVConfig(capacity_kwp=1000, tilt_deg=22.5, azimuth_deg=180, system_loss=0.14)
LAT, LON = 32.06, 118.79
TODAY = date.today()
TOMORROW = TODAY + timedelta(days=1)


def mock_weather(past_days: int = 0, forecast_days: int = 2):
    return MockWeatherProvider().irradiance_window(
        lat=LAT,
        lon=LON,
        timezone_name="Asia/Shanghai",
        past_days=past_days,
        forecast_days=forecast_days,
    )


def measured_from_mock(days_back: int = 5) -> list[tuple[datetime, float]]:
    """用 mock 辐射经物理链生成的"实测"历史（和模型同源，校准系数应接近 1）。"""
    from app.domain.forecast.pv_physics import day_power_values

    history: list[tuple[datetime, float]] = []
    weather = mock_weather(past_days=days_back, forecast_days=0)
    for offset in range(days_back, 0, -1):
        day = TODAY - timedelta(days=offset)
        values = day_power_values(weather, PV, LAT, LON, day, CN_TZ)
        for slot, value in enumerate(values):
            ts = datetime.combine(day, datetime.min.time(), tzinfo=CN_TZ) + timedelta(
                minutes=15 * slot
            )
            history.append((ts, value))
    return history


class TestPhysicsChain:
    def test_cold_start_forecast_no_history(self, site_client: TestClient) -> None:
        """无历史数据：auto → 纯物理链，冷启动可出预测。"""
        response = site_client.post(
            "/api/v1/forecast",
            json={
                "site_id": "S-API",
                "targets": ["pv_power"],
                "horizon_hours": 24,
                "resolution_minutes": 15,
                "options": {"pv_model": "auto"},
            },
        )
        assert response.status_code == 200, response.text
        result = response.json()["results"][0]
        assert result["model"]["name"] == "pvlib_pvwatts"
        assert len(result["points"]) == 96
        values = [p["value"] for p in result["points"]]
        assert max(values) > 0
        assert all(v == 0 for v in values[:20])  # 凌晨为 0
        assert result["accuracy"] is None
        assert "物理模型" in result["notes"]

    def test_explicit_physics_model(self, site_client: TestClient) -> None:
        response = site_client.post(
            "/api/v1/forecast",
            json={
                "site_id": "S-API",
                "targets": ["pv_power"],
                "horizon_hours": 24,
                "resolution_minutes": 15,
                "options": {"pv_model": "physics"},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["model"]["name"] == "pvlib_pvwatts"


class TestBlend:
    def test_factor_near_one_when_sources_agree(self) -> None:
        history = measured_from_mock(days_back=5)
        weather = mock_weather(past_days=12, forecast_days=2)
        past = [p for p in weather if p.ts.date() < TOMORROW]
        future = [p for p in weather if p.ts.date() >= TOMORROW]
        result = blend_pv_forecast(
            past, future, history, PV, LAT, LON, TOMORROW, CN_TZ
        )
        assert result.model.name == "pvlib_blend"
        assert result.accuracy is not None
        # 同源数据：融合结果应接近纯物理输出（校准系数 ≈ 1）
        noon = result.points[48].value
        assert 500 < noon <= 1000
        assert result.accuracy.value > 0.95  # 同源回测应很准
        assert result.p10 and result.p90
        assert all(
            lo <= p.value <= hi
            for lo, p, hi in zip(result.p10, result.points, result.p90)
        )

    def test_blend_api_with_history(self, site_client: TestClient) -> None:
        """有历史 + 气象可用：auto → blend。"""
        history = measured_from_mock(days_back=4)
        for day in sorted({ts.date() for ts, _ in history}):
            points = [
                {"ts": ts.isoformat(), "value": value}
                for ts, value in history
                if ts.date() == day
            ]
            response = site_client.post(
                "/api/v1/sites/S-API/measurements",
                json={
                    "data_type": "pv_power",
                    "unit": "kW",
                    "resolution_minutes": 15,
                    "points": points,
                },
            )
            assert response.status_code == 200, response.text
        response = site_client.post(
            "/api/v1/forecast",
            json={"site_id": "S-API", "targets": ["pv_power"], "horizon_hours": 24},
        )
        assert response.status_code == 200, response.text
        result = response.json()["results"][0]
        assert result["model"]["name"] == "pvlib_blend"
        assert "校准系数" in result["notes"]


class TestIntraday:
    def test_intraday_deviation_correction(self) -> None:
        """今日上午实测偏低 → 剩余时段预测应低于基线。"""
        history: list[tuple[datetime, float]] = []
        for offset in range(5, 0, -1):  # 5 天基线：中午 800
            day = TODAY - timedelta(days=offset)
            for slot in range(96):
                ts = datetime.combine(day, datetime.min.time(), tzinfo=CN_TZ) + timedelta(
                    minutes=15 * slot
                )
                history.append((ts, 800.0 if 10 <= ts.hour <= 14 else 0.0))
        # 今日 10:00-10:15 实测只有 400（偏低）
        issued = datetime.combine(TODAY, datetime.min.time(), tzinfo=CN_TZ).replace(hour=10)
        for slot in [40, 41]:
            ts = datetime.combine(TODAY, datetime.min.time(), tzinfo=CN_TZ) + timedelta(
                minutes=15 * slot
            )
            history.append((ts, 400.0))

        result = predict_pv_intraday(history, issued, cap_kw=1000, tz=CN_TZ)
        assert result.model.name == "pv_intraday_deviation"
        # 偏差 = 400 - 800 = -400，alpha=0.25 → 修正 -100；中午基线 800 → 预测 700
        noon_point = next(p for p in result.points if p.ts.hour == 12 and p.ts.minute == 0)
        assert noon_point.value == 700.0
        # 只覆盖今日剩余时段
        assert all(p.ts > issued for p in result.points)
        assert all(p.ts.date() == TODAY for p in result.points)

    def test_intraday_slot_includes_issued_minute(self) -> None:
        history: list[tuple[datetime, float]] = []
        for offset in range(3, 0, -1):
            day = TODAY - timedelta(days=offset)
            for slot in range(96):
                ts = datetime.combine(day, datetime.min.time(), tzinfo=CN_TZ) + timedelta(minutes=15 * slot)
                history.append((ts, 800.0 if 10 <= ts.hour <= 14 else 0.0))
        issued = datetime.combine(TODAY, datetime.min.time(), tzinfo=CN_TZ).replace(hour=10, minute=15)
        result = predict_pv_intraday(history, issued, cap_kw=1000, tz=CN_TZ)
        assert result.points[0].ts.hour == 10 and result.points[0].ts.minute == 30

    def test_intraday_requires_history(self) -> None:
        import pytest

        issued = datetime.combine(TODAY, datetime.min.time(), tzinfo=CN_TZ).replace(hour=10)
        with pytest.raises(ValueError, match="至少 3 天"):
            predict_pv_intraday([], issued, cap_kw=1000, tz=CN_TZ)
