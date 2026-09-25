"""Open-Meteo 实现：逐小时辐射预报。

免费层约定：非商用、10k 次/天、无 SLA。生产部署前需确认商用许可或自建。
"""

from __future__ import annotations

from datetime import datetime

import httpx

from app.api.errors import provider_error
from app.infrastructure.db import _zone
from app.infrastructure.weather.base import WeatherPoint

API_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m"


class OpenMeteoProvider:
    name = "open_meteo"
    version = "v1"

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout = timeout_seconds

    def irradiance_forecast(
        self,
        *,
        lat: float,
        lon: float,
        timezone_name: str,
        days: int,
    ) -> list[WeatherPoint]:
        return self.irradiance_window(
            lat=lat, lon=lon, timezone_name=timezone_name, past_days=0, forecast_days=days
        )

    def irradiance_window(
        self,
        *,
        lat: float,
        lon: float,
        timezone_name: str,
        past_days: int,
        forecast_days: int,
    ) -> list[WeatherPoint]:
        """过去 past_days 天（模型分析值）+ 未来 forecast_days 天的连续逐小时辐射。"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": HOURLY_VARS,
            "timezone": timezone_name,
            "forecast_days": forecast_days,
            "past_days": past_days,
        }
        data = self._request(params)
        return self._parse(data, timezone_name)

    def _request(self, params: dict) -> dict:
        last_exc: Exception | None = None
        for _ in range(2):  # 传输层错误重试一次
            try:
                response = httpx.get(API_URL, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                if response.status_code in (400, 401, 403, 404, 429):
                    raise provider_error(
                        f"open-meteo request rejected: HTTP {response.status_code} {response.text[:200]}"
                    )
                last_exc = provider_error(f"open-meteo upstream error: HTTP {response.status_code}")
            except httpx.HTTPError as exc:
                last_exc = exc
        raise provider_error(f"open-meteo unavailable: {last_exc}")

    def _parse(self, data: dict, timezone_name: str) -> list[WeatherPoint]:
        try:
            hourly = data["hourly"]
            times = hourly["time"]
        except (KeyError, TypeError) as exc:
            raise provider_error(f"open-meteo unexpected payload: {exc}") from exc
        tz = _zone(timezone_name)
        points: list[WeatherPoint] = []
        for index, raw_ts in enumerate(times):
            ts = datetime.fromisoformat(raw_ts).replace(tzinfo=tz)  # API 按时区返回无时点偏移的本地时间
            points.append(
                WeatherPoint(
                    ts=ts,
                    ghi_wm2=hourly["shortwave_radiation"][index] or 0.0,
                    dni_wm2=hourly["direct_normal_irradiance"][index] or 0.0,
                    dhi_wm2=hourly["diffuse_radiation"][index] or 0.0,
                    temperature_c=hourly.get("temperature_2m", [None] * len(times))[index],
                    wind_speed_ms=hourly.get("wind_speed_10m", [None] * len(times))[index],
                )
            )
        return points
