"""确定性 Mock 天气 Provider：晴空钟形曲线，供单测和离线开发。

不依赖任何外部服务；同一参数永远返回同一结果。
"""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta

from app.infrastructure.db import _zone
from app.infrastructure.weather.base import WeatherPoint

SUNRISE_HOUR = 6.0
SUNSET_HOUR = 18.0
PEAK_GHI_WM2 = 950.0


class MockWeatherProvider:
    name = "mock_weather"
    version = "v1"

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
        tz = _zone(timezone_name)
        today = datetime.now(tz).date()
        start = datetime.combine(today - timedelta(days=past_days), time.min, tzinfo=tz)
        total_hours = 24 * (past_days + forecast_days)
        points: list[WeatherPoint] = []
        for hour_index in range(total_hours):
            ts = start + timedelta(hours=hour_index)
            hour = ts.hour
            if SUNRISE_HOUR <= hour < SUNSET_HOUR:
                ratio = math.sin(math.pi * (hour - SUNRISE_HOUR) / (SUNSET_HOUR - SUNRISE_HOUR))
                ghi = round(PEAK_GHI_WM2 * ratio, 1)
                dni = round(ghi * 0.75, 1)
                dhi = round(ghi * 0.25, 1)
            else:
                ghi = dni = dhi = 0.0
            points.append(
                WeatherPoint(
                    ts=ts,
                    ghi_wm2=ghi,
                    dni_wm2=dni,
                    dhi_wm2=dhi,
                    temperature_c=25.0,
                    humidity_pct=60.0,
                    wind_speed_ms=2.0,
                )
            )
        return points
