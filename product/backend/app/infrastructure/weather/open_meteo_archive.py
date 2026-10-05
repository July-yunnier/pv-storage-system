"""Open-Meteo 历史再分析（archive）实现：为历史回放提供气象输入。

⚠️ 口径声明：archive 接口返回的是再分析/观测融合值，不是当年真实的预报。
回放实验中它作为"天气预报的代理"，报告里必须写明这一近似。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx

from app.api.errors import provider_error
from app.infrastructure.db import _zone
from app.infrastructure.weather.base import WeatherPoint

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m"


class OpenMeteoArchiveProvider:
    name = "open_meteo_archive"
    version = "v1"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout = timeout_seconds

    def irradiance_range(
        self,
        *,
        lat: float,
        lon: float,
        timezone_name: str,
        start: date,
        end: date,
    ) -> list[WeatherPoint]:
        """[start, end) 的逐小时辐射（再分析）。"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": HOURLY_VARS,
            "timezone": timezone_name,
            "start_date": start.isoformat(),
            "end_date": (end - timedelta(days=1)).isoformat(),
        }
        try:
            response = httpx.get(ARCHIVE_URL, params=params, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise provider_error(f"open-meteo archive unavailable: {exc}") from exc
        if response.status_code != 200:
            raise provider_error(
                f"open-meteo archive rejected: HTTP {response.status_code} {response.text[:200]}"
            )
        data = response.json()
        try:
            hourly = data["hourly"]
            times = hourly["time"]
        except (KeyError, TypeError) as exc:
            raise provider_error(f"open-meteo archive unexpected payload: {exc}") from exc
        tz = _zone(timezone_name)
        return [
            WeatherPoint(
                ts=datetime.fromisoformat(raw).replace(tzinfo=tz),
                ghi_wm2=hourly["shortwave_radiation"][i] or 0.0,
                dni_wm2=hourly["direct_normal_irradiance"][i] or 0.0,
                dhi_wm2=hourly["diffuse_radiation"][i] or 0.0,
                temperature_c=(hourly.get("temperature_2m") or [None] * len(times))[i],
                wind_speed_ms=(hourly.get("wind_speed_10m") or [None] * len(times))[i],
            )
            for i, raw in enumerate(times)
        ]
