"""天气数据契约与 Provider 接口。

辐射量单位统一 W/m²（气象惯例）；时间为带时区的站点本地时间。
- ghi_wm2: 水平面总辐射（Global Horizontal Irradiance）
- dni_wm2: 法向直射辐射（Direct Normal Irradiance）
- dhi_wm2: 散射辐射（Diffuse Horizontal Irradiance）
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field


class WeatherPoint(BaseModel):
    ts: datetime
    ghi_wm2: float = Field(ge=0)
    dni_wm2: float = Field(ge=0)
    dhi_wm2: float = Field(ge=0)
    temperature_c: float | None = None
    wind_speed_ms: float | None = None


class WeatherProvider(Protocol):
    """按站点坐标返回逐小时辐射预报。"""

    name: str
    version: str

    def irradiance_forecast(
        self,
        *,
        lat: float,
        lon: float,
        timezone_name: str,
        days: int,
    ) -> list[WeatherPoint]:
        """返回从站点当地今日 00:00 起 days 天的逐小时点（24*days 个）。"""
        ...
