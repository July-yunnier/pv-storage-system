"""天气 Provider 工厂：WEATHER_PROVIDER 环境变量切换。

open_meteo（默认）/ mock（离线开发、测试）。
"""

from __future__ import annotations

import os

from app.infrastructure.weather.base import WeatherProvider
from app.infrastructure.weather.mock import MockWeatherProvider
from app.infrastructure.weather.open_meteo import OpenMeteoProvider


def get_weather_provider() -> WeatherProvider:
    if os.getenv("WEATHER_PROVIDER", "open_meteo") == "mock":
        return MockWeatherProvider()
    return OpenMeteoProvider()
