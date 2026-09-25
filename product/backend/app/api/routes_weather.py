"""站点天气接口：按站点坐标返回逐小时辐射预报。

Provider 通过 infrastructure.weather.factory 选择（WEATHER_PROVIDER 环境变量）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_db
from app.api.errors import not_found
from app.infrastructure.db import Database
from app.infrastructure.weather.factory import get_weather_provider

router = APIRouter(prefix="/sites/{site_id}/weather", tags=["weather"])


@router.get("")
def get_site_weather(
    site_id: str,
    days: int = Query(default=3, ge=1, le=16),
    db: Database = Depends(get_db),
) -> dict:
    site = db.get_site(site_id)
    if site is None:
        raise not_found(f"site not found: {site_id}")

    provider = get_weather_provider()
    points = provider.irradiance_forecast(
        lat=site.location.lat,
        lon=site.location.lon,
        timezone_name=site.location.timezone,
        days=days,
    )
    return {
        "site_id": site_id,
        "provider": {"name": provider.name, "version": provider.version},
        "resolution_minutes": 60,
        "days": days,
        "points": [point.model_dump(mode="json") for point in points],
        "notes": "辐射量为数值气象预报输出，分辨率逐小时；用于光伏日前预测与冷启动场景",
    }
