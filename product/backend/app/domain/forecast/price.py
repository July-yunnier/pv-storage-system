"""价格预测：TOU 模式由政策给定，直接展开 TouProvider，无需预测。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo

from app.domain.forecast.result import ForecastModel, ForecastPoint, TargetForecast
from app.domain.models import TariffRead
from app.domain.tariff import TouProvider


def forecast_price_tou(
    tariff: TariffRead,
    start_date: date,
    days: int,
    tz: tzinfo,
    resolution_minutes: int = 15,
) -> TargetForecast:
    provider = TouProvider(tariff)
    points: list[ForecastPoint] = []
    for offset in range(days):
        day_start = datetime.combine(start_date + timedelta(days=offset), time.min, tzinfo=tz)
        points.extend(
            ForecastPoint(ts=p.ts, value=p.price)
            for p in provider.points(day_start, 24 * 60 // resolution_minutes, resolution_minutes)
        )
    return TargetForecast(
        target="price",
        unit="CNY/kWh",
        model=ForecastModel(name="tou_policy", version="v1"),
        points=points,
        p10=[point.value for point in points],
        p90=[point.value for point in points],
        accuracy=None,
        notes="政策给定，无需预测",
    )
