"""电价中心 API：未来电价预测、历史电价、现货序列导入。

与 `/forecast` 的分工：
- `/forecast` 是统一预测入口（pv/load/price），价格走政策表；
- 这里回答"未来电价曲线长什么样、凭什么这么认为、能给储能留多少套利空间"，
  是电价中心页面的专用聚合接口，输出更贴近决策（分位数 + 套利窗口）。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.api import errors
from app.api.dependencies import get_db
from app.application import algorithm_registry, observability_service, price_service
from app.domain.forecast import price_calendar as cal
from app.domain.models import SiteRead, SpotPriceRead, SpotPriceUpload
from app.infrastructure.db import Database, _zone
from app.domain.tariff import TouProvider

router = APIRouter(prefix="/sites/{site_id}/price", tags=["price"])


class PricePointView(BaseModel):
    ts: datetime
    p10: float | None = None
    p50: float
    p90: float | None = None
    tou_price: float | None = None
    tier: str | None = None


class PriceForecastSummary(BaseModel):
    mean: float
    max: float
    min: float
    peak_valley_spread: float
    base_tou_spread: float | None = None
    max_hours: list[str] = Field(default_factory=list)
    min_hours: list[str] = Field(default_factory=list)
    arbitrage: dict | None = None
    holidays: list[str] = Field(default_factory=list)


class PriceForecastView(BaseModel):
    site_id: str
    issued_at: datetime
    horizon_hours: int
    resolution_minutes: int
    model: str
    method_note: str
    data_note: str
    uncertainty: bool
    points: list[PricePointView]
    summary: PriceForecastSummary
    accuracy: dict | None = None
    interval_coverage: float | None = None
    warning: str | None = None
    interval_coverage_evaluated_on: str | None = None


class PriceHistoryView(BaseModel):
    site_id: str
    unit: str
    resolution: str
    points: list[dict]
    stats: dict
    bounds: dict | None = None


def _site(db: Database, site_id: str) -> SiteRead:
    site = db.get_site(site_id)
    if site is None:
        raise errors.not_found(f"site not found: {site_id}")
    return site


def _tier_map(site: SiteRead, db: Database) -> tuple[TouProvider | None, dict[int, str]]:
    tariff = db.get_tariff(site.tariff_id or "T-DEFAULT")
    if tariff is None or tariff.mode != "tou":
        return None, {}
    provider = TouProvider(tariff)
    tiers: dict[int, str] = {}
    for hour in range(24):
        probe = datetime(2024, 1, 1, hour, 0, tzinfo=_zone(site.location.timezone))
        try:
            tiers[hour] = provider.tier_at(probe).name
        except ValueError:
            continue
    return provider, tiers


@router.get("/forecast", response_model=PriceForecastView)
def get_price_forecast(
    site_id: str,
    days: int = Query(default=1, ge=1, le=7, description="预测天数（1-7）"),
    # 取值集合由算法注册表定义（GET /api/v1/algorithms），"auto" 为服务端降级链策略键
    model: str = Query(default="auto", description="auto 或注册表 price 条目 product_key"),
    uncertainty: bool = Query(default=True, description="是否返回 P10/P90 区间"),
    resolution_minutes: int = Query(default=15, description="输出粒度，仅支持 15/30/60"),
    db: Database = Depends(get_db),
) -> PriceForecastView:
    if resolution_minutes not in (15, 30, 60):
        raise errors.validation_error("resolution_minutes must be 15, 30 or 60")
    site = _site(db, site_id)
    tz = _zone(site.location.timezone)
    issued_at = datetime.now(tz)
    start_date = issued_at.date() + timedelta(days=1)
    if model != "auto":
        try:
            algorithm_registry.validate_product_choice("price", model)
        except ValueError as exc:
            raise errors.validation_error(str(exc)) from exc
    target, series, data_note = price_service.forecast_price(
        db,
        site,
        start_date,
        days,
        tz,
        model=model,
        include_uncertainty=uncertainty,
        resolution_minutes=resolution_minutes,
    )
    if not target.points:
        raise errors.data_quality_error("empty price forecast")

    price_service.archive_price_forecast(db, site, target, issued_at)
    observability_service.reconcile_site(db, site_id)
    coverage_status = observability_service.price_interval_status(db, site_id, target.model.name)

    provider, tiers = _tier_map(site, db)
    tou_by_hour: dict[int, float] = {}
    if provider is not None:
        probe_day = datetime.combine(start_date, datetime.min.time(), tzinfo=tz)
        for hour in range(24):
            try:
                tou_by_hour[hour] = provider.price_at(probe_day.replace(hour=hour)).price
            except ValueError:
                continue

    points = [
        PricePointView(
            ts=point.ts,
            p10=(target.p10[i] if target.p10 else None),
            p50=point.value,
            p90=(target.p90[i] if target.p90 else None),
            tou_price=tou_by_hour.get(point.ts.hour),
            tier=tiers.get(point.ts.hour),
        )
        for i, point in enumerate(target.points)
    ]

    values = [p.value for p in target.points]
    holiday_days = sorted(
        {
            point.ts.date().isoformat()
            for point in target.points
            if cal.is_holiday(point.ts.date())
        }
    )
    summary = PriceForecastSummary(
        mean=round(sum(values) / len(values), 4),
        max=round(max(values), 4),
        min=round(min(values), 4),
        peak_valley_spread=round(max(values) - min(values), 4),
        base_tou_spread=_tou_spread(points),
        max_hours=_top_hours(points, reverse=True),
        min_hours=_top_hours(points, reverse=False),
        arbitrage=_arbitrage(target, resolution_minutes),
        holidays=holiday_days,
    )

    return PriceForecastView(
        site_id=site_id,
        issued_at=issued_at,
        horizon_hours=days * 24,
        resolution_minutes=resolution_minutes,
        model=target.model.name,
        method_note=target.notes,
        data_note=data_note or "tariff",
        uncertainty=bool(uncertainty),
        points=points,
        summary=summary,
        accuracy=target.accuracy.model_dump(mode="json") if target.accuracy else None,
        interval_coverage=coverage_status["interval_coverage"],
        warning=coverage_status["warning"],
        interval_coverage_evaluated_on=coverage_status["evaluated_on"],
    )


def _tou_spread(points: list[PricePointView]) -> float | None:
    tou = [p.tou_price for p in points if p.tou_price is not None]
    if len(tou) < 2:
        return None
    return round(max(tou) - min(tou), 4)


def _top_hours(points: list[PricePointView], reverse: bool, k: int = 3) -> list[str]:
    ordered = sorted(points, key=lambda p: p.p50, reverse=reverse)[:k]
    seen: list[str] = []
    for point in ordered:
        text = f"{point.ts.hour:02d}:{point.ts.minute:02d}"
        if text not in seen:
            seen.append(text)
    return seen


def _arbitrage(target, resolution_minutes: int) -> dict | None:
    """复用服务层的充放窗口求解，避免前后两处口径漂移。"""
    return price_service.best_arbitrage(
        [(point.ts, point.value) for point in target.points],
        slot_hours=price_service.DEFAULT_SLOT_HOURS,
        resolution_minutes=resolution_minutes,
        round_trip_eff=price_service.DEFAULT_ROUND_TRIP_EFF,
    )


@router.get("/history", response_model=PriceHistoryView)
def get_price_history(
    site_id: str,
    days: int = Query(default=30, ge=1, le=365),
    resolution: Literal["hourly", "daily"] = Query(default="hourly"),
    db: Database = Depends(get_db),
) -> PriceHistoryView:
    site = _site(db, site_id)
    tz = _zone(site.location.timezone)
    end = datetime.now(tz)
    start = end - timedelta(days=days)
    rows = db.get_spot_prices(site_id, start, end)
    points = [
        {
            "ts": ts.astimezone(tz).isoformat(),
            "price": value,
            "source": source,
        }
        for ts, value, source in rows
        if value is not None
    ]
    values = [p["price"] for p in points]
    stats = {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4) if values else None,
        "max": round(max(values), 4) if values else None,
        "min": round(min(values), 4) if values else None,
        "negative_hours": sum(1 for v in values if v < 0),
        "spike_hours": sum(1 for v in values if v > 1.5),
    }
    if resolution == "daily":
        buckets: dict[str, list[float]] = {}
        for point in points:
            buckets.setdefault(point["ts"][:10], []).append(point["price"])
        points = [
            {"ts": day, "price": round(sum(vals) / len(vals), 4), "source": "daily_mean"}
            for day, vals in sorted(buckets.items())
        ]
    bounds = db.spot_price_bounds(site_id)
    return PriceHistoryView(
        site_id=site_id,
        unit="CNY/kWh",
        resolution=resolution,
        points=points,
        stats=stats,
        bounds=(
            {"first": bounds[0].astimezone(tz).isoformat(), "last": bounds[1].astimezone(tz).isoformat()}
            if bounds
            else None
        ),
    )


@router.post("/spot-prices")
def upload_spot_prices(
    site_id: str, payload: SpotPriceUpload, db: Database = Depends(get_db)
) -> dict:
    try:
        accepted = db.insert_spot_prices(site_id, payload)
    except KeyError as exc:
        raise errors.not_found(str(exc)) from exc
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    return {"site_id": site_id, "accepted": accepted, "source": payload.source}


@router.get("/spot-prices", response_model=list[SpotPriceRead])
def list_spot_prices(
    site_id: str,
    days: int = Query(default=7, ge=1, le=365),
    db: Database = Depends(get_db),
) -> list[SpotPriceRead]:
    site = _site(db, site_id)
    tz = _zone(site.location.timezone)
    end = datetime.now(tz)
    rows = db.get_spot_prices(site_id, end - timedelta(days=days), end)
    return [
        SpotPriceRead(
            ts=ts.astimezone(tz), price_cny_per_kwh=value, source=source, quality="measured"
        )
        for ts, value, source in rows
    ]


@router.delete("/spot-prices")
def clear_spot_prices(site_id: str, db: Database = Depends(get_db)) -> dict:
    _site(db, site_id)
    return {"site_id": site_id, "deleted": db.delete_spot_prices(site_id)}
