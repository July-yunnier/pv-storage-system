"""预测编排：读取历史测量 → 调 forecast 纯函数 → 组装 TargetForecast。不依赖 FastAPI。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.api import errors
from app.domain.forecast.load import predict_load_profile
from app.domain.forecast.price import forecast_price_tou
from app.domain.forecast.pv import predict_pv_diurnal
from app.domain.forecast.result import ForecastPoint, ForecastResponse, TargetForecast
from app.domain.models import ForecastRequest, SiteRead
from app.infrastructure.db import Database, _zone

N_DAYS = 12
MAX_HORIZON_HOURS = 72


def _history(
    db: Database, site: SiteRead, data_type: str, before: date, n_days: int
) -> list[tuple[datetime, float]]:
    """before 之前最近 n_days 天的可用测量（缺失/invalid 不计入）。"""
    history: list[tuple[datetime, float]] = []
    for offset in range(n_days, 0, -1):
        day = before - timedelta(days=offset)
        rows = db.get_measurements_for_date(
            site.site_id, data_type, day, site.location.timezone
        )
        history.extend(
            (row.ts, row.value)
            for row in rows
            if row.value is not None and row.quality in ("measured", "estimated")
        )
    return history


def forecast_series(
    db: Database,
    site: SiteRead,
    data_type: str,
    target_date: date,
    n_days: int = N_DAYS,
) -> list[ForecastPoint]:
    """供调度复用的单日 pv/load 预测点。历史不足抛 ValueError。"""
    tz = _zone(site.location.timezone)
    history = _history(db, site, data_type, target_date, n_days)
    if data_type == "pv_power":
        forecast = predict_pv_diurnal(history, target_date, site.pv.capacity_kwp, tz, n_days)
    else:
        forecast = predict_load_profile(history, target_date, tz, n_days)
    return forecast.points


def _forecast_target(
    db: Database,
    site: SiteRead,
    target: str,
    start_date: date,
    days: int,
    include_uncertainty: bool,
) -> TargetForecast:
    tz = _zone(site.location.timezone)
    if target == "price":
        tariff_id = site.tariff_id or "T-DEFAULT"
        tariff = db.get_tariff(tariff_id)
        if tariff is None:
            raise errors.validation_error(f"tariff not found: {tariff_id}")
        return forecast_price_tou(tariff, start_date, days, tz)

    history = _history(db, site, target, start_date, N_DAYS)
    merged: TargetForecast | None = None
    for offset in range(days):
        day = start_date + timedelta(days=offset)
        try:
            if target == "pv_power":
                forecast = predict_pv_diurnal(history, day, site.pv.capacity_kwp, tz, N_DAYS)
            else:
                forecast = predict_load_profile(history, day, tz, N_DAYS)
        except ValueError as exc:
            raise errors.data_quality_error(str(exc)) from exc
        if merged is None:
            merged = forecast
        else:
            merged.points.extend(forecast.points)
            if merged.p10 is not None and forecast.p10 is not None:
                merged.p10.extend(forecast.p10)
                merged.p90 = (merged.p90 or []) + (forecast.p90 or [])
    assert merged is not None
    if not include_uncertainty:
        merged.p10 = None
        merged.p90 = None
    return merged


def run_forecast(db: Database, payload: ForecastRequest) -> ForecastResponse:
    site = db.get_site(payload.site_id)
    if site is None:
        raise errors.not_found(f"site not found: {payload.site_id}")
    if payload.resolution_minutes != 15:
        raise errors.validation_error("MVP only supports 15-minute resolution")
    if payload.horizon_hours > MAX_HORIZON_HOURS or payload.horizon_hours % 24 != 0:
        raise errors.validation_error(
            f"horizon_hours must be a positive multiple of 24 and at most {MAX_HORIZON_HOURS}"
        )

    issued_at = payload.issued_at or datetime.now(timezone.utc)
    tz = _zone(site.location.timezone)
    # 日前预测：覆盖站点时区次日 00:00 起的完整天
    start_date = issued_at.astimezone(tz).date() + timedelta(days=1)
    days = payload.horizon_hours // 24

    results = [
        _forecast_target(
            db, site, target, start_date, days, payload.options.include_uncertainty
        )
        for target in payload.targets
    ]
    return ForecastResponse(
        site_id=site.site_id,
        issued_at=issued_at,
        resolution_minutes=payload.resolution_minutes,
        results=results,
    )
