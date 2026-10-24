"""预测编排：读取历史测量/气象预报 → 调 forecast 纯函数 → 组装 TargetForecast。

不依赖 FastAPI。光伏模型选择：
- auto（默认）：历史 ≥3 天 → blend 融合（气象失败回退统计法）；否则 physics 物理链
- statistical：同刻均值基线（需 ≥3 天历史）
- physics：纯 pvlib 物理链（无需历史，冷启动可用）
- intraday=True：日内滚动修正（光伏/负荷，覆盖今日剩余时段）

负荷模型选择：
- statistical（默认）：同刻均值基线
- advanced：MES 排产+多任务注意力模型；advanced_load_input 缺省时服务端自动构造
  特征（30 天负荷历史 + 气象温湿度 + 班次日历），显式传入则用于实验/回放
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.api import errors
from app.application import algorithm_registry, price_service
from app.application.load_features import build_advanced_load_inputs, predict_advanced
from app.domain.forecast.blend import blend_pv_forecast
from app.domain.forecast.load import predict_load_intraday, predict_load_profile
from app.domain.forecast.load_advanced import AdvancedLoadInputs
from app.domain.forecast.price import forecast_price_tou
from app.domain.forecast.pv import (
    predict_pv_diurnal,
    predict_pv_intraday,
    predict_pv_physics,
)
from app.domain.forecast.result import ForecastPoint, ForecastResponse, TargetForecast
from app.domain.forecast.stats import MIN_HISTORY_DAYS, daily_series
from app.domain.models import ForecastRequest, SiteRead
from app.infrastructure.db import Database, _zone
from app.infrastructure.weather.factory import get_weather_provider

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


def _history_days_available(history: list[tuple[datetime, float]], tz) -> int:
    return len(daily_series(history, tz))


def _weather_window(
    site: SiteRead, start_date: date, days: int, past_days: int
) -> tuple[list, str, datetime]:
    """取覆盖 [start-past_days, start+days) 的逐小时辐射窗口。

    返回 (气象点, provider 名, 拉取时间 UTC)——后两者用于预测溯源。
    """
    tz = _zone(site.location.timezone)
    today = datetime.now(tz).date()
    forecast_days = max((start_date + timedelta(days=days) - today).days, 1)
    provider = get_weather_provider()
    fetched_at = datetime.now(timezone.utc)
    points = provider.irradiance_window(
        lat=site.location.lat,
        lon=site.location.lon,
        timezone_name=site.location.timezone,
        past_days=past_days,
        forecast_days=forecast_days,
    )
    return points, provider.name, fetched_at


def _with_provenance(
    forecast: TargetForecast, provider_name: str, fetched_at: datetime
) -> TargetForecast:
    """把气象来源与拉取时间回填到 pv 预测的 provenance（blend/physics 均有）。"""
    base = forecast.provenance
    provenance = base.model_copy(
        update={"provider": provider_name, "fetched_at": fetched_at}
    ) if base else None
    return forecast.model_copy(update={"provenance": provenance})


def _pv_day_forecast(
    db: Database,
    site: SiteRead,
    day: date,
    history: list[tuple[datetime, float]],
    pv_model: str,
    explain: bool = False,
) -> TargetForecast:
    """单日光伏预测，按模型选择路径。auto 语义见模块 docstring。"""
    tz = _zone(site.location.timezone)
    pv = site.pv
    lat, lon = site.location.lat, site.location.lon
    has_history = _history_days_available(history, tz) >= MIN_HISTORY_DAYS

    if pv_model == "statistical" or (pv_model == "auto" and has_history):
        try:
            if pv_model == "auto":
                # 有历史：融合优先，气象不可用时回退统计法
                try:
                    weather, provider_name, fetched_at = _weather_window(
                        site, day, 1, past_days=N_DAYS
                    )
                    past = [p for p in weather if p.ts.date() < day]
                    future = [p for p in weather if p.ts.date() >= day]
                    return _with_provenance(
                        blend_pv_forecast(
                            past, future, history, pv, lat, lon, day, tz, N_DAYS,
                            explain=explain,
                        ),
                        provider_name,
                        fetched_at,
                    )
                except (errors.AppError, ValueError):
                    pass  # 气象不可用或融合数据不足 → 回退统计法
            return predict_pv_diurnal(history, day, pv.capacity_kwp, tz, N_DAYS)
        except ValueError as exc:
            raise errors.data_quality_error(str(exc)) from exc

    # physics / auto 无历史：纯物理链
    weather, provider_name, fetched_at = _weather_window(site, day, 1, past_days=0)
    provider = get_weather_provider()
    return _with_provenance(
        predict_pv_physics(weather, day, pv, lat, lon, tz, provider.name, explain=explain),
        provider_name,
        fetched_at,
    )


def forecast_series(
    db: Database,
    site: SiteRead,
    data_type: str,
    target_date: date,
    n_days: int = N_DAYS,
) -> list[ForecastPoint]:
    """供调度复用的单日 pv/load 预测点。全部路径失败抛 ValueError。"""
    history = _history(db, site, data_type, target_date, n_days)
    if data_type == "pv_power":
        try:
            return _pv_day_forecast(db, site, target_date, history, "auto").points
        except errors.AppError as exc:
            raise ValueError(f"pv forecast unavailable: {exc.message}") from exc
    tz = _zone(site.location.timezone)
    return predict_load_profile(history, target_date, tz, n_days).points


def _forecast_target(
    db: Database,
    site: SiteRead,
    target: str,
    start_date: date,
    days: int,
    include_uncertainty: bool,
    pv_model: str,
    load_model: str,
    advanced_load_input,
    intraday: bool,
    issued_at: datetime,
    explain: bool = False,
) -> TargetForecast:
    tz = _zone(site.location.timezone)
    if target == "price":
        tariff_id = site.tariff_id or "T-DEFAULT"
        tariff = db.get_tariff(tariff_id)
        if tariff is None:
            raise errors.validation_error(f"tariff not found: {tariff_id}")
        return forecast_price_tou(tariff, start_date, days, tz)

    history = _history(db, site, target, start_date, N_DAYS)
    if intraday and target == "pv_power":
        # 日内滚动修正：今日已发生实测 + 今日剩余时段；历史包含今日已发生部分
        history_today = _history(db, site, target, start_date + timedelta(days=1), N_DAYS)
        try:
            return predict_pv_intraday(
                history_today, issued_at.astimezone(tz), site.pv.capacity_kwp, tz, N_DAYS
            )
        except ValueError as exc:
            raise errors.data_quality_error(str(exc)) from exc

    if intraday and target == "load_power":
        history_today = _history(db, site, target, start_date + timedelta(days=1), N_DAYS)
        try:
            return predict_load_intraday(
                history_today, issued_at.astimezone(tz), tz, N_DAYS
            )
        except ValueError as exc:
            raise errors.data_quality_error(str(exc)) from exc

    merged: TargetForecast | None = None
    for offset in range(days):
        day = start_date + timedelta(days=offset)
        try:
            if target == "pv_power":
                forecast = _pv_day_forecast(
                    db, site, day, history, pv_model, explain=explain
                )
            elif load_model == "advanced":
                if days != 1:
                    raise ValueError("advanced load model currently supports a 24-hour horizon")
                if advanced_load_input is None:
                    # 未显式给特征：服务端自动构造（历史负荷+气象温湿度+班次日历）
                    advanced_load_input = build_advanced_load_inputs(db, site, day)
                forecast = predict_advanced(
                    AdvancedLoadInputs(
                        history_features=advanced_load_input.history_features,
                        history_mes_labels=advanced_load_input.history_mes_labels,
                        target_features=advanced_load_input.target_features,
                        target_mes_labels=advanced_load_input.target_mes_labels,
                    ),
                    day,
                    tz,
                )
            else:
                forecast = predict_load_profile(history, day, tz, N_DAYS)
        except (ValueError, RuntimeError) as exc:
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
    if payload.options.pv_model == "ampdnet":
        from app.application.ampdnet_service import run_ampdnet_forecast
        return run_ampdnet_forecast(db, site, payload)
    if payload.horizon_hours > MAX_HORIZON_HOURS or payload.horizon_hours % 24 != 0:
        raise errors.validation_error(
            f"horizon_hours must be a positive multiple of 24 and at most {MAX_HORIZON_HOURS}"
        )

    issued_at = payload.issued_at or datetime.now(timezone.utc)
    tz = _zone(site.location.timezone)
    # 算法选择键统一由注册表校验（"auto" 是光伏服务策略键：有历史走融合，否则物理链）
    try:
        if payload.options.pv_model != "auto":
            algorithm_registry.validate_product_choice("pv_power", payload.options.pv_model)
        algorithm_registry.validate_product_choice("load_power", payload.options.load_model)
    except ValueError as exc:
        raise errors.validation_error(str(exc)) from exc
    # 日前预测：覆盖站点时区次日 00:00 起的完整天；intraday 模式覆盖今日剩余时段
    if payload.options.intraday:
        start_date = issued_at.astimezone(tz).date()
        days = 1
    else:
        start_date = issued_at.astimezone(tz).date() + timedelta(days=1)
        days = int(payload.horizon_hours // 24)

    results = [
        _forecast_target(
            db,
            site,
            target,
            start_date,
            days,
            payload.options.include_uncertainty,
            payload.options.pv_model,
            payload.options.load_model,
            payload.options.advanced_load_input,
            payload.options.intraday,
            issued_at,
            payload.options.explain,
        )
        for target in payload.targets
    ]
    _archive_forecasts(db, site, results, issued_at, tz)
    return ForecastResponse(
        site_id=site.site_id,
        issued_at=issued_at,
        resolution_minutes=payload.resolution_minutes,
        results=results,
    )


def _archive_forecasts(
    db: Database,
    site: SiteRead,
    results: list[TargetForecast],
    issued_at: datetime,
    tz,
) -> None:
    """观测体系：每次发出的预测原样封存，供实测到位后对账。封存失败不阻断主流程。"""
    for result in results:
        if not result.points:
            continue
        try:
            if result.target == "price":
                price_service.archive_price_forecast(db, site, result, issued_at)
                continue
            db.insert_forecast_archive(
                site_id=site.site_id,
                target=result.target,
                method=result.model.name,
                target_date=result.points[0].ts.astimezone(tz).date(),
                issued_at=issued_at,
                points=[{"ts": p.ts.isoformat(), "value": p.value} for p in result.points],
                provenance=result.provenance.model_dump(mode="json") if result.provenance else None,
            )
        except Exception:  # noqa: BLE001 - 观测写库不应影响预测主链路
            continue
