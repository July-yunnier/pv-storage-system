"""电价预测编排：现货历史 + 负荷 + 天气 + 政策表 → 分位数曲线 → 套利窗口。

三条数据流在同一个小时网格上对齐：
1. **价格**：站点现货电价历史（`spot_prices`）；没有历史就不预测，退回政策表。
2. **负荷**：历史取实测小时均值；未来取"同刻均值画像"（与负荷模块同源口径）。
   负荷是电价的第一驱动量——GEFCom2014-P 官方只给了系统/分区负荷预报当协变量，
   原因也在这里。
3. **天气**：温度走 Open-Meteo 真实预报。温度不进价格方程，而是通过负荷
   间接作用；把它显式放进特征，模型才能学到"高温→空调负荷→尖峰电价"。
   辐照同理（光伏大发 → 净负荷下凹 → 午间电价塌陷），这是新能源占比高的
   市场里最重要的一条价格形成机制。

粒度约定：模型在**小时**上训练与推理，对外按 15 分钟**阶跃**展开（同一小时
内价格相同）。不做插值——插值会造出现货市场里不存在的价格。
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.api import errors
from app.domain.forecast import price_calendar as cal
from app.domain.forecast.price import forecast_price_tou
from app.domain.forecast.price_features import PriceSeries
from app.domain.forecast.price_model import (
    LgbmQuantileForecaster,
    PriceForecastOutput,
    diurnal_quantile_forecast,
)
from app.domain.forecast.result import (
    AccuracyCaliber,
    ForecastAccuracy,
    ForecastModel,
    ForecastPoint,
    ForecastProvenance,
    TargetForecast,
)
from app.domain.models import SiteRead
from app.domain.tariff import TouProvider
from app.infrastructure.db import Database, _zone
from app.infrastructure.weather.factory import get_weather_provider

HISTORY_DAYS = 120
MIN_STAT_DAYS = 3
DEFAULT_SLOT_HOURS = 2.0
DEFAULT_ROUND_TRIP_EFF = 0.88


class NoSpotHistory(ValueError):
    """站点还没有现货电价历史，无法做数据驱动预测。"""


def model_dir() -> Path:
    override = os.environ.get("PRICE_MODEL_DIR")
    if override:
        return Path(override)
    # backend/app/application/price_service.py -> backend/data/price_models
    return Path(__file__).resolve().parents[2] / "data" / "price_models"


# --------------------------------------------------------------------------- 数据组装


def _load_hourly(
    db: Database, site: SiteRead, start: datetime, end: datetime
) -> dict[pd.Timestamp, float]:
    """把区间内的负荷实测按小时取均值；返回 {本地 naive 小时: kW}。"""
    tz = _zone(site.location.timezone)
    values: dict[pd.Timestamp, list[float]] = {}
    cursor = start.astimezone(tz).date()
    last = end.astimezone(tz).date()
    while cursor <= last:
        for row in db.get_measurements_for_date(site.site_id, "load_power", cursor, site.location.timezone):
            if row.value is None or row.quality not in ("measured", "estimated"):
                continue
            key = pd.Timestamp(row.ts.astimezone(tz).replace(tzinfo=None)).floor("h")
            values.setdefault(key, []).append(float(row.value))
        cursor += timedelta(days=1)
    return {key: float(np.mean(items)) for key, items in values.items()}


def _diurnal_load_profile(
    observed: dict[pd.Timestamp, float], target_ts: pd.DatetimeIndex, fallback: float | None
) -> np.ndarray:
    """未来负荷的"同刻均值画像"：按 (日类型, 小时) 取历史均值。"""
    if not observed:
        return np.full(len(target_ts), np.nan if fallback is None else fallback)
    frame = pd.DataFrame(
        {
            "ts": list(observed.keys()),
            "value": list(observed.values()),
        }
    )
    frame["hour"] = frame["ts"].dt.hour
    frame["dtype"] = [
        "holiday" if cal.is_holiday(t.date()) else ("workday" if cal.is_workday(t.date()) else "weekend")
        for t in frame["ts"]
    ]
    by_dtype = frame.groupby(["dtype", "hour"])["value"].mean()
    by_hour = frame.groupby("hour")["value"].mean()
    overall = float(frame["value"].mean())

    out = np.full(len(target_ts), np.nan)
    for pos, ts in enumerate(target_ts):
        dtype = (
            "holiday"
            if cal.is_holiday(ts.date())
            else ("workday" if cal.is_workday(ts.date()) else "weekend")
        )
        value = by_dtype.get((dtype, ts.hour))
        if value is None or not np.isfinite(value):
            value = by_hour.get(ts.hour, overall)
        out[pos] = float(value)
    return out


def _weather_arrays(
    site: SiteRead, grid: pd.DatetimeIndex, history_days: int
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """取逐小时温度与辐照，按 grid 对齐；不可用时返回 None。"""
    try:
        provider = get_weather_provider()
        points = provider.irradiance_window(
            lat=site.location.lat,
            lon=site.location.lon,
            timezone_name=site.location.timezone,
            past_days=min(history_days, 30),
            forecast_days=7,
        )
    except Exception:  # noqa: BLE001 - 气象不可用不得阻断电价预测
        return None, None, None

    tz = _zone(site.location.timezone)
    temp_map: dict[pd.Timestamp, float] = {}
    ghi_map: dict[pd.Timestamp, float] = {}
    for point in points:
        key = pd.Timestamp(point.ts.astimezone(tz).replace(tzinfo=None)).floor("h")
        if point.temperature_c is not None:
            temp_map[key] = float(point.temperature_c)
        ghi_map[key] = float(point.ghi_wm2)
    if not ghi_map:
        return None, None, provider.name
    temp = np.asarray([temp_map.get(ts, np.nan) for ts in grid], dtype="float64")
    ghi = np.asarray([ghi_map.get(ts, np.nan) for ts in grid], dtype="float64")
    return temp, ghi, provider.name


def build_price_series(
    db: Database,
    site: SiteRead,
    tariff,
    target_end: datetime,
    history_days: int = HISTORY_DAYS,
) -> tuple[PriceSeries, pd.Timestamp, str | None]:
    """组装小时粒度价格序列，返回 (序列, 发报小时, 气象来源)。"""
    tz = _zone(site.location.timezone)
    bounds = db.spot_price_bounds(site.site_id)
    if bounds is None:
        raise NoSpotHistory("站点尚无现货电价历史")

    first_utc, _ = bounds
    first_local = pd.Timestamp(first_utc.astimezone(tz).replace(tzinfo=None))
    end_local = pd.Timestamp(target_end.astimezone(tz).replace(tzinfo=None))
    start_local = max(first_local.floor("h"), (end_local - timedelta(days=history_days)).floor("h"))
    grid = pd.date_range(start_local, end_local.floor("h"), freq="h")
    if len(grid) < 48:
        raise NoSpotHistory(f"现货电价历史不足：{len(grid)} 小时")

    rows = db.get_spot_prices(
        site.site_id,
        start_local.replace(tzinfo=tz).astimezone(timezone.utc),
        (end_local + timedelta(hours=1)).replace(tzinfo=tz).astimezone(timezone.utc),
    )
    price_map = {
        pd.Timestamp(ts.astimezone(tz).replace(tzinfo=None)).floor("h"): value for ts, value, _ in rows
    }
    sources = {source for _, _, source in rows}
    price = np.asarray([price_map.get(ts, np.nan) for ts in grid], dtype="float64")
    if np.isfinite(price).sum() < 24:
        raise NoSpotHistory("现货电价有效点不足 24 个")

    observed_load = _load_hourly(db, site, start_local.replace(tzinfo=tz), end_local.replace(tzinfo=tz))
    load = np.asarray([observed_load.get(ts, np.nan) for ts in grid], dtype="float64")
    missing_mask = ~np.isfinite(load)
    if missing_mask.any():
        profile = _diurnal_load_profile(observed_load, grid[missing_mask], None)
        load[missing_mask] = profile

    provider = TouProvider(tariff)
    tou = np.full(len(grid), np.nan)
    for pos, ts in enumerate(grid):
        try:
            tou[pos] = provider.price_at(ts.to_pydatetime().replace(tzinfo=tz)).price
        except ValueError:
            tou[pos] = np.nan

    temperature, ghi, weather_name = _weather_arrays(site, grid, history_days)
    series = PriceSeries(
        ts=grid,
        price=price,
        load=load if np.isfinite(load).any() else None,
        temperature=temperature,
        ghi=ghi,
        tou_price=tou,
    )
    # 数据来源必须把"电价从哪来"说清楚——前端据此决定是否打"演示数据"提示
    parts = ["spot:" + (",".join(sorted(sources)) if sources else "unknown")]
    if weather_name:
        parts.append(f"weather:{weather_name}")
    if series.load is None:
        parts.append("load:none")
    return series, grid[-1], ";".join(parts)


# --------------------------------------------------------------------------- 套利窗口


def best_arbitrage(
    points: list[tuple[datetime, float]],
    slot_hours: float = DEFAULT_SLOT_HOURS,
    resolution_minutes: int = 15,
    round_trip_eff: float = DEFAULT_ROUND_TRIP_EFF,
) -> dict | None:
    """在单调充放一次（先充后放）的约束下求最优窗口。

    不能"取全局最低价当充电、全局最高价当放电"——那样会挑出一个时间上不成立的
    组合（最低价出现在最高价之后）。这里穷举 (charge_start, discharge_start) 对，
    要求放电窗口整体晚于充电窗口，再比净价差。窗口数 ≤ 96×96，成本可忽略。
    """
    steps_per_hour = max(1, 60 // resolution_minutes)
    slot = max(1, int(round(slot_hours * steps_per_hour)))
    values = np.asarray([value for _, value in points], dtype="float64")
    n = len(values)
    if n < 2 * slot:
        return None

    csum = np.concatenate([[0.0], np.cumsum(values)])
    best: tuple[float, int, int] | None = None
    for i in range(0, n - 2 * slot + 1):
        charge_mean = (csum[i + slot] - csum[i]) / slot
        for j in range(i + slot, n - slot + 1):
            discharge_mean = (csum[j + slot] - csum[j]) / slot
            net = discharge_mean * round_trip_eff - charge_mean / round_trip_eff
            if best is None or net > best[0]:
                best = (net, i, j)
    if best is None:
        return None

    _, i, j = best
    charge_mean = float((csum[i + slot] - csum[i]) / slot)
    discharge_mean = float((csum[j + slot] - csum[j]) / slot)
    return {
        "charge_start": points[i][0].isoformat(),
        "discharge_start": points[j][0].isoformat(),
        "slot_hours": round(slot / steps_per_hour, 2),
        "charge_price": round(charge_mean, 4),
        "discharge_price": round(discharge_mean, 4),
        "gross_spread": round(discharge_mean - charge_mean, 4),
        "net_spread": round(discharge_mean * round_trip_eff - charge_mean / round_trip_eff, 4),
        "round_trip_eff": round_trip_eff,
        "note": "单次两充两放的最优时间对（先充后放），未计需量电费、电池退化与功率约束",
    }


def _expand_to_resolution(
    ts: pd.DatetimeIndex, values: np.ndarray, resolution_minutes: int
) -> list[float]:
    """小时内价格恒定：按阶跃展开，不插值。"""
    repeats = max(1, 60 // resolution_minutes)
    return np.repeat(values, repeats).astype("float64").tolist()


# --------------------------------------------------------------------------- 主入口


def _recursive_day_ahead(
    forecaster: LgbmQuantileForecaster, series: PriceSeries, issue_idx: int, days: int
) -> PriceForecastOutput:
    """多日前预测：**按天递归**，每天仍只做 1..24 步的直接多步预测。

    工件只在 24 步上训练过，硬喂 horizon=72 的特征组合属于外推，会失真。
    因此第 d+1 天把第 d 天的 p50 写入历史再发报——天级递归的误差累积远小于
    小时级递归，且与生产上"每天重发"的节奏一致。此性质在文档中披露。
    """
    price = np.asarray(series.price, dtype="float64").copy()
    parts: list[PriceForecastOutput] = []
    for day in range(days):
        origin = issue_idx + day * 24
        if origin + 24 >= len(series.ts):
            break
        working = PriceSeries(
            ts=series.ts,
            price=price,
            load=series.load,
            load_total=series.load_total,
            temperature=series.temperature,
            ghi=series.ghi,
            tou_price=series.tou_price,
        )
        part = forecaster.predict(working, origin, 24)
        parts.append(part)
        price[origin + 1 : origin + 25] = part.p50
    if not parts:
        raise ValueError("no day could be forecast")
    return PriceForecastOutput(
        ts=pd.DatetimeIndex(np.concatenate([p.ts.values for p in parts])),
        p10=np.concatenate([p.p10 for p in parts]),
        p50=np.concatenate([p.p50 for p in parts]),
        p90=np.concatenate([p.p90 for p in parts]),
        method=parts[0].method,
        version=parts[0].version,
        notes=parts[0].notes + ("；多日按天递归（每日重发），非一次前向 72 步" if days > 1 else ""),
    )


def forecast_price(
    db: Database,
    site: SiteRead,
    start_date: date,
    days: int,
    tz,
    model: str = "auto",
    include_uncertainty: bool = True,
    resolution_minutes: int = 15,
) -> tuple[TargetForecast, PriceSeries | None, str | None]:
    """返回 (预测 DTO, 小时序列, 数据来源描述)。"""
    tariff_id = site.tariff_id or "T-DEFAULT"
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise errors.validation_error(f"tariff not found: {tariff_id}")

    if model == "tou":
        return forecast_price_tou(tariff, start_date, days, tz), None, "tariff"

    target_end = datetime.combine(start_date + timedelta(days=days - 1), time(23), tzinfo=tz)
    try:
        series, issue_hour, provenance_note = build_price_series(db, site, tariff, target_end)
    except NoSpotHistory as exc:
        if model == "auto":
            return forecast_price_tou(tariff, start_date, days, tz), None, str(exc)
        raise errors.data_quality_error(str(exc)) from exc

    window_start = datetime.combine(start_date, time(0), tzinfo=tz).replace(tzinfo=None)
    issue_idx = int(series.ts.searchsorted(pd.Timestamp(window_start), side="left")) - 1
    if issue_idx < 23:
        if model == "auto":
            return forecast_price_tou(tariff, start_date, days, tz), None, "history too short"
        raise errors.data_quality_error("现货电价历史不足以支撑发报时刻")

    horizon = days * 24
    forecast: PriceForecastOutput | None = None
    notes: list[str] = []

    artifact = model_dir() / site.site_id
    if model in ("auto", "lgbm") and (artifact / "manifest.json").exists():
        try:
            forecaster = LgbmQuantileForecaster.load(artifact)
            forecast = _recursive_day_ahead(forecaster, series, issue_idx, days)
        except (RuntimeError, ValueError, KeyError, FileNotFoundError, ImportError) as exc:
            # ImportError：机器上没装 lightgbm 但工件存在。不能 500，退回统计画像。
            if model == "lgbm":
                raise errors.data_quality_error(f"price model failed: {exc}") from exc
            notes.append(f"LightGBM 工件不可用，回退统计画像（{exc}）")

    if forecast is None:
        forecast = diurnal_quantile_forecast(series, issue_idx, horizon)

    p10 = _expand_to_resolution(forecast.ts, forecast.p10, resolution_minutes)
    p50 = _expand_to_resolution(forecast.ts, forecast.p50, resolution_minutes)
    p90 = _expand_to_resolution(forecast.ts, forecast.p90, resolution_minutes)
    steps = pd.date_range(
        datetime.combine(start_date, time(0), tzinfo=tz),
        periods=len(p50),
        freq=f"{resolution_minutes}min",
    )

    components = {
        key: _expand_to_resolution(forecast.ts, values, resolution_minutes)
        for key, values in forecast.components.items()
    }

    slot_hours = max(1, int(DEFAULT_SLOT_HOURS))
    arbitrage = best_arbitrage(
        [(ts.to_pydatetime().replace(tzinfo=tz), float(v)) for ts, v in zip(steps, p50)],
        slot_hours=slot_hours,
        resolution_minutes=resolution_minutes,
    )

    target = TargetForecast(
        target="price",
        unit="CNY/kWh",
        model=ForecastModel(
            name=forecast.method,
            version=forecast.version,
        ),
        points=[ForecastPoint(ts=ts.to_pydatetime().replace(tzinfo=tz), value=float(v)) for ts, v in zip(steps, p50)],
        p10=p10 if include_uncertainty else None,
        p90=p90 if include_uncertainty else None,
        accuracy=_archive_accuracy(db, site),
        provenance=ForecastProvenance(
            provider=provenance_note,
            window=f"小时粒度模型，输出按 {resolution_minutes} 分钟阶跃展开；现货历史 {len(series.ts)} 小时",
        ),
        notes="；".join([forecast.notes, *notes]),
        components=components or None,
    )

    if arbitrage:
        target.notes += (
            f"；最优单次充放：{arbitrage['charge_start'][11:16]} 起充"
            f"（{arbitrage['charge_price']:.3f}）、{arbitrage['discharge_start'][11:16]} 起放"
            f"（{arbitrage['discharge_price']:.3f}），净价差 {arbitrage['net_spread']:.3f} 元/kWh"
            f"（含 {arbitrage['round_trip_eff']:.0%} 往返效率，未计需量）"
        )
    return target, series, provenance_note


def _archive_accuracy(db: Database, site: SiteRead) -> ForecastAccuracy | None:
    """从观测体系取该站点电价方法的最近一次对账结果（没有就返回 None）。"""
    rows = db.list_forecast_accuracy(site.site_id, target="price")
    if not rows:
        return None
    latest = sorted(rows, key=lambda r: r["target_date"])[-1]
    metrics = latest["metrics"]
    if "mae" not in metrics:
        return None
    return ForecastAccuracy(
        metric="MAE",
        value=float(metrics["mae"]),
        caliber=AccuracyCaliber(
            normalization="CNY/kWh 绝对值误差，不做容量归一",
            cap_kw=0.0,
            includes_nighttime=True,
            spatial_scale="site",
            baseline=metrics.get("baseline", "naive_daily"),
        ),
        evaluated_on=latest["target_date"],
    )
