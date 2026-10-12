"""高级负荷模型特征构造：measurements 历史负荷 + 气象窗口温湿度 + 班次日历 → AdvancedLoadInputs。

特征列序与训练工件 manifest 严格一致（工件内含标准化参数，这里喂原始物理量）：
- history（30 天 × 96 槽）：[total_load_kw, temperature_c, humidity_pct, slot_sin, slot_cos, is_workday]
- target（目标日 96 槽）：[temperature_c, humidity_pct, slot_sin, slot_cos, is_workday]

缺口策略：负荷缺失槽用可用天同刻均值填补；温湿度缺测用窗口内均值，
全缺时退回训练脚本的缺省值（20°C / 50%）。
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

from app.domain.forecast.load_advanced import AdvancedLoadInputs, predict_with_artifact
from app.domain.forecast.result import TargetForecast
from app.domain.models import SiteRead
from app.domain.shift_calendar import ShiftCalendar
from app.infrastructure.db import Database, _zone
from app.infrastructure.model_store import default_load_model_path, load_bundle
from app.infrastructure.weather.base import WeatherPoint
from app.infrastructure.weather.factory import get_weather_provider

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15
DEFAULT_HISTORY_DAYS = 30  # 与训练脚本 --history-days 默认值一致
DEFAULT_TEMPERATURE_C = 20.0   # 与训练脚本缺省一致
DEFAULT_HUMIDITY_PCT = 50.0


def artifact_history_days(model_path: str | None = None) -> int:
    """从模型工件旁的 manifest 读 history_days；读不到退回默认 30。"""
    raw = model_path or os.getenv("LOAD_MODEL_PATH", "")
    if raw:
        manifest = Path(raw).with_suffix(".manifest.json")
        try:
            days = int(json.loads(manifest.read_text(encoding="utf-8"))["history_days"])
            if days > 0:
                return days
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return DEFAULT_HISTORY_DAYS


def _cyclic(slot: int) -> tuple[float, float]:
    angle = 2 * math.pi * slot / SLOTS_PER_DAY
    return math.sin(angle), math.cos(angle)


def _hourly_series(
    points: list[WeatherPoint], attr: str, default: float
) -> dict[datetime, float]:
    """逐小时温湿度序列：缺测先窗口均值，全缺用缺省值。"""
    pairs = [(p.ts.replace(minute=0, second=0, microsecond=0), getattr(p, attr)) for p in points]
    valid = [v for _, v in pairs if v is not None]
    fill = sum(valid) / len(valid) if valid else default
    return {hour: (value if value is not None else fill) for hour, value in pairs}


def _interp(hourly: dict[datetime, float], ts: datetime, default: float) -> float:
    """小时锚点线性插值到 15 分钟槽；锚点缺失退回窗口均值/缺省值。"""
    if not hourly:
        return default
    anchor = ts.replace(minute=0, second=0, microsecond=0)
    current = hourly.get(anchor)
    nxt = hourly.get(anchor + timedelta(hours=1))
    if current is None:
        if nxt is not None:
            return nxt
        return sum(hourly.values()) / len(hourly)
    if nxt is None:
        return current
    frac = ts.minute / 60.0
    return current * (1.0 - frac) + nxt * frac


def _load_slot_map(
    db: Database, site: SiteRead, day: date, tz
) -> dict[int, float]:
    """单日 minute→kW（仅 measured/estimated）。"""
    rows = db.get_measurements_for_date(site.site_id, "load_power", day, site.location.timezone)
    day_start = datetime.combine(day, time.min, tzinfo=tz)
    values: dict[int, float] = {}
    for row in rows:
        if row.value is None or row.quality not in ("measured", "estimated"):
            continue
        minute = int((row.ts.astimezone(tz) - day_start).total_seconds() // 60)
        if 0 <= minute < 24 * 60:
            values[minute] = row.value
    return values


def build_advanced_load_inputs(
    db: Database,
    site: SiteRead,
    target_date: date,
    *,
    history_days: int | None = None,
    calendar: ShiftCalendar | None = None,
) -> AdvancedLoadInputs:
    """为 target_date 构造高级负荷模型输入。历史不足/气象不可用抛 ValueError 或 AppError。"""
    tz = _zone(site.location.timezone)
    days_count = history_days or artifact_history_days()
    cal = calendar or site.load.mes_calendar or ShiftCalendar()
    history_start = target_date - timedelta(days=days_count)

    # 负荷历史：先收集每天槽值，再用可用天同刻均值补缺
    per_day: list[tuple[date, dict[int, float]]] = [
        (history_start + timedelta(days=offset),
         _load_slot_map(db, site, history_start + timedelta(days=offset), tz))
        for offset in range(days_count)
    ]
    slot_means: dict[int, float] = {}
    for slot in range(SLOTS_PER_DAY):
        minute = slot * SLOT_MINUTES
        present = [day_values[minute] for _, day_values in per_day if minute in day_values]
        if present:
            slot_means[slot] = sum(present) / len(present)
    if not slot_means:
        raise ValueError(
            f"高级负荷模型需要最近 {days_count} 天负荷历史，当前站点无可用测量"
        )

    # 气象窗口：历史段（再分析/模型分析值）+ 目标日（预报）
    today = datetime.now(tz).date()
    provider = get_weather_provider()
    points = provider.irradiance_window(
        lat=site.location.lat,
        lon=site.location.lon,
        timezone_name=site.location.timezone,
        past_days=(today - history_start).days,
        forecast_days=max((target_date - today).days + 1, 1),
    )
    temperature = _hourly_series(points, "temperature_c", DEFAULT_TEMPERATURE_C)
    humidity = _hourly_series(points, "humidity_pct", DEFAULT_HUMIDITY_PCT)

    history_features: list[list[float]] = []
    history_mes: list[int] = []
    for day, day_values in per_day:
        is_workday = 1.0 if day.weekday() in cal.workdays else 0.0
        for slot in range(SLOTS_PER_DAY):
            minute = slot * SLOT_MINUTES
            load_kw = day_values.get(minute, slot_means[slot])
            ts = datetime.combine(day, time.min, tzinfo=tz) + timedelta(minutes=minute)
            sin_v, cos_v = _cyclic(slot)
            history_features.append([
                round(load_kw, 6),
                round(_interp(temperature, ts, DEFAULT_TEMPERATURE_C), 6),
                round(_interp(humidity, ts, DEFAULT_HUMIDITY_PCT), 6),
                round(sin_v, 6),
                round(cos_v, 6),
                is_workday,
            ])
        history_mes.extend(cal.labels_for_day(day))

    is_workday = 1.0 if target_date.weekday() in cal.workdays else 0.0
    target_features: list[list[float]] = []
    for slot in range(SLOTS_PER_DAY):
        ts = datetime.combine(target_date, time.min, tzinfo=tz) + timedelta(
            minutes=slot * SLOT_MINUTES
        )
        sin_v, cos_v = _cyclic(slot)
        target_features.append([
            round(_interp(temperature, ts, DEFAULT_TEMPERATURE_C), 6),
            round(_interp(humidity, ts, DEFAULT_HUMIDITY_PCT), 6),
            round(sin_v, 6),
            round(cos_v, 6),
            is_workday,
        ])

    return AdvancedLoadInputs(
        history_features=history_features,
        history_mes_labels=history_mes,
        target_features=target_features,
        target_mes_labels=cal.labels_for_day(target_date),
    )


def predict_advanced(
    inputs: AdvancedLoadInputs,
    target_date: date,
    tz,
) -> TargetForecast:
    """加载工件（带缓存）并跑高级负荷模型推理。工件缺失/不可读抛 ValueError 或 RuntimeError。"""
    bundle = load_bundle(default_load_model_path())
    return predict_with_artifact(bundle.state_dict, bundle.metadata, inputs, target_date, tz)
