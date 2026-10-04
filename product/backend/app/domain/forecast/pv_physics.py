"""光伏物理预测链：辐射预报 → pvlib 五步 → 15 分钟功率。

五步链（每步都是 pvlib 的同行评审实现，不自研物理）：
  ① 太阳位置   solarposition.get_solarposition
  ② 组件面辐射 irradiance.get_total_irradiance（Hay-Davies 转置）
  ③ 电池温度   temperature.faiman
  ④ 直流功率   pvsystem.pvwatts_dc（含温度修正）
  ⑤ 损耗与裁剪  system_loss 折减，裁剪到 [0, capacity_kwp]

粒度约定：气象预报逐小时，线性插值到 15 分钟（写进 assumptions/notes）。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, tzinfo

import pandas as pd
import pvlib

from app.domain.models import PVConfig
from app.infrastructure.weather.base import WeatherPoint

SLOTS_PER_DAY = 96
SLOT_MINUTES = 15


def _to_quarter_hourly(points: list[WeatherPoint]) -> pd.DataFrame:
    """逐小时辐射 → 15 分钟线性插值。夜间/缺失按 0 处理由上游保证。"""
    frame = pd.DataFrame(
        {
            "ghi": [p.ghi_wm2 for p in points],
            "dni": [p.dni_wm2 for p in points],
            "dhi": [p.dhi_wm2 for p in points],
            "temp_air": [p.temperature_c if p.temperature_c is not None else 25.0 for p in points],
            "wind": [p.wind_speed_ms if p.wind_speed_ms is not None else 2.0 for p in points],
        },
        index=pd.DatetimeIndex([p.ts for p in points]),
    )
    quarter = frame.resample(f"{SLOT_MINUTES}min").interpolate("linear")
    # resample 末尾不生成最后一条之后的外推点；起点前不外推（上游从 00:00 开始）
    return quarter.ffill().bfill()


def pv_power_series(
    weather: list[WeatherPoint],
    pv: PVConfig,
    lat: float,
    lon: float,
) -> pd.Series:
    """辐射序列 → 光伏功率序列（kW），15 分钟粒度。"""
    if not weather:
        raise ValueError("weather points are required")
    frame = _to_quarter_hourly(weather)
    times = frame.index

    position = pvlib.solarposition.get_solarposition(times, lat, lon)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=pv.tilt_deg,
        surface_azimuth=pv.azimuth_deg,
        solar_zenith=position["apparent_zenith"],
        solar_azimuth=position["azimuth"],
        dni=frame["dni"],
        ghi=frame["ghi"],
        dhi=frame["dhi"],
        dni_extra=pvlib.irradiance.get_extra_radiation(times),
        model="haydavies",
    )
    temp_cell = pvlib.temperature.faiman(poa["poa_global"], frame["temp_air"], frame["wind"])
    effective = poa["poa_global"].clip(lower=0) * (1 - pv.system_loss)
    power_w = pvlib.pvsystem.pvwatts_dc(
        effective, temp_cell, pdc0=pv.capacity_kwp * 1000.0, gamma_pdc=-0.004
    )
    power_kw = (power_w / 1000.0).clip(lower=0.0, upper=pv.capacity_kwp)
    return pd.Series(power_kw, index=times, name="pv_power_kw")


def day_power_values(
    weather: list[WeatherPoint],
    pv: PVConfig,
    lat: float,
    lon: float,
    target_date: date,
    tz: tzinfo,
) -> list[float]:
    """取目标日 96 点的功率值列表。"""
    series = pv_power_series(weather, pv, lat, lon)
    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    values: list[float] = []
    for slot in range(SLOTS_PER_DAY):
        ts = pd.Timestamp(day_start + timedelta(minutes=SLOT_MINUTES * slot))
        values.append(round(float(series.get(ts, 0.0)), 6))
    return values


def clearsky_series(
    pv: PVConfig,
    lat: float,
    lon: float,
    target_date: date,
    tz: tzinfo,
) -> list[float]:
    """晴空基准：无云日的 96 点功率（Ineichen 晴空模型 + 同一转换链）。

    供仿真种子数据和"云况评估"使用：实测/晴空 ≈ 当天云量水平。
    """
    day_start = datetime.combine(target_date, time.min, tzinfo=tz)
    times = pd.DatetimeIndex(
        [day_start + timedelta(minutes=SLOT_MINUTES * slot) for slot in range(SLOTS_PER_DAY)]
    )
    position = pvlib.solarposition.get_solarposition(times, lat, lon)
    airmass_abs = pvlib.atmosphere.get_absolute_airmass(
        pvlib.atmosphere.get_relative_airmass(position["apparent_zenith"])
    )
    linke = pvlib.clearsky.lookup_linke_turbidity(times, lat, lon)
    clearsky = pvlib.clearsky.ineichen(position["apparent_zenith"], airmass_abs, linke)
    weather = [
        WeatherPoint(
            ts=ts.to_pydatetime(),
            ghi_wm2=float(row["ghi"]),
            dni_wm2=float(row["dni"]),
            dhi_wm2=float(row["dhi"]),
        )
        for ts, row in clearsky.iterrows()
    ]
    return day_power_values(weather, pv, lat, lon, target_date, tz)
