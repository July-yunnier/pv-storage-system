"""Causal feature contract shared by offline load/price experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CALENDAR = ["hour_sin", "hour_cos", "week_sin", "week_cos", "month_sin", "month_cos"]
LAGS = (0, 1, 2, 3, 6, 12, 24, 48, 168)


def calendar_features(times: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(times, utc=True)
    hour = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60
    weekday = dt.dt.dayofweek.to_numpy()
    month = dt.dt.month.to_numpy() - 1
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * weekday / 7), np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
    ]).astype(np.float32)


@dataclass
class ForecastWindows:
    rows: pd.DataFrame
    x: np.ndarray
    seq: np.ndarray
    exog: np.ndarray
    calendar: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    exog_names: list[str]
    history_size: int
    horizon: int

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.rows["split"].to_numpy() == split)


def build_windows(raw: pd.DataFrame, task: str, horizon: int, history_size: int = 48,
                  variant: str = "full") -> ForecastWindows:
    """Require contiguous history and target by timestamp, never by row count.

    Only exogenous variables observed at or before the origin are inputs.
    No future weather observations or unverifiable NWP issue times are used.
    Daily boundaries are fixed from the complete source, independently of h.
    """
    if task not in {"load", "price"} or horizon not in {1, 4}:
        raise ValueError("hourly load/price contract supports h=1 or h=4")
    if history_size < 2:
        raise ValueError("history_size must be at least 2")
    raw = raw.copy()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw = raw.sort_values("ts", kind="stable")
    if raw["ts"].duplicated().any():
        raise ValueError("duplicate source timestamps must be resolved before windowing")
    raw = raw.set_index("ts").reindex(pd.date_range(raw.ts.min(), raw.ts.max(), freq="h"))
    days = raw.index.floor("D").unique()
    dev_end = days[int(len(days) * .70)]
    train_end = days[int(int(len(days) * .70) * .80)]
    y_series = raw[task].astype(float)
    if task == "load":
        exog_names = [f"w{i}" for i in range(1, 26) if f"w{i}" in raw]
    else:
        exog_names = [c for c in ("forecast_total_load", "forecast_zonal_load", "load_observed", "load_forecast", "pv_power") if c in raw]
    if variant == "no_exogenous":
        exog_names = []
    exog_raw = raw[exog_names].to_numpy(np.float32)
    values = y_series.to_numpy(np.float32)
    all_calendar = calendar_features(pd.Series(raw.index))
    maxlag = max(max(LAGS), history_size - 1)
    origins = np.arange(maxlag, len(raw) - horizon)
    seq_offsets = np.arange(-history_size + 1, 1)
    seq = values[origins[:, None] + seq_offsets]
    exog = exog_raw[origins[:, None] + seq_offsets]
    calendar = all_calendar[origins[:, None] + np.arange(-history_size + 1, horizon + 1)]
    lag_x = values[origins[:, None] - np.array(LAGS)]
    target_values = values[origins[:, None] + np.arange(1, horizon + 1)]
    origin_ts = raw.index[origins]
    target_ts = raw.index[origins + horizon]
    splits = np.where(target_ts < train_end, "train", np.where(target_ts < dev_end, "validation", "test"))
    # Embargo boundary origins: no training label reaches a validation/test
    # origin, and no validation label reaches a test origin.
    in_same_split = ((splits == "train") | ((splits == "validation") & (origin_ts >= train_end)) | ((splits == "test") & (origin_ts >= dev_end)))
    valid = (np.isfinite(seq).all(axis=1) & np.isfinite(target_values).all(axis=1)
             & np.isfinite(lag_x).all(axis=1) & np.isfinite(exog).all(axis=(1, 2)) & in_same_split)
    feature_names = [f"lag_{lag}" for lag in LAGS] + exog_names
    parts = [lag_x, exog_raw[origins]]
    if variant != "no_calendar":
        parts.append(all_calendar[origins + horizon])
        feature_names += CALENDAR
    else:
        calendar = np.zeros_like(calendar)
    x = np.column_stack(parts)
    rows = pd.DataFrame({
        "forecast_origin_ts": origin_ts, "target_ts": target_ts,
        "origin_value": values[origins], "target_value": target_values[:, -1],
        "source_position": origins, "split": splits,
    })
    return ForecastWindows(rows.loc[valid].reset_index(drop=True), x[valid], seq[valid],
                           exog[valid], calendar[valid], target_values[valid], feature_names,
                           exog_names, history_size, horizon)


def regression_metrics(y: np.ndarray, prediction: np.ndarray, cap: float | None = None) -> dict:
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.shape != prediction.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics require equal, nonempty, finite arrays")
    error = prediction - y
    mae, rmse = float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))
    result = {"n": int(y.size), "MAE": mae, "RMSE": rmse}
    if cap is not None:
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError("Cap must be finite and positive")
        result.update({"1-MAE/Cap": 1-mae/cap, "1-RMSE/Cap": 1-rmse/cap, "Cap": cap,
                       "passes_MAE_96": bool(1-mae/cap >= .96), "passes_RMSE_96": bool(1-rmse/cap >= .96)})
    return result
