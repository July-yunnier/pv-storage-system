"""电价预测特征工程（纯函数，小时粒度）。

设计约束——**发报时刻 t0 之后的信息一律不许进特征**：

- 历史侧（origin 及以前）：价格滞后、滚动统计、当前负荷；
- 未来侧（target 小时）：只允许"当时已知的未来量"——日前负荷预测、
  目录电价（政策表）、数值天气预报（温度/辐照）、日历。

这正是 MQ-CNN（Wen et al., NIPS 2017）的 direct multi-horizon 设定：
一次前向直接吐出 1..H 小时的分位数，不靠递归，误差不随步长累积。

行样本是 (origin, horizon) 二元组：origin 是"最后一小时已知数据"，
horizon 是提前小时数。这样既能训日前的 24 点，也能训 72/168 点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.domain.forecast import price_calendar as cal

#: 特征名固定顺序。新增特征必须追加到末尾，旧模型工件才不会错位。
FEATURE_COLUMNS: tuple[str, ...] = (
    "hour",
    "dow",
    "is_weekend",
    "is_holiday",
    "is_pre_holiday",
    "is_post_holiday",
    "is_workday",
    "month",
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
    "season_pressure",
    "horizon",
    "tou_price_target",
    "tou_price_origin",
    "tou_delta",
    "price_origin",
    "price_spread_origin",
    "price_lag_d1",
    "price_lag_w1",
    "price_mean_24",
    "price_std_24",
    "price_min_24",
    "price_max_24",
    "price_mean_168",
    "price_rank_168",
    "load_target",
    "load_origin",
    "load_ratio_168",
    "temp_target",
    "temp_origin",
    "ghi_target",
)

#: 需要提前 168 小时历史才能算的特征，冷启动时整列缺失。
WEEK_LAG_HOURS = 168
DAY_LAG_HOURS = 24


@dataclass
class PriceSeries:
    """连续小时序列。所有数组与 ts 等长，缺失用 NaN（不填 0）。"""

    ts: pd.DatetimeIndex
    price: np.ndarray
    load: np.ndarray | None = None          # 目标时刻已知的负荷（GEFCom-P 的 Zonal Load 预报）
    load_total: np.ndarray | None = None    # 系统总负荷（GEFCom-P 提供，线上可为空）
    temperature: np.ndarray | None = None
    ghi: np.ndarray | None = None
    tou_price: np.ndarray | None = None     # 目录电价，政策给定，未来时刻也有值

    def __post_init__(self) -> None:
        n = len(self.ts)
        if n == 0:
            raise ValueError("price series cannot be empty")
        if len(self.price) != n:
            raise ValueError("price length must match ts")
        for field in ("load", "load_total", "temperature", "ghi", "tou_price"):
            value = getattr(self, field)
            if value is not None and len(value) != n:
                raise ValueError(f"{field} length must match ts")

    def slice_index(self, issue_ts: pd.Timestamp) -> int:
        """返回 <= issue_ts 的最后一个索引；早于序列起点返回 -1。"""
        return int(self.ts.searchsorted(issue_ts, side="right")) - 1


def ensure_hourly(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """把任意分辨率的时间戳提升为连续小时网格（左端点对齐）。"""
    if len(ts) == 0:
        raise ValueError("empty ts")
    return pd.date_range(ts.min().floor("h"), ts.max().floor("h"), freq="h")


def to_hourly_frame(
    ts: pd.DatetimeIndex, values: np.ndarray, how: str = "mean"
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """把 15 分钟/日级序列聚合到小时网格；缺口保留 NaN。"""
    frame = pd.Series(np.asarray(values, dtype="float64"), index=pd.DatetimeIndex(ts))
    grid = ensure_hourly(frame.index)
    return grid, frame.resample("h").agg(how).reindex(grid).to_numpy(dtype="float64")


def forward_fill_gaps(price: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """短缺口前向填充（<=max_gap 小时），长缺口保持 NaN 由调用方决定丢弃。"""
    filled = pd.Series(np.asarray(price, dtype="float64")).ffill(limit=max_gap)
    return filled.to_numpy(dtype="float64")


def _calendar_block(ts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    days = [t.date() for t in ts]
    hour = np.asarray([t.hour for t in ts], dtype="float32")
    dow = np.asarray([t.dayofweek for t in ts], dtype="float32")
    month = np.asarray([t.month for t in ts], dtype="float32")
    doy = np.asarray([t.dayofyear for t in ts], dtype="float32")
    return {
        "hour": hour,
        "dow": dow,
        "is_weekend": (dow >= 5).astype("float32"),
        "is_holiday": np.asarray([1.0 if cal.is_holiday(d) else 0.0 for d in days], dtype="float32"),
        "is_pre_holiday": np.asarray([float(cal.neighbor_flags(d)[0]) for d in days], dtype="float32"),
        "is_post_holiday": np.asarray([float(cal.neighbor_flags(d)[1]) for d in days], dtype="float32"),
        "is_workday": np.asarray([1.0 if cal.is_workday(d) else 0.0 for d in days], dtype="float32"),
        "month": month,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25).astype("float32"),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25).astype("float32"),
        "hour_sin": np.sin(2 * np.pi * hour / 24.0).astype("float32"),
        "hour_cos": np.cos(2 * np.pi * hour / 24.0).astype("float32"),
        "season_pressure": np.asarray(
            [float(cal.season_pressure(d)) for d in days], dtype="float32"
        ),
    }


class _CausalStats:
    """全序列一次性算好的因果滚动统计（窗口只回看，不可能泄漏未来）。"""

    def __init__(self, price: np.ndarray) -> None:
        s = pd.Series(np.asarray(price, dtype="float64"))
        self.mean_24 = s.rolling(DAY_LAG_HOURS, min_periods=1).mean().to_numpy()
        self.std_24 = s.rolling(DAY_LAG_HOURS, min_periods=1).std(ddof=0).to_numpy()
        self.min_24 = s.rolling(DAY_LAG_HOURS, min_periods=1).min().to_numpy()
        self.max_24 = s.rolling(DAY_LAG_HOURS, min_periods=1).max().to_numpy()
        self.mean_168 = s.rolling(WEEK_LAG_HOURS, min_periods=1).mean().to_numpy()
        self.rank_168 = s.rolling(WEEK_LAG_HOURS, min_periods=1).rank(pct=True).to_numpy()


def build_samples(
    series: PriceSeries,
    origins: np.ndarray | None = None,
    horizon: int = 24,
    min_origin: int = WEEK_LAG_HOURS - 1,
    require_target: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """构造 (origin, horizon) 样本。

    Args:
        origins: 允许的 origin 索引；None 表示全部合法 origin。
        horizon: 最大提前小时数，生成 1..horizon 全部步长。
        min_origin: 最小 origin 索引（保证 168h 滚窗可用）。
        require_target: True 时丢弃目标价格缺失的样本（训练用）；推理时应为 False。

    Returns:
        (X, y, target_ts, origin_ts)，四者行数一致。
    """
    n = len(series.ts)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    price = forward_fill_gaps(series.price)
    stats = _CausalStats(price)

    if origins is None:
        origins = np.arange(n)
    origins = np.asarray(origins, dtype=int)
    origins = origins[(origins >= min_origin) & (origins < n)]

    cal_block = _calendar_block(series.ts)
    stats24 = (stats.mean_24, stats.std_24, stats.min_24, stats.max_24)
    load_hourly = series.load
    load_stats = _CausalStats(np.nan_to_num(load_hourly, nan=np.nan)) if load_hourly is not None else None

    frames: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    target_stamps: list[pd.DatetimeIndex] = []
    origin_stamps: list[pd.DatetimeIndex] = []

    for step in range(1, horizon + 1):
        target_idx = origins + step
        keep = target_idx < n
        if not keep.any():
            continue
        o = origins[keep]
        t = target_idx[keep]
        cols = [
            cal_block["hour"][t],
            cal_block["dow"][t],
            cal_block["is_weekend"][t],
            cal_block["is_holiday"][t],
            cal_block["is_pre_holiday"][t],
            cal_block["is_post_holiday"][t],
            cal_block["is_workday"][t],
            cal_block["month"][t],
            cal_block["doy_sin"][t],
            cal_block["doy_cos"][t],
            cal_block["hour_sin"][t],
            cal_block["hour_cos"][t],
            cal_block["season_pressure"][t],
            np.full(o.size, float(step), dtype="float32"),
            _at(series.tou_price, t),
            _at(series.tou_price, o),
            _at(series.tou_price, t) - _at(series.tou_price, o),
            price[o],
            price[o] - _at(series.tou_price, o),
            _shifted(price, o, DAY_LAG_HOURS - 1),
            _shifted(price, o, WEEK_LAG_HOURS - 1),
            stats24[0][o],
            stats24[1][o],
            stats24[2][o],
            stats24[3][o],
            stats.mean_168[o],
            stats.rank_168[o],
            _at(series.load, t),
            _at(series.load, o),
            _load_ratio(series.load, load_stats, t, o),
            _at(series.temperature, t),
            _at(series.temperature, o),
            _at(series.ghi, t),
        ]
        frames.append(np.column_stack(cols).astype("float32"))
        ys.append(price[t].astype("float64"))
        target_stamps.append(series.ts[t])
        origin_stamps.append(series.ts[o])

    if not frames:
        return (
            pd.DataFrame(columns=list(FEATURE_COLUMNS)),
            np.asarray([]),
            pd.DatetimeIndex([]),
            pd.DatetimeIndex([]),
        )

    x = pd.DataFrame(np.vstack(frames), columns=list(FEATURE_COLUMNS))
    y = np.concatenate(ys)
    target_index = pd.DatetimeIndex(np.concatenate([s.values for s in target_stamps]))
    origin_index = pd.DatetimeIndex(np.concatenate([s.values for s in origin_stamps]))

    if require_target:
        mask = np.isfinite(y)
        x = x.loc[mask].reset_index(drop=True)
        y = y[mask]
        target_index = target_index[mask]
        origin_index = origin_index[mask]
    return x, y, target_index, origin_index


def _at(values: np.ndarray | None, idx: np.ndarray) -> np.ndarray:
    if values is None:
        return np.full(idx.size, np.nan, dtype="float32")
    return np.asarray(values, dtype="float64")[idx].astype("float32")


def _shifted(values: np.ndarray, idx: np.ndarray, back: int) -> np.ndarray:
    """取 idx-back 处的值，越界给 NaN（不做环绕，避免把序列终点泄漏到起点）。"""
    src = idx - back
    out = np.full(idx.size, np.nan)
    ok = src >= 0
    out[ok] = values[src[ok]]
    return out.astype("float32")


def _load_ratio(
    load: np.ndarray | None, stats: "_CausalStats | None", target_idx: np.ndarray, origin_idx: np.ndarray
) -> np.ndarray:
    if load is None or stats is None:
        return np.full(target_idx.size, np.nan, dtype="float32")
    base = stats.mean_168[origin_idx]
    value = np.asarray(load, dtype="float64")[target_idx]
    ratio = np.divide(value, base, out=np.full(base.shape, np.nan), where=(base != 0))
    return ratio.astype("float32")


def select_active_columns(x: pd.DataFrame, min_non_null: float = 0.05) -> list[str]:
    """训练集上非空比例达标的列。全空列（如无目录电价的市场）直接剔除。"""
    if len(x) == 0:
        return []
    ratio = x.notna().mean()
    return [c for c in FEATURE_COLUMNS if c in x.columns and ratio[c] >= min_non_null]


def column_medians(x: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for name in columns:
        value = float(x[name].median()) if name in x.columns else np.nan
        medians[name] = 0.0 if not np.isfinite(value) else value
    return medians


def to_matrix(
    x: pd.DataFrame, columns: list[str], medians: dict[str, float] | None = None
) -> np.ndarray:
    """按列取矩阵并填缺失（LightGBM 能吃 NaN，但神经网络与线性模型不能）。"""
    block = x.reindex(columns=columns).to_numpy(dtype="float32").copy()
    if medians:
        for pos, name in enumerate(columns):
            value = medians.get(name)
            if value is not None:
                block[:, pos] = np.where(np.isfinite(block[:, pos]), block[:, pos], value)
    return block
