"""GEFCom2014 电价赛道（Price Track）读取与任务切分。

数据事实（已逐文件核对，勿凭记忆改动）：
- 15 个任务，每个任务是**单日 24 小时日前预测**，目标为 Zone 1 的小时级
  locational marginal price（LMP，$/MWh）。
- 每个 `TaskN_P.csv` 给出从 2011-01-01 到该任务预测日的完整历史，最后一
  个自然日的 `Zonal Price` 留空待预测；`Forecasted Total Load` 与
  `Forecasted Zonal Load` 是**当时已知的日前负荷预报**（可作为未来协变量）。
- Task 15 的文件覆盖最全（至 2013-12-17），因此统一用它作为历史序列；
  2013-12-17 当天的真实电价由 `Solution to Task15_P.csv` 提供。
- `BenchmarkN_P.csv` 是官方基准模型给出的 0.01–0.99 共 99 个分位数预测，
  可以直接当作强基准对手。

用法::

    series, days = load_gefcom_price("/path/to/Price")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain.forecast.price_features import PriceSeries

TASK_DAYS: tuple[str, ...] = (
    "2013-06-16",
    "2013-06-17",
    "2013-06-24",
    "2013-07-04",
    "2013-07-09",
    "2013-07-13",
    "2013-07-16",
    "2013-07-18",
    "2013-07-19",
    "2013-07-20",
    "2013-07-24",
    "2013-07-25",
    "2013-12-07",
    "2013-12-08",
    "2013-12-17",
)

BENCHMARK_QUANTILES: tuple[float, ...] = tuple(
    round(0.01 * i, 2) for i in range(1, 100)
)


@dataclass
class GefcomPrice:
    series: PriceSeries
    task_days: list[pd.Timestamp]
    root: Path


def _parse_timestamp(values: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, format="%m%d%Y %H:%M"))


def load_gefcom_price(root: str | Path) -> GefcomPrice:
    """读取并拼成连续小时序列。root 指向解压后的 `Price/` 目录。"""
    base = Path(root)
    if not (base / "Task 15" / "Task15_P.csv").exists():
        raise FileNotFoundError(f"GEFCom2014 price data not found under {base}")

    frame = pd.read_csv(base / "Task 15" / "Task15_P.csv")
    frame["ts"] = _parse_timestamp(frame["timestamp"])
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    solution = base / "Solution to Task15" / "Solution to Task15_P.csv"
    if solution.exists():
        sol = pd.read_csv(solution)
        sol["ts"] = _parse_timestamp(sol["timestamp"])
        lookup = dict(zip(sol["ts"], sol["Zonal Price"]))
        frame["Zonal Price"] = [
            price if np.isfinite(price) else lookup.get(ts, np.nan)
            for ts, price in zip(frame["ts"], frame["Zonal Price"])
        ]

    grid = pd.date_range(frame["ts"].min(), frame["ts"].max(), freq="h")
    frame = frame.set_index("ts").reindex(grid)
    series = PriceSeries(
        ts=grid,
        price=frame["Zonal Price"].to_numpy(dtype="float64"),
        load=frame["Forecasted Zonal Load"].to_numpy(dtype="float64"),
        load_total=frame["Forecasted Total Load"].to_numpy(dtype="float64"),
    )
    days = [pd.Timestamp(day) for day in TASK_DAYS]
    return GefcomPrice(series=series, task_days=days, root=base)


def load_task_benchmark(root: str | Path, task: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """读取官方基准的 99 分位数预测，返回 (时间戳, 形状 (24, 99) 的数组)。"""
    base = Path(root)
    candidates = [
        base / f"Task {task}" / f"Benchmark{task}_P.csv",
        base / f"Task {task}" / f"Benchmark{task}_P_new3.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"benchmark file for task {task} not found")

    frame = pd.read_csv(path)
    frame["ts"] = _parse_timestamp(frame["timestamp"])
    quantile_cols = [c for c in frame.columns if c not in ("ZONEID", "timestamp", "ts")]
    quantile_cols = sorted(quantile_cols, key=lambda c: float(c))
    return pd.DatetimeIndex(frame["ts"]), frame[quantile_cols].to_numpy(dtype="float64")


def actual_for_day(gefcom: GefcomPrice, day: pd.Timestamp) -> np.ndarray:
    """取某任务日的 24 小时真实电价。"""
    window = gefcom.series.ts[(gefcom.series.ts >= day) & (gefcom.series.ts < day + pd.Timedelta("24h"))]
    if len(window) != 24:
        raise ValueError(f"expected 24 hours for {day.date()}, got {len(window)}")
    pos = gefcom.series.ts.get_indexer(window)
    actual = gefcom.series.price[pos]
    if not np.isfinite(actual).all():
        raise ValueError(f"actual price missing for {day.date()}")
    return actual


def future_load(gefcom: GefcomPrice, day: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """任务日已知的日前负荷预报（Total / Zonal）。"""
    window = gefcom.series.ts[(gefcom.series.ts >= day) & (gefcom.series.ts < day + pd.Timedelta("24h"))]
    pos = gefcom.series.ts.get_indexer(window)
    return gefcom.series.load_total[pos], gefcom.series.load[pos]
