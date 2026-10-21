"""电价序列接入与规范化：把省交易中心/代理购电导出的原始表变成模型能吃的序列。

真实数据落地时最容易出错的四件事，全部在这里一次做对：

1. **单位**：广东省现货出清价按「元/兆瓦时」发布（数值几百），模型要「元/千瓦时」。
   差 1000 倍，错了不会报错，只会让预测和套利窗口整体错三个数量级。
2. **时区**：交易中心导出的时间戳通常是**本地时间且不带时区**。直接当 UTC 存会偏 8 小时，
   表现是"曲线整体平移"，很难一眼看出来。
3. **粒度**：现货是 15 分钟出清，模型是小时粒度。必须用**均值**聚合，
   不能用插值或取整点——15 分钟里有 4 个价，取端点会丢一半信息。
4. **负电价**：现货允许负值，任何"小于 0 视为异常并置空"的清洗都会毁掉最贵的信息。

本模块只做规范化与体检，不猜业务含义：原始表长什么样，由调用方通过列名/单位/时区参数声明。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

#: 常见列名别名，减少调用方记忆负担（仍然允许显式指定）
TS_ALIASES = ("ts", "timestamp", "time", "时间", "时间戳", "日期时间", "datetime")
PRICE_ALIASES = (
    "price", "clearing_price", "lmp", "出清价", "出清价格", "电价",
    "节点电价", "统一出清价", "价格",
)


@dataclass
class IngestReport:
    rows_in: int = 0
    rows_out: int = 0
    unit_detected: str = ""
    scale_applied: float = 1.0
    duplicates_removed: int = 0
    missing_hours: int = 0
    negative_points: int = 0
    spike_points: int = 0
    first: datetime | None = None
    last: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "unit_detected": self.unit_detected,
            "scale_applied": self.scale_applied,
            "duplicates_removed": self.duplicates_removed,
            "missing_hours": self.missing_hours,
            "negative_points": self.negative_points,
            "spike_points": self.spike_points,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
            "warnings": self.warnings,
        }


def _pick_column(frame: pd.DataFrame, explicit: str | None, aliases: tuple[str, ...], what: str) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"{what} column not found: {explicit}; columns={list(frame.columns)}")
        return explicit
    for name in aliases:
        if name in frame.columns:
            return name
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in aliases:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise ValueError(f"cannot locate {what} column; columns={list(frame.columns)}")


def detect_unit(prices: np.ndarray) -> tuple[str, float]:
    """按数量级判断单位，返回 (单位名, 到元/千瓦时的换算系数)。

    广东现货出清价典型区间 0–1500 元/兆瓦时；折成元/千瓦时是 0–1.5。
    阈值取 20：中位数高于 20 基本只可能是元/兆瓦时。
    """
    values = prices[np.isfinite(prices)]
    if values.size == 0:
        return "unknown", 1.0
    median = float(np.median(np.abs(values)))
    if median > 20.0:
        return "CNY/MWh", 0.001
    return "CNY/kWh", 1.0


def normalize_spot(
    frame: pd.DataFrame,
    *,
    ts_col: str | None = None,
    price_col: str | None = None,
    tz: str = "Asia/Shanghai",
    unit: str = "auto",
) -> tuple[pd.DataFrame, IngestReport]:
    """原始表 → 干净的 15 分钟/小时序列（列：ts, price，ts 带时区）。

    `unit` 可显式给 `CNY/MWh` / `CNY/kWh`，给 `auto` 时按数量级判断并记入报告。
    """
    report = IngestReport()
    if frame.empty:
        raise ValueError("empty price table")

    ts_name = _pick_column(frame, ts_col, TS_ALIASES, "timestamp")
    price_name = _pick_column(frame, price_col, PRICE_ALIASES, "price")

    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(frame[ts_name], errors="coerce"),
            "price": pd.to_numeric(frame[price_name], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ts"])
    # 稳定排序：同一时间戳出现多次时，保留"后出现的"才有确定语义
    out = out.sort_values("ts", kind="mergesort")
    report.rows_in = int(len(frame))

    before = len(out)
    out = out.drop_duplicates(subset="ts", keep="last")
    report.duplicates_removed = before - len(out)

    if unit == "auto":
        detected, scale = detect_unit(out["price"].to_numpy(dtype="float64"))
        report.unit_detected = detected
    elif unit in ("CNY/MWh", "元/MWh", "yuan_per_mwh"):
        detected, scale = "CNY/MWh", 0.001
        report.unit_detected = "CNY/MWh"
    elif unit in ("CNY/kWh", "元/kWh", "yuan_per_kwh"):
        detected, scale = "CNY/kWh", 1.0
        report.unit_detected = "CNY/kWh"
    else:  # pragma: no cover - 参数校验
        raise ValueError(f"unsupported unit: {unit}")

    out["price"] = out["price"] * scale
    report.scale_applied = scale

    # 时区：无时区信息按交易所在地本地时间解释
    if out["ts"].dt.tz is None:
        out["ts"] = out["ts"].dt.tz_localize(ZoneInfo(tz))
        report.warnings.append(f"timestamps had no timezone; assumed {tz}")
    else:
        out["ts"] = out["ts"].dt.tz_convert(ZoneInfo(tz))

    prices = out["price"].to_numpy(dtype="float64")
    report.negative_points = int(np.sum(prices < 0))
    report.spike_points = int(np.sum(prices > 3.0))  # 元/kWh 下 3 元已属极端尖峰

    # 连续性检查：按原始粒度推断应有步长
    if len(out) >= 3:
        step = out["ts"].diff().dropna().median()
        expected = pd.date_range(out["ts"].iloc[0], out["ts"].iloc[-1], freq=step)
        report.missing_hours = int(max(len(expected) - len(out), 0))

    report.rows_out = int(len(out))
    report.first = out["ts"].iloc[0].to_pydatetime()
    report.last = out["ts"].iloc[-1].to_pydatetime()
    if report.negative_points:
        report.warnings.append(f"{report.negative_points} negative prices kept as-is (valid in spot market)")
    if report.missing_hours > 24:
        report.warnings.append(f"{report.missing_hours} missing stamps detected; check export window")
    return out.reset_index(drop=True), report


def aggregate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """按小时取均值（现货 15 分钟 → 小时）。缺口保留 NaN，不填充。"""
    if frame.empty:
        raise ValueError("empty frame")
    indexed = frame.set_index("ts")
    hourly = indexed["price"].resample("1h").mean().dropna()
    return hourly.reset_index()
