"""原生频率的电价数据校验、日历特征和隔离的评估窗口。无深度学习依赖。"""
from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

FEATURE_NAMES = ["day_sin", "day_cos", "week_sin", "week_cos", "year_sin", "year_cos", "weekend"]

def calendar_features(ts: datetime) -> list[float]:
    day = (ts.hour * 60 + ts.minute) / 1440
    week = (ts.weekday() + day) / 7
    year = (ts.timetuple().tm_yday - 1 + day) / 365.25
    return [math.sin(2 * math.pi * day), math.cos(2 * math.pi * day),
            math.sin(2 * math.pi * week), math.cos(2 * math.pi * week),
            math.sin(2 * math.pi * year), math.cos(2 * math.pi * year),
            float(ts.weekday() >= 5)]


def validate_series(history: list[tuple[datetime, float]], resolution: int, *, allow_gaps: bool = False) -> None:
    if resolution not in (15, 60):
        raise ValueError("price model supports native 15 or 60 minute prices")
    if not history:
        raise ValueError("price history is empty")
    for i, (ts, value) in enumerate(history):
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ValueError("price timestamps must include timezone")
        if ts.minute % resolution or ts.second or ts.microsecond:
            raise ValueError("price timestamps must align with native resolution")
        if not math.isfinite(value):
            raise ValueError("price values must be finite")
        if i:
            delta = ts.timestamp() - history[i - 1][0].timestamp()
            if delta <= 0 or delta % (resolution * 60) or (not allow_gaps and delta != resolution * 60):
                raise ValueError("price history must be continuous, ordered and without duplicates")


def read_series(path: Path, timezone_name: str, resolution: int, default_id: str, skip_gapped_windows=False):
    series = {}
    tz = ZoneInfo(timezone_name)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"ts", "price"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires ts,price[,series_id]")
        for row in reader:
            ts = datetime.fromisoformat(row["ts"])
            if ts.tzinfo is None or ts.utcoffset() is None:
                raise ValueError("CSV timestamps must include timezone")
            series_id = row.get("series_id", default_id).strip()
            if not series_id:
                raise ValueError("series_id must not be empty")
            series.setdefault(series_id, []).append((ts.astimezone(tz), float(row["price"])))
    for values in series.values():
        validate_series(values, resolution, allow_gaps=skip_gapped_windows)
    if not series:
        raise ValueError("CSV is empty")
    return dict(sorted(series.items()))


def split_origins(series, context: int, horizon: int, steps_per_day: int):
    """按同一绝对时间边界分组切分，防止一条序列训练看见另一条序列测试期。"""
    all_times = sorted({ts for values in series.values() for ts, _ in values})
    train_cut = all_times[int(len(all_times) * 0.7)]
    validation_cut = all_times[int(len(all_times) * 0.85)]
    groups = {"train": [], "validation": [], "test": []}
    for index, (series_id, values) in enumerate(series.items()):
        # 每日同一原点，测试目标不重叠；context 可以引用更早 split 的已知价格。
        for origin in range(context, len(values) - horizon + 1, horizon):
            # 缺口前后分开使用；不插值，不让输入/目标窗口跨越断档。
            window = values[origin - context:origin + horizon]
            if window[-1][0].timestamp() - window[0][0].timestamp() != (len(window) - 1) * 86400 / steps_per_day:
                continue
            first, last = values[origin][0], values[origin + horizon - 1][0]
            if last < train_cut:
                split = "train"
            elif first >= train_cut and last < validation_cut:
                split = "validation"
            elif first >= validation_cut:
                split = "test"
            else:
                continue  # 不允许一个 target window 跨 split
            groups[split].append((series_id, index, origin))
    if any(not rows for rows in groups.values()):
        raise ValueError("数据不足，无法构造隔离的 train/validation/test 窗口")
    trained_ids = {row[0] for row in groups["train"]}
    if trained_ids != set(series):
        raise ValueError("each series must have training windows before the shared train cutoff")
    return groups, train_cut, validation_cut
