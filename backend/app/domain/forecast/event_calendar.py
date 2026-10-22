"""事件日历：把"计划性事件"变成电价模型可用的外生特征。

统计模型抓不住的三类价格冲击，本质上都不是随机噪声，而是**有日历的事件**：

1. **机组检修**：大修/小修计划提前数月公布，检修容量直接压缩供给，尖峰概率上升。
   特征取 `in_maintenance`（是否在窗口内）与 `days_to_maintenance`（距下一次检修天数）。
2. **寒潮 / 高温预警**：气象台提前 3–7 天发布，驱动采暖/制冷负荷。
   特征取 `cold_wave` / `heat_wave` 强度（0–1）。
3. **燃料价格**：煤价/气价决定边际机组成本，是电价的中期锚。
   特征取 `fuel_index_norm`（相对近一年分位）与 `fuel_change_7d`（7 日变化率）。

纪律：
- V2 按发布时间逐 origin 回放，只纳入发报时刻已知的信息；V1 仅保留演示兼容。
  检修计划、预警和燃料指数都必须记录实际发布时间，事后信息不能提前进入特征。
- 事件缺失时整列为 NaN，`select_active_columns` 会自动剔除，不影响无事件的市场。
- 本模块只负责"把日历变成数字"，不做任何价格推断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

#: 事件特征列（顺序固定，追加到 FEATURE_COLUMNS 末尾）
EVENT_FEATURES: tuple[str, ...] = (
    "event_pressure",
    "in_maintenance",
    "days_to_maintenance",
    "cold_wave",
    "heat_wave",
    "fuel_index_norm",
    "fuel_change_7d",
)

_MAINTENANCE = "maintenance"
_COLD = "cold_wave"
_HEAT = "heat_wave"
_SEVERITY = {_MAINTENANCE: 0.6, _COLD: 0.9, _HEAT: 0.8}


@dataclass(frozen=True)
class MarketEvent:
    kind: str
    start: date
    end: date
    severity: float = 1.0
    note: str = ""

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass
class EventCalendar:
    """事件集合 + 燃料价格指数。所有方法都是纯函数，便于测试。"""

    events: list[MarketEvent] = field(default_factory=list)
    fuel: pd.Series | None = None          # index = date, value = 指数
    name: str = "empty"
    data: object | None = None  # schema v2, publication-aware records

    # ---- 构造 ----
    @classmethod
    def from_yaml(cls, path: str | Path) -> "EventCalendar":
        import yaml

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"event calendar not found: {source}")
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if payload.get("schema_version") == 2:
            from app.domain.forecast.event_data import EventData

            data = EventData(payload, source.parent)
            return cls(name=data.name, data=data)
        if payload.get("schema_version", 1) != 1:
            raise ValueError("unsupported event schema_version")
        events: list[MarketEvent] = []
        for item in payload.get("events", []):
            kind = str(item["kind"])
            if kind not in _SEVERITY:
                raise ValueError(f"unsupported event kind: {kind}")
            start = _as_date(item["start"])
            end = _as_date(item["end"])
            if end < start:
                raise ValueError(f"event end before start: {item}")
            events.append(
                MarketEvent(
                    kind=kind,
                    start=start,
                    end=end,
                    severity=float(item.get("severity", 1.0)),
                    note=str(item.get("note", "")),
                )
            )
        fuel = None
        fuel_path = payload.get("fuel_index_csv")
        if fuel_path:
            csv_path = (source.parent / fuel_path).resolve()
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                fuel = pd.Series(
                    frame["index"].astype(float).to_numpy(),
                    index=pd.to_datetime(frame["date"]).dt.date,
                ).sort_index()
        return cls(events=events, fuel=fuel, name=str(payload.get("name", source.stem)))

    # ---- 特征 ----
    def features(self, ts: pd.DatetimeIndex, issued_at=None) -> dict[str, np.ndarray]:
        """V2 requires an aware issue time; legacy data is demonstration-only."""
        if self.data is not None:
            return self.data.features(ts, issued_at)
        return self._legacy_features(ts)

    def _legacy_features(self, ts: pd.DatetimeIndex) -> dict[str, np.ndarray]:
        """按小时时间戳生成事件特征。日期粒度事件在当日内保持不变。"""
        days = [t.date() for t in ts]
        n = len(days)
        pressure = np.zeros(n)
        maintenance = np.zeros(n)
        cold = np.zeros(n)
        heat = np.zeros(n)

        for i, day in enumerate(days):
            for event in self.events:
                if not event.covers(day):
                    continue
                weight = _SEVERITY[event.kind] * event.severity
                pressure[i] += weight
                if event.kind == _MAINTENANCE:
                    maintenance[i] = 1.0
                elif event.kind == _COLD:
                    cold[i] = max(cold[i], event.severity)
                elif event.kind == _HEAT:
                    heat[i] = max(heat[i], event.severity)

        return {
            "event_pressure": pressure,
            "in_maintenance": maintenance,
            "days_to_maintenance": self._days_to_maintenance(days),
            "cold_wave": cold,
            "heat_wave": heat,
            "fuel_index_norm": self._fuel_norm(days),
            "fuel_change_7d": self._fuel_change(days),
        }

    def _days_to_maintenance(self, days: list[date]) -> np.ndarray:
        """距下一次检修开始的天数；检修窗口内为 0；60 天内没有则 NaN。"""
        windows = [(e.start, e.end) for e in self.events if e.kind == _MAINTENANCE]
        starts = sorted(start for start, _ in windows)
        out = np.full(len(days), np.nan)
        if not starts:
            return out
        for i, day in enumerate(days):
            if any(start <= day <= end for start, end in windows):
                out[i] = 0.0
                continue
            upcoming = [start for start in starts if start > day]
            if not upcoming:
                continue
            out[i] = float(min((upcoming[0] - day).days, 60))
        return out

    def _fuel_norm(self, days: list[date]) -> np.ndarray:
        """燃料指数在过去 365 天里的分位（0–1），刻画"当前贵不贵"。"""
        if self.fuel is None or self.fuel.empty:
            return np.full(len(days), np.nan)
        values = self.fuel
        out = np.full(len(days), np.nan)
        for i, day in enumerate(days):
            window = values[(values.index >= day - timedelta(days=365)) & (values.index <= day)]
            if len(window) < 30:
                continue
            current = window.iloc[-1]
            out[i] = float((window <= current).mean())
        return out

    def _fuel_change(self, days: list[date]) -> np.ndarray:
        """燃料指数 7 日变化率。"""
        if self.fuel is None or self.fuel.empty:
            return np.full(len(days), np.nan)
        values = self.fuel
        out = np.full(len(days), np.nan)
        for i, day in enumerate(days):
            window = values[values.index <= day]
            if len(window) < 8:
                continue
            latest = window.iloc[-1]
            week_ago = window.iloc[-8]
            if week_ago:
                out[i] = float(latest / week_ago - 1.0)
        return out

    # ---- 描述 ----
    def summary(self) -> dict:
        if self.data is not None:
            return self.data.summary()
        by_kind: dict[str, int] = {}
        for event in self.events:
            by_kind[event.kind] = by_kind.get(event.kind, 0) + 1
        return {
            "name": self.name,
            "schema_version": 1,
            "data_kind": "example",
            "events": len(self.events),
            "by_kind": by_kind,
            "fuel_points": 0 if self.fuel is None else int(len(self.fuel)),
        }


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def load_calendar(path: str | Path | None) -> EventCalendar | None:
    """Reload on each request so periodic imports and source hash changes take effect."""
    if not path:
        return None
    try:
        return EventCalendar.from_yaml(path)
    except FileNotFoundError:
        return None
