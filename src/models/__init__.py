"""数据模型层。

统一约定（全系统必须遵守）
--------------------------
1. **时间戳一律 ISO 8601 带时区**，如 2026-09-19T08:00:00+08:00
2. **单位写进字段名**：capacity_kwp / power_kw / price_yuan_per_kwh
   —— kW / kWp / kWh 混淆是能源行业最常见的错误来源
3. **缺失值用 None，绝不用 0 代替**
   —— 夜间功率 0 是有效观测值，用 0 填缺失会把「未观测」当成「确认为零」

模块
----
site      站点档案（Location / PVConfig / LoadConfig / StorageConfig / Site）
forecast  预测结果与精度口径（ForecastResult / AccuracyCaliber）
schedule  调度结果（ScheduleEntry / DispatchResult）
"""
from __future__ import annotations

from .forecast import AccuracyCaliber, ForecastResult, ForecastTarget, make_accuracy
from .schedule import Action, DispatchResult, ScheduleEntry
from .site import (
    LoadConfig,
    Location,
    PVConfig,
    Site,
    StorageConfig,
    site_from_config,
)

__all__ = [
    # site
    "Location",
    "PVConfig",
    "LoadConfig",
    "StorageConfig",
    "Site",
    "site_from_config",
    # forecast
    "ForecastTarget",
    "AccuracyCaliber",
    "ForecastResult",
    "make_accuracy",
    # schedule
    "Action",
    "ScheduleEntry",
    "DispatchResult",
]
