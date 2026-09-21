"""调度结果模型。

一次调度计算输出三样东西：
    1. 每个时段做什么（充/放/待机，多大功率）
    2. SOC 轨迹（用户最关心"电池还剩多少"）
    3. 收益测算（对比"不配储能"的基线）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

Action = Literal["charge", "discharge", "idle"]


@dataclass
class ScheduleEntry:
    """单个时段的调度动作。"""

    ts: datetime
    action: Action
    power_kw: float
    """功率，约定：**正 = 充电，负 = 放电**，0 = 待机。"""

    soc_eod: float
    """该时段结束时的 SOC（0~1）。"""

    tier: str | None = None
    """电价时段标签（sharp/peak/flat/valley），现货模式下可为 None。"""

    price_yuan_per_kwh: float = 0.0
    """该时段的到户电价。"""


@dataclass
class DispatchResult:
    """调度计算结果。"""

    entries: list[ScheduleEntry] = field(default_factory=list)

    # ---- 电费测算（元）----
    baseline_cost_yuan: float = 0.0
    """不配储能时的电费。"""

    optimized_cost_yuan: float = 0.0
    """按本调度执行后的电费。"""

    # ---- 运行统计 ----
    charged_kwh: float = 0.0
    """总充电量（交流侧）。"""

    discharged_kwh: float = 0.0
    """总放电量（交流侧）。"""

    cycles_used: float = 0.0
    """当日等效循环次数。"""

    peak_grid_kw: float = 0.0
    """并网点最大功率（用于需量电费测算）。"""

    baseline_peak_grid_kw: float = 0.0
    """不配储能时的并网点最大功率。"""

    @property
    def savings_yuan(self) -> float:
        return self.baseline_cost_yuan - self.optimized_cost_yuan

    @property
    def savings_pct(self) -> float:
        if self.baseline_cost_yuan <= 0:
            return 0.0
        return self.savings_yuan / self.baseline_cost_yuan * 100

    @property
    def peak_reduction_kw(self) -> float:
        """削峰量——需量电费就是靠这个省的。"""
        return self.baseline_peak_grid_kw - self.peak_grid_kw

    def soc_series(self) -> np.ndarray:
        return np.array([e.soc_eod for e in self.entries])

    def power_series(self) -> np.ndarray:
        return np.array([e.power_kw for e in self.entries])

    def summary(self) -> str:
        lines = [
            f"时段数 {len(self.entries)}",
            f"电费  基线 {self.baseline_cost_yuan:>10,.1f} 元"
            f"   优化后 {self.optimized_cost_yuan:>10,.1f} 元"
            f"   节省 {self.savings_yuan:>9,.1f} 元 ({self.savings_pct:+.1f}%)",
            f"充放  充 {self.charged_kwh:,.1f} kWh  放 {self.discharged_kwh:,.1f} kWh"
            f"  等效循环 {self.cycles_used:.2f} 次",
            f"削峰  并网峰值 {self.baseline_peak_grid_kw:,.1f} → "
            f"{self.peak_grid_kw:,.1f} kW  (降 {self.peak_reduction_kw:,.1f} kW)",
        ]
        return "\n".join(lines)
