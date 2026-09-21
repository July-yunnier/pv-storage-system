"""预测结果模型。

设计原则：**精度必须带口径**。

同一条预测曲线，只改变评估子集，指标就能差好几个百分点。例如某光伏预测
在含夜间样本时是 95.06%，剔除夜间的 51% 样本后只有 92.95%——同一份预测，
同一个模型。

所以本系统所有预测结果都把口径随结果一起返回，不接受"裸精度数字"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

ForecastTarget = Literal["pv_power", "load_power", "price"]


@dataclass
class AccuracyCaliber:
    """精度口径。缺任何一项，精度数字都无法跨方案比较。"""

    normalization: str
    """归一化基准：rated_capacity（额定容量）/ installed_capacity / mean_power。"""

    cap_kw: float
    """归一化所用的分母值（kW）。"""

    includes_nighttime: bool
    """评估样本是否包含夜间。这一项对光伏指标影响最大。"""

    spatial_scale: str = "single_site"
    """空间尺度：single_site（单站点）/ regional（区域聚合）。"""

    baseline: str = "persistence"
    """基准对照：persistence / diurnal_persistence / none。无基准的精度说服力弱。"""

    def describe(self) -> str:
        night = "含夜间" if self.includes_nighttime else "纯白天"
        scale = "单站" if self.spatial_scale == "single_site" else "区域"
        return (f"1-RMSE/{self.normalization}={self.cap_kw:.1f}kW · "
                f"{night} · {scale} · 基准对照={self.baseline}")


@dataclass
class ForecastResult:
    """一条预测曲线及其元信息。

    values / p10 / p90 的长度必须一致，时间轴由 issued_at + resolution 推导。
    """

    target: ForecastTarget
    unit: str
    issued_at: datetime
    resolution_minutes: int
    values: np.ndarray
    p10: np.ndarray | None = None
    p90: np.ndarray | None = None
    model_name: str = ""
    accuracy: AccuracyCaliber | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        n = len(self.values)
        for name, arr in (("p10", self.p10), ("p90", self.p90)):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} 长度 {len(arr)} 与 values 长度 {n} 不一致")

    @property
    def horizon_hours(self) -> float:
        return len(self.values) * self.resolution_minutes / 60.0

    @property
    def has_uncertainty(self) -> bool:
        return self.p10 is not None and self.p90 is not None

    def summary(self) -> str:
        parts = [
            f"{self.target}  {self.unit}",
            f"{len(self.values)} 点 / {self.horizon_hours:.1f} h "
            f"@ {self.resolution_minutes} min",
            f"范围 {float(np.min(self.values)):.2f} ~ {float(np.max(self.values)):.2f}",
        ]
        if self.model_name:
            parts.append(f"模型 {self.model_name}")
        if self.has_uncertainty:
            parts.append("含 p10/p90 区间")
        line = " | ".join(parts)
        if self.accuracy:
            line += f"\n    口径：{self.accuracy.describe()}"
        return line


def make_accuracy(
    cap_kw: float,
    includes_nighttime: bool,
    normalization: str = "rated_capacity",
    spatial_scale: str = "single_site",
    baseline: str = "persistence",
) -> AccuracyCaliber:
    """构造口径信息的便捷函数。"""
    return AccuracyCaliber(
        normalization=normalization,
        cap_kw=cap_kw,
        includes_nighttime=includes_nighttime,
        spatial_scale=spatial_scale,
        baseline=baseline,
    )
