"""电价 Provider 抽象接口。

设计目标
--------
让上游优化器与前端对「电价模式」完全无感知。
无论底层是目录分时电价、电力现货，还是两者叠加，
下游拿到的永远是同一个 PricePoint 序列。

为什么必须这样做
----------------
中国工商业到户电价的构成是：

    到户电价 = 电能量价格 + 输配电价 + 政府性基金及附加

参与现货市场后，**只有「电能量价格」这一项被现货价替换**，
输配电价和政府性基金仍按原规则执行。

所以「分时 + 现货」不是两个价格相加，而是部分替换。
把这个差异消化在 Provider 内部，优化器和前端一行都不用改。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

Tier = Literal["sharp", "peak", "flat", "valley"]

TIER_LABELS: dict[str, str] = {
    "sharp": "尖峰",
    "peak": "高峰",
    "flat": "平段",
    "valley": "低谷",
}


@dataclass(frozen=True)
class PricePoint:
    """某一时刻的电价，含构成拆解与时段标签。"""

    ts: datetime
    energy_price: float
    """电能量价格（元/kWh）——参与现货时被市场价替代的那部分。"""

    price: float
    """到户电价（元/kWh）= energy_price + 输配电价 + 政府性基金及附加。"""

    tier: Tier | None = None
    """时段标签。现货模式下可为 None，但通常仍保留供 UI 画时段色块。"""

    p10: float | None = None
    """价格区间下界（元/kWh）。仅现货/预测模式有值。"""

    p90: float | None = None
    """价格区间上界（元/kWh）。"""

    @property
    def tier_label(self) -> str:
        return TIER_LABELS.get(self.tier or "", "—")


class TariffProvider(ABC):
    """电价数据源统一接口。

    三种实现：
        TouProvider          峰谷分时（读目录电价表，无需预测）
        SpotProvider         电力现货（接市场接口或价格预测模型）
        CompositeProvider    叠加（现货价替代电能量部分，保留时段标签）
    """

    #: 是否提供价格不确定性区间（现货模式为 True）
    provides_uncertainty: bool = False

    #: 是否需要上游提供电价预测结果
    requires_price_forecast: bool = False

    @abstractmethod
    def price_at(self, ts: datetime) -> PricePoint:
        """取单个时刻的电价。"""

    def price_series(
        self,
        start: datetime,
        end: datetime,
        resolution_minutes: int = 15,
    ) -> list[PricePoint]:
        """取时间区间的电价序列，[start, end)。

        默认按 resolution 逐点调用 price_at；
        若实现内部有更高效的方式（如查表），可覆写。
        """
        if resolution_minutes <= 0:
            raise ValueError("resolution_minutes 必须为正数")
        step = timedelta(minutes=resolution_minutes)
        out: list[PricePoint] = []
        cur = start
        # 用循环计数而非 datetime 累加，避免 DST 等边界问题
        n = int((end - start).total_seconds() // (resolution_minutes * 60))
        for _ in range(n):
            out.append(self.price_at(cur))
            cur = cur + step
        return out

    def describe(self) -> dict:
        """返回该 Provider 的自描述信息，用于接口返回与前端展示。"""
        return {
            "provider": type(self).__name__,
            "provides_uncertainty": self.provides_uncertainty,
            "requires_price_forecast": self.requires_price_forecast,
        }
