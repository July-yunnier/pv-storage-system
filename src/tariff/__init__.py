"""电价模块：统一的电价数据源抽象。

对外只暴露一个工厂函数 :func:`get_provider`，
上层无需知道底层是分时、现货还是叠加。
"""
from __future__ import annotations

from .base import TIER_LABELS, PricePoint, TariffProvider, Tier
from .tou import TouProvider, load_tou_from_yaml

__all__ = [
    "PricePoint",
    "TariffProvider",
    "Tier",
    "TIER_LABELS",
    "TouProvider",
    "load_tou_from_yaml",
    "get_provider",
]


def get_provider(mode: str, **kwargs) -> TariffProvider:
    """按模式构造电价 Provider。

    参数
    ----
    mode : {"tou", "spot", "tou_plus_spot"}
        - ``tou``            峰谷分时，读目录电价表，**无需价格预测**
        - ``spot``           电力现货，需提供价格预测结果
        - ``tou_plus_spot``  现货价替代电能量部分，保留时段标签
    """
    mode = (mode or "tou").strip().lower()

    if mode == "tou":
        table = kwargs.get("tou_table")
        if table is None:
            path = kwargs.get("template_path")
            if path is None:
                raise ValueError("tou 模式需要 tou_table 或 template_path")
            return load_tou_from_yaml(path)
        return TouProvider(
            tou_table=table,
            transmission_and_fund=kwargs.get("transmission_and_fund", 0.0),
            seasonal_overrides=kwargs.get("seasonal_overrides"),
            province=kwargs.get("province", ""),
        )

    if mode in ("spot", "tou_plus_spot"):
        raise NotImplementedError(
            f"电价模式 '{mode}' 尚未实现。\n"
            "实现要点：\n"
            "  1. 新建 src/tariff/spot.py，继承 TariffProvider\n"
            "  2. spot 模式：energy_price 取自市场出清价/预测价，tier 置 None\n"
            "  3. tou_plus_spot 模式：energy_price 取现货价，但 tier 仍按目录表标注\n"
            "  4. 两种模式都要填 p10/p90（来自分位数预测）\n"
            "  5. 到户电价 = energy_price + transmission_and_fund，"
            "切勿把分时价与现货价相加"
        )

    raise ValueError(f"未知的电价模式: {mode!r}，可选 tou / spot / tou_plus_spot")
