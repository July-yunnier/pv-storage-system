"""峰谷分时电价 Provider。

数据来源：省级目录电价文件（config/tariff/<province>.yaml）
特点：电价由政策给定，**不需要任何预测模型**。
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .base import PricePoint, TariffProvider, Tier

#: 查找优先级：尖峰 > 高峰 > 平段 > 低谷
#: 若某时刻同时落在多个区间（配置写错时），按此顺序取第一个命中
_TIER_ORDER: tuple[Tier, ...] = ("sharp", "peak", "flat", "valley")


def _to_minutes(hhmm: str) -> int:
    """'17:30' -> 1050"""
    hh, mm = hhmm.strip().split(":")
    return int(hh) * 60 + int(mm)


def _parse_ranges(hours: Iterable[str]) -> list[tuple[int, int]]:
    """解析形如 ['13:00-17:00', '19:00-20:00'] 的时段列表。

    支持跨零点（如 '23:00-07:00'），此时返回的 (start, end) 满足 start > end。
    """
    out: list[tuple[int, int]] = []
    for item in hours:
        a, b = item.split("-")
        out.append((_to_minutes(a), _to_minutes(b)))
    return out


def _in_range(minute: int, start: int, end: int) -> bool:
    """判断一天中的第 minute 分钟是否落在 [start, end) 内。

    左闭右开；start == end 视为空区间；
    start > end 表示跨零点。
    """
    if start == end:
        return False
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


class TouProvider(TariffProvider):
    """峰谷分时电价（目录电价 + 固定输配与基金）。"""

    provides_uncertainty = False
    requires_price_forecast = False

    def __init__(
        self,
        tou_table: dict[str, dict],
        transmission_and_fund: float = 0.0,
        seasonal_overrides: dict | None = None,
        province: str = "",
    ) -> None:
        """
        参数
        ----
        tou_table
            形如 {"peak": {"price": 0.95, "hours": ["13:00-17:00"]}, ...}
        transmission_and_fund
            输配电价 + 政府性基金及附加（元/kWh），固定部分
        seasonal_overrides
            季节性时段调整，形如 {"summer": {"months": [7, 8], "sharp": {"hours": [...]}}}
        """
        self.province = province
        self.transmission_and_fund = float(transmission_and_fund)
        self.seasonal_overrides = seasonal_overrides or {}

        self._prices: dict[str, float] = {}
        self._ranges: dict[str, list[tuple[int, int]]] = {}
        self._seasonal: dict[str, dict] = {}

        self._load_table(tou_table)

        for name, cfg in self.seasonal_overrides.items():
            months = set(cfg.get("months", []))
            rngs: dict[str, list[tuple[int, int]]] = {}
            for tier in _TIER_ORDER:
                if tier in cfg and "hours" in cfg[tier]:
                    rngs[tier] = _parse_ranges(cfg[tier]["hours"])
            self._seasonal[name] = {"months": months, "ranges": rngs}

    # ------------------------------------------------------------ 内部
    def _load_table(self, tou_table: dict[str, dict]) -> None:
        for tier in _TIER_ORDER:
            if tier not in tou_table:
                continue
            cfg = tou_table[tier]
            self._prices[tier] = float(cfg["price"])
            self._ranges[tier] = _parse_ranges(cfg.get("hours", []))

    def _ranges_for(self, month: int) -> dict[str, list[tuple[int, int]]]:
        """按月份取生效的时段表（季节性优先，缺失的档位回退到基础表）。"""
        for spec in self._seasonal.values():
            if month in spec["months"]:
                merged = dict(self._ranges)
                merged.update(spec["ranges"])
                return merged
        return self._ranges

    def tier_at(self, ts: datetime) -> Tier:
        minute = ts.hour * 60 + ts.minute
        ranges = self._ranges_for(ts.month)
        for tier in _TIER_ORDER:
            for a, b in ranges.get(tier, []):
                if _in_range(minute, a, b):
                    return tier
        return "flat"   # 配置未覆盖的时刻，兜底为平段

    # ------------------------------------------------------------ 接口
    def price_at(self, ts: datetime) -> PricePoint:
        tier = self.tier_at(ts)
        energy = self._prices.get(tier, 0.0)
        return PricePoint(
            ts=ts,
            energy_price=energy,
            price=energy + self.transmission_and_fund,
            tier=tier,
        )

    def describe(self) -> dict:
        info = super().describe()
        info.update({
            "province": self.province,
            "transmission_and_fund": self.transmission_and_fund,
            "prices": dict(self._prices),
            "note": "分时电价由政策给定，无需价格预测",
        })
        return info

    # ------------------------------------------------------------ 校验
    def validate_coverage(self, resolution_minutes: int = 15) -> list[int]:
        """检查一天 24 小时是否被完整覆盖，返回未被覆盖的分钟列表。

        配置漏掉时段是最常见的错误——它不会报错，只会静默地按平段计价，
        导致收益测算偏差。建议启动时跑一次。
        """
        uncovered: list[int] = []
        for minute in range(0, 24 * 60, resolution_minutes):
            hit = False
            for tier in _TIER_ORDER:
                for a, b in self._ranges.get(tier, []):
                    if _in_range(minute, a, b):
                        hit = True
                        break
                if hit:
                    break
            if not hit:
                uncovered.append(minute)
        return uncovered


def load_tou_from_yaml(path: str, voltage_level: str | None = None) -> TouProvider:
    """从省级电价 YAML 载入。

    文件结构见 config/tariff/jiangsu.yaml。

    voltage_level
        用于从 transmission_and_fund 表中取对应电压等级的固定加价。
        若为 None，取表中第一个非 note 项。
    """
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 输配电价按电压等级取；表中可能混有 note 之类的说明键
    tf_cfg = cfg.get("transmission_and_fund", {})
    if isinstance(tf_cfg, dict):
        candidates = {k: v for k, v in tf_cfg.items()
                      if isinstance(v, (int, float))}
        if voltage_level and voltage_level in candidates:
            tf = float(candidates[voltage_level])
        elif candidates:
            tf = float(next(iter(candidates.values())))
        else:
            tf = 0.0
    else:
        tf = float(tf_cfg or 0.0)

    return TouProvider(
        tou_table=cfg["energy_price"],
        transmission_and_fund=tf,
        seasonal_overrides=cfg.get("seasonal_overrides"),
        province=cfg.get("province", ""),
    )
