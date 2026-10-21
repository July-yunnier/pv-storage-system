"""区间感知调度：把 P10/P90 真正用进决策，而不是只画在图上。

原有 `RuleDispatchEngine` 只看分时电价的"时段标签"（谷/峰/尖），本质是一个
确定性规则——它默认"预测的价格就是会发生的价格"。一旦换成现货或代理购电，
价格带会张开，按 P50 决策等于**用中位数赌尾部**：

- 充电时按 P50 算便宜，实际结算可能落在 P90 —— 成本比预期高一截；
- 放电时按 P50 算值钱，实际可能落在 P10 —— 收益比预期低一截；
- 更糟的是：`P50` 上看着有价差、`P10/P90` 上其实没有，这种"假套利"会白跑循环、
  磨损电池，还把 SOC 用光，等到真正的尖峰反而放不出来（缺电风险）。

本引擎的约束是三条**区间约束**：

1. **充电按 P90 计价**：只有「最坏情况下也够便宜」的时段才充；
2. **放电按 P10 计价**：只有「最坏情况下也够值钱」的时段才放；
3. **鲁棒价差门槛**：`P10_top × η − P90_low / η ≥ margin` 才允许这一轮循环；
   通不过就不做——不做也是一种决策，这一点必须写进调度记录。

另外保留 **SOC 保底**：放电一般只放到 `soc_min + reserve_ratio`，除非 P10 超过
当日放电阈值的 `reserve_override` 倍（真正的尖峰才允许打穿保底）。

边界：这是带约束的贪心，不是数学规划。它不能证明全局最优，但每一步决策都能
用一句话解释（"这个时段即使按最坏价格算，充电也是便宜的"），
比黑箱 LP 更适合现场解释与审计。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from app.domain.models import (
    BenefitBreakdown,
    DispatchPoint,
    MeasurementRead,
    ScheduleResponse,
    SiteRead,
)


@dataclass(frozen=True)
class PriceBand:
    """单个时段的电价三点的。tier 可空（现货市场没有分时标签）。"""

    ts: datetime
    p10: float
    p50: float
    p90: float
    tier: str | None = None


@dataclass(frozen=True)
class IntervalInputs:
    site: SiteRead
    target_date: date
    initial_soc: float
    pv_points: list[MeasurementRead]
    load_points: list[MeasurementRead]
    bands: list[PriceBand]
    resolution_minutes: int = 15
    margin_ratio: float = 0.02       # 鲁棒净价差门槛，按当日 P50 均值的比例（跨价格量级可比）
    margin_robust: float = 0.0       # 兼容旧调用：绝对门槛（元/kWh），与 margin_ratio 取较大者
    reserve_ratio: float = 0.10      # 在 soc_min 之上额外保留的比例
    charge_quantile: float = 0.15    # 最便宜的 15% 时段才算"够便宜"
    discharge_quantile: float = 0.85 # 最贵的 15% 时段才算"够值钱"
    reserve_override: float = 1.5    # P10 超过阈值多少倍时允许打穿保底


def _quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


class IntervalDispatchEngine:
    """用区间做决策的调度引擎。"""

    name = "interval_robust"

    def run(self, inputs: IntervalInputs) -> ScheduleResponse:
        storage = inputs.site.storage
        n = len(inputs.pv_points)
        if n == 0 or len(inputs.load_points) != n or len(inputs.bands) != n:
            raise ValueError("pv / load / price band series must be non-empty and aligned")

        dt_hours = inputs.resolution_minutes / 60
        capacity = storage.capacity_kwh
        soc_energy = inputs.initial_soc * capacity
        min_energy = storage.soc_min * capacity
        max_energy = storage.soc_max * capacity
        reserve_energy = min_energy + inputs.reserve_ratio * capacity
        usable = (storage.soc_max - storage.soc_min) * capacity
        max_discharge_energy = storage.cycles_per_day_max * usable

        p10 = np.asarray([b.p10 for b in inputs.bands], dtype="float64")
        p50 = np.asarray([b.p50 for b in inputs.bands], dtype="float64")
        p90 = np.asarray([b.p90 for b in inputs.bands], dtype="float64")

        # 当日鲁棒极值：充电看 P90 的低位，放电看 P10 的高位
        charge_cut = _quantile(p90, inputs.charge_quantile)
        discharge_cut = _quantile(p10, inputs.discharge_quantile)
        eff = storage.round_trip_eff
        robust_spread = discharge_cut * eff - charge_cut / eff
        price_level = float(np.nanmean(p50)) if np.isfinite(p50).any() else 0.0
        margin_abs = max(inputs.margin_robust, inputs.margin_ratio * max(price_level, 1e-9))
        gate_open = bool(np.isfinite(robust_spread) and robust_spread >= margin_abs)

        points: list[DispatchPoint] = []
        baseline_cost = 0.0
        optimized_cost = 0.0
        discharged_energy = 0.0
        discharge_value = 0.0
        grid_charge_cost = 0.0
        grid_charge_energy = 0.0
        surplus_charge_energy = 0.0
        reserve_shortfall = 0.0
        blocked_slots = 0

        for i, (pv, load, band) in enumerate(zip(inputs.pv_points, inputs.load_points, inputs.bands)):
            if pv.ts != load.ts or pv.ts != band.ts:
                raise ValueError("pv / load / price timestamps must align")
            if pv.value is None or load.value is None:
                raise ValueError(f"存在缺失点，无法生成调度: {pv.ts.isoformat()}")
            pv_kw = float(pv.value)
            load_kw = float(load.value)
            net = load_kw - pv_kw
            baseline_cost += max(net, 0.0) * dt_hours * float(band.p50)

            charge_kw = 0.0
            discharge_kw = 0.0
            curtailment_kw = 0.0
            headroom = max(max_energy - soc_energy, 0.0)
            available = max(soc_energy - min_energy, 0.0)

            surplus_accounted = False
            if net < 0:
                # 光伏余电充电：近乎零成本，任何策略都该先吃掉，不参与门槛判断
                surplus = -net
                charge_kw = min(storage.pcs_kw, surplus, headroom / storage.charge_eff / dt_hours)
                curtailment_kw = max(surplus - charge_kw, 0.0)
                surplus_accounted = True
            elif gate_open and p90[i] <= charge_cut:
                charge_kw = min(storage.pcs_kw, headroom / storage.charge_eff / dt_hours)
                limit = inputs.site.demand_limit_kw
                if limit is not None:
                    charge_kw = min(charge_kw, max(limit - net, 0.0))
                if charge_kw <= 0:
                    charge_kw = 0.0
            elif gate_open and p10[i] >= discharge_cut and net > 0:
                # 保底：一般只放到 reserve_energy，尖峰才允许打穿
                floor = min_energy if p10[i] >= discharge_cut * inputs.reserve_override else reserve_energy
                available_above_floor = max(soc_energy - floor, 0.0)
                discharge_kw = min(
                    storage.pcs_kw,
                    net,
                    available_above_floor * storage.discharge_eff / dt_hours,
                    max(max_discharge_energy - discharged_energy, 0.0) / dt_hours,
                )
            elif net > 0 and soc_energy <= reserve_energy + 1e-9:
                # 记录"想放但没电"的时段，作为缺电风险的直接度量
                if p10[i] >= discharge_cut:
                    blocked_slots += 1

            if charge_kw > 0 and discharge_kw > 0:  # pragma: no cover - 不变量
                raise RuntimeError("a dispatch point cannot charge and discharge together")

            soc_start = soc_energy / capacity
            soc_energy += charge_kw * dt_hours * storage.charge_eff
            soc_energy -= discharge_kw * dt_hours / storage.discharge_eff
            soc_energy = min(max(soc_energy, min_energy), max_energy)
            soc_end = soc_energy / capacity

            grid_import = max(net + charge_kw - discharge_kw, 0.0)
            grid_export = max(-(net + charge_kw - discharge_kw), 0.0)
            if not storage.allow_grid_export:
                # 余电未被电池吸收的部分上面已计入 curtailment，这里避免重复计数
                if not surplus_accounted:
                    curtailment_kw += grid_export
                grid_export = 0.0

            # 结算用真实价格（回放里由调用方传入实际价；在线用 P50 作无偏估计）
            settle_price = float(band.p50)
            optimized_cost += grid_import * dt_hours * settle_price
            discharged_energy += discharge_kw * dt_hours
            discharge_value += discharge_kw * dt_hours * settle_price

            charge_pv = 0.0
            charge_grid = 0.0
            if charge_kw > 0:
                if net < 0:
                    charge_pv = charge_kw
                    surplus_charge_energy += charge_kw * dt_hours
                else:
                    charge_grid = charge_kw
                    grid_charge_energy += charge_kw * dt_hours
                    grid_charge_cost += charge_kw * dt_hours * float(band.p90)

            # 保底违约电量：SOC 低于 reserve 且正在用高价电的时段
            if net > 0 and p10[i] >= discharge_cut:
                reserve_shortfall += max(reserve_energy - soc_energy, 0.0) * dt_hours

            action = "idle"
            power = 0.0
            if charge_kw > 0:
                action, power = "charge", charge_kw
            elif discharge_kw > 0:
                action, power = "discharge", discharge_kw

            points.append(
                DispatchPoint(
                    ts=pv.ts,
                    action=action,
                    power_kw=round(power, 6),
                    soc_start=round(soc_start, 6),
                    soc_end=round(soc_end, 6),
                    pv_power_kw=round(pv_kw, 6),
                    load_power_kw=round(load_kw, 6),
                    price=round(float(band.p50), 6),
                    tier=band.tier,
                    grid_import_kw=round(grid_import, 6),
                    grid_export_kw=round(grid_export, 6),
                    pv_curtailment_kw=round(curtailment_kw, 6),
                    charge_from_pv_kw=round(charge_pv, 6),
                    charge_from_grid_kw=round(charge_grid, 6),
                )
            )

        cycles = discharged_energy / usable if usable > 0 else 0.0
        constraints = [
            f"soc_range[{storage.soc_min},{storage.soc_max}]",
            f"pcs_limit_kw={storage.pcs_kw}",
            f"max_cycles_per_day={storage.cycles_per_day_max}",
            f"robust_spread_gate={margin_abs:.4f}",
            f"margin_ratio={inputs.margin_ratio:.3f}",
            f"charge_cut_p90={charge_cut:.4f}",
            f"discharge_cut_p10={discharge_cut:.4f}",
            f"reserve_ratio={inputs.reserve_ratio:.2f}",
            f"reserve_override={inputs.reserve_override:.2f}",
            f"gate_open={gate_open}",
            f"blocked_slots={blocked_slots}",
        ]
        if inputs.site.demand_limit_kw is not None:
            constraints.append(f"demand_limit_kw={inputs.site.demand_limit_kw}")

        return ScheduleResponse(
            site_id=inputs.site.site_id,
            date=inputs.target_date,
            strategy=self.name,
            expected_benefit_yuan=round(baseline_cost - optimized_cost, 6),
            baseline_cost_yuan=round(baseline_cost, 6),
            optimized_cost_yuan=round(optimized_cost, 6),
            equivalent_full_cycles=round(cycles, 6),
            schedule=points,
            constraints_applied=constraints,
            benefit_breakdown=BenefitBreakdown(
                discharge_value_yuan=round(discharge_value, 6),
                grid_charge_cost_yuan=round(grid_charge_cost, 6),
                discharge_kwh=round(discharged_energy, 6),
                grid_charge_kwh=round(grid_charge_energy, 6),
                surplus_charge_kwh=round(surplus_charge_energy, 6),
            ),
        )


def robust_gate(
    p10: np.ndarray,
    p90: np.ndarray,
    round_trip_eff: float,
    margin: float,
    charge_quantile: float = 0.15,
    discharge_quantile: float = 0.85,
    price_level: float | None = None,
) -> dict:
    """只看区间就判断"今天值不值得做一轮循环"。

    返回 gate_open / robust_spread / 充放阈值，供调度记录与前端提示复用。
    """
    charge_cut = _quantile(p90, charge_quantile)
    discharge_cut = _quantile(p10, discharge_quantile)
    spread = discharge_cut * round_trip_eff - charge_cut / round_trip_eff
    return {
        "gate_open": bool(np.isfinite(spread) and spread >= margin),
        "robust_spread": round(float(spread), 6),
        "charge_cut": round(float(charge_cut), 6),
        "discharge_cut": round(float(discharge_cut), 6),
        "margin": margin,
    }
