from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.domain.models import (
    BenefitBreakdown,
    DispatchPoint,
    MeasurementRead,
    PricePoint,
    ScheduleResponse,
    SiteRead,
)


@dataclass(frozen=True)
class DispatchInputs:
    site: SiteRead
    target_date: date
    initial_soc: float
    pv_points: list[MeasurementRead]
    load_points: list[MeasurementRead]
    price_points: list[PricePoint]
    resolution_minutes: int = 15


class RuleDispatchEngine:
    """MVP 规则式调度：谷段/光伏余电充电，高峰/尖峰放电。"""

    def run(self, inputs: DispatchInputs) -> ScheduleResponse:
        storage = inputs.site.storage
        if not inputs.pv_points or not inputs.load_points:
            raise ValueError("pv_power and load_power measurements are required")
        if len(inputs.pv_points) != len(inputs.load_points):
            raise ValueError("pv_power and load_power series must have the same length")
        if len(inputs.price_points) != len(inputs.pv_points):
            raise ValueError("price and measurement series must have the same length")

        dt_hours = inputs.resolution_minutes / 60
        soc_energy = inputs.initial_soc * storage.capacity_kwh
        min_energy = storage.soc_min * storage.capacity_kwh
        max_energy = storage.soc_max * storage.capacity_kwh
        usable_capacity = (storage.soc_max - storage.soc_min) * storage.capacity_kwh
        demand_limit_kw = inputs.site.demand_limit_kw
        points: list[DispatchPoint] = []
        baseline_cost = 0.0
        optimized_cost = 0.0
        discharged_energy = 0.0
        discharge_value = 0.0
        grid_charge_cost = 0.0
        grid_charge_energy = 0.0
        surplus_charge_energy = 0.0
        max_discharge_energy = storage.cycles_per_day_max * usable_capacity

        for pv, load, price in zip(inputs.pv_points, inputs.load_points, inputs.price_points):
            if pv.ts != load.ts or pv.ts != price.ts:
                raise ValueError("pv, load and price timestamps must align")
            if pv.value is None or load.value is None:
                raise ValueError(f"存在缺失点，无法生成调度: {pv.ts.isoformat()}")
            pv_kw = pv.value
            load_kw = load.value
            net_load = load_kw - pv_kw
            baseline_grid_import = max(net_load, 0.0)
            baseline_cost += baseline_grid_import * dt_hours * price.price

            charge_kw = 0.0
            discharge_kw = 0.0
            curtailment_kw = 0.0
            tier = price.tier

            headroom_kwh = max(max_energy - soc_energy, 0.0)
            available_kwh = max(soc_energy - min_energy, 0.0)

            if net_load < 0:
                # 光伏余电充电：不经过电网，不受需量限制
                surplus_kw = -net_load
                charge_kw = min(
                    storage.pcs_kw,
                    surplus_kw,
                    headroom_kwh / storage.charge_eff / dt_hours,
                )
                curtailment_kw = max(surplus_kw - charge_kw, 0.0)
            elif tier == "valley":
                charge_kw = min(
                    storage.pcs_kw,
                    headroom_kwh / storage.charge_eff / dt_hours,
                )
                if demand_limit_kw is not None:
                    charge_kw = min(charge_kw, max(demand_limit_kw - net_load, 0.0))
            elif tier in {"peak", "sharp"} and net_load > 0:
                discharge_kw = min(
                    storage.pcs_kw,
                    net_load,
                    available_kwh * storage.discharge_eff / dt_hours,
                    max(max_discharge_energy - discharged_energy, 0.0) / dt_hours,
                )

            if charge_kw > 0 and discharge_kw > 0:
                # 引擎内部不变量被破坏属于程序 bug，不是业务输入错误
                raise RuntimeError("a dispatch point cannot charge and discharge together")

            soc_start = soc_energy / storage.capacity_kwh
            soc_energy += charge_kw * dt_hours * storage.charge_eff
            soc_energy -= discharge_kw * dt_hours / storage.discharge_eff
            soc_energy = min(max(soc_energy, min_energy), max_energy)
            soc_end = soc_energy / storage.capacity_kwh

            grid_import = max(net_load + charge_kw - discharge_kw, 0.0)
            grid_export = max(-(net_load + charge_kw - discharge_kw), 0.0)
            if not storage.allow_grid_export:
                curtailment_kw += grid_export
                grid_export = 0.0

            action = "idle"
            power_kw = 0.0
            if charge_kw > 0:
                action, power_kw = "charge", charge_kw
            elif discharge_kw > 0:
                action, power_kw = "discharge", discharge_kw

            optimized_cost += grid_import * dt_hours * price.price
            discharged_energy += discharge_kw * dt_hours
            discharge_value += discharge_kw * dt_hours * price.price
            if charge_kw > 0:
                if net_load < 0:
                    surplus_charge_energy += charge_kw * dt_hours
                else:
                    grid_charge_energy += charge_kw * dt_hours
                    grid_charge_cost += charge_kw * dt_hours * price.price
            points.append(
                DispatchPoint(
                    ts=pv.ts,
                    action=action,
                    power_kw=round(power_kw, 6),
                    soc_start=round(soc_start, 6),
                    soc_end=round(soc_end, 6),
                    pv_power_kw=round(pv_kw, 6),
                    load_power_kw=round(load_kw, 6),
                    price=price.price,
                    tier=tier,
                    grid_import_kw=round(grid_import, 6),
                    grid_export_kw=round(grid_export, 6),
                    pv_curtailment_kw=round(curtailment_kw, 6),
                )
            )

        cycles = discharged_energy / usable_capacity if usable_capacity > 0 else 0.0
        constraints = [
            f"soc_range[{storage.soc_min},{storage.soc_max}]",
            f"pcs_limit_kw={storage.pcs_kw}",
            f"max_cycles_per_day={storage.cycles_per_day_max}",
            f"grid_export_enabled={storage.allow_grid_export}",
        ]
        if demand_limit_kw is not None:
            constraints.append(f"demand_limit_kw={demand_limit_kw}")
        return ScheduleResponse(
            site_id=inputs.site.site_id,
            date=inputs.target_date,
            strategy="rule_v1",
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
