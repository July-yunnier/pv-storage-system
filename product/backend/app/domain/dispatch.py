from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite

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
    # Optional day-ahead/intraday forecasts used by the look-ahead controller.
    # When omitted, the legacy tariff/tier rule is retained exactly.
    forecast_load_points: list[MeasurementRead] | None = None
    forecast_price_points: list[PricePoint] | None = None
    forecast_pv_points: list[MeasurementRead] | None = None
    reserve_soc: float | None = None
    peak_shaving_target_kw: float | None = None
    lookahead_hours: int = 4


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
        if inputs.lookahead_hours <= 0:
            raise ValueError("lookahead_hours must be positive")
        if inputs.resolution_minutes <= 0 or not storage.soc_min <= inputs.initial_soc <= storage.soc_max:
            raise ValueError("resolution must be positive and initial_soc within storage bounds")
        for series in (inputs.pv_points, inputs.load_points, inputs.forecast_load_points, inputs.forecast_pv_points):
            if series is None:
                continue
            if any(p.value is None or not isfinite(p.value) or p.value < 0 for p in series):
                raise ValueError("存在缺失点或非法功率，无法生成调度")
        for series in (inputs.price_points, inputs.forecast_price_points):
            if series is not None and any(not isfinite(p.price) for p in series):
                raise ValueError("price must be finite")
        if inputs.forecast_pv_points is not None:
            if len(inputs.forecast_pv_points) != len(inputs.pv_points) or any(p.ts != q.ts for p, q in zip(inputs.forecast_pv_points, inputs.pv_points)):
                raise ValueError("forecast pv timestamps and length must align")
        if inputs.forecast_load_points is not None:
            if len(inputs.forecast_load_points) != len(inputs.load_points):
                raise ValueError("forecast load series must have the same length")
            if any(p.ts != q.ts for p, q in zip(inputs.forecast_load_points, inputs.load_points)):
                raise ValueError("forecast load timestamps must align")
        if inputs.forecast_price_points is not None:
            if len(inputs.forecast_price_points) != len(inputs.price_points):
                raise ValueError("forecast price series must have the same length")
            if any(p.ts != q.ts for p, q in zip(inputs.forecast_price_points, inputs.price_points)):
                raise ValueError("forecast price timestamps must align")
        if inputs.reserve_soc is not None and not storage.soc_min <= inputs.reserve_soc <= storage.soc_max:
            raise ValueError("reserve_soc must be within storage bounds")

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
        lookahead_steps = max(1, int(inputs.lookahead_hours * 60 / inputs.resolution_minutes))
        predictive_mode = inputs.forecast_load_points is not None or inputs.forecast_price_points is not None
        target_kw = inputs.peak_shaving_target_kw
        if target_kw is None:
            target_kw = demand_limit_kw
        if target_kw is not None and target_kw <= 0:
            raise ValueError("peak_shaving_target_kw must be positive")
        reserve_floor_energy = (
            max(min_energy, inputs.reserve_soc * storage.capacity_kwh)
            if inputs.reserve_soc is not None
            else min_energy
        )
        used_forecast_load = False
        used_forecast_price = False
        active_peak_shaving = False
        reserved_energy_events = 0
        unmet_shaving_slots = 0

        for index, (pv, load, price) in enumerate(zip(inputs.pv_points, inputs.load_points, inputs.price_points)):
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

            # Forecast-aware reserve: preserve enough energy for a later
            # predicted peak instead of emptying the battery in an earlier
            # flat/peak interval.  This is deliberately based only on supplied
            # forecast arrays; the default path has no look-ahead leakage.
            future_end = min(len(inputs.load_points), index + lookahead_steps + 1)
            future_indices = range(index + 1, future_end)
            future_load = inputs.forecast_load_points if inputs.forecast_load_points is not None else None
            future_prices = inputs.forecast_price_points if inputs.forecast_price_points is not None else None
            if future_load is not None:
                used_forecast_load = True
            if future_prices is not None:
                used_forecast_price = True
            future_net = []
            for j in future_indices:
                f_load = future_load[j].value if future_load is not None else None
                if f_load is not None:
                    # Never use future measured PV to decide a current action.
                    f_pv = inputs.forecast_pv_points[j].value if inputs.forecast_pv_points is not None else 0.0
                    if f_pv is not None:
                        future_net.append(max(f_load - f_pv, 0.0))
            predicted_peak = max(future_net, default=0.0)
            future_reserve = reserve_floor_energy
            if target_kw is not None and predicted_peak > target_kw:
                # Reserve PCS-limited excess over all higher future peaks,
                # capped by usable battery energy.
                future_reserve += min(usable_capacity, sum(
                    min(storage.pcs_kw, max(value-target_kw, 0.0))*dt_hours/storage.discharge_eff
                    for value in future_net if value > net_load
                ))
                future_reserve = min(max_energy, future_reserve)
                reserved_energy_events += 1
            elif future_prices is not None:
                # Preserve discharge energy for more expensive future periods,
                # even if no explicit demand target was configured.
                current_forecast_price = future_prices[index].price
                later_expensive = [j for j in future_indices if future_prices[j].price > current_forecast_price + 1e-9]
                if later_expensive:
                    reserve_for_price = sum(
                        min(storage.pcs_kw, max(future_load[j].value - (inputs.forecast_pv_points[j].value if inputs.forecast_pv_points is not None else 0), 0))
                        * dt_hours / storage.discharge_eff for j in later_expensive
                    ) if future_load is not None else 0.0
                    future_reserve = min(max_energy, future_reserve + reserve_for_price)
                    if reserve_for_price > 0:
                        reserved_energy_events += 1
            available_after_reserve_kwh = max(soc_energy - future_reserve, 0.0)

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
                    max(available_after_reserve_kwh, 0.0) * storage.discharge_eff / dt_hours
                    if predictive_mode else available_kwh * storage.discharge_eff / dt_hours,
                    max(max_discharge_energy - discharged_energy, 0.0) / dt_hours,
                )

            if predictive_mode and net_load > 0:
                # AI path replaces the current-tier rule. A price spread must
                # cover round-trip losses; otherwise charging loses money.
                charge_kw = 0.0
                discharge_kw = 0.0
                current_price = future_prices[index].price if future_prices is not None else price.price
                future_max_price = max((future_prices[j].price for j in future_indices), default=current_price) if future_prices is not None else current_price
                future_min_price = min((future_prices[j].price for j in future_indices), default=current_price) if future_prices is not None else current_price
                # Active demand shaving is allowed in flat/valley periods only
                # when a forecast says the current import exceeds the target.
                if target_kw is not None and net_load > target_kw:
                    active_peak_shaving = True
                    # A shaving action has priority over valley/arbitrage
                    # charging at the same interval; never charge and
                    # discharge simultaneously.
                    charge_kw = 0.0
                    discharge_kw = min(
                        storage.pcs_kw,
                        net_load - target_kw,
                        max(available_after_reserve_kwh, 0.0) * storage.discharge_eff / dt_hours,
                        max(max_discharge_energy - discharged_energy, 0.0) / dt_hours,
                    )
                # Price arbitrage: charge from grid when the forecast price is
                # below a later forecast price, while respecting demand limit.
                if (
                    discharge_kw == 0
                    and (target_kw is None or net_load < target_kw)
                    and future_prices is not None
                    and future_max_price * storage.round_trip_eff > current_price + 1e-9
                    and headroom_kwh > 0
                ):
                    charge_kw = min(storage.pcs_kw, headroom_kwh / storage.charge_eff / dt_hours)
                    if demand_limit_kw is not None:
                        charge_kw = min(charge_kw, max(demand_limit_kw - net_load, 0.0))
                    if target_kw is not None:
                        charge_kw = min(charge_kw, max(target_kw-net_load, 0.0))
                elif discharge_kw == 0 and charge_kw == 0 and (
                    (future_prices is not None and current_price >= future_max_price and current_price > future_min_price/storage.round_trip_eff)
                    or (future_prices is None and tier in {"peak", "sharp"})
                ):
                    discharge_kw = min(storage.pcs_kw, net_load,
                                       available_after_reserve_kwh*storage.discharge_eff/dt_hours,
                                       max(max_discharge_energy-discharged_energy, 0.0)/dt_hours)

            if charge_kw > 0 and discharge_kw > 0:
                # 引擎内部不变量被破坏属于程序 bug，不是业务输入错误
                raise RuntimeError("a dispatch point cannot charge and discharge together")

            soc_start = soc_energy / storage.capacity_kwh
            soc_energy += charge_kw * dt_hours * storage.charge_eff
            soc_energy -= discharge_kw * dt_hours / storage.discharge_eff
            soc_energy = min(max(soc_energy, min_energy), max_energy)
            soc_end = soc_energy / storage.capacity_kwh

            grid_import = max(net_load + charge_kw - discharge_kw, 0.0)
            if predictive_mode and target_kw is not None and grid_import > target_kw + 1e-6:
                unmet_shaving_slots += 1
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
        if predictive_mode:
            constraints.append("forecast_lookahead_enabled=true")
            constraints.append(f"lookahead_hours={inputs.lookahead_hours}")
        if used_forecast_load:
            constraints.append("forecast_load_used=true")
        if used_forecast_price:
            constraints.append("forecast_price_used=true")
        if target_kw is not None and predictive_mode:
            constraints.append(f"peak_shaving_target_kw={target_kw}")
        if inputs.reserve_soc is not None:
            constraints.append(f"reserve_soc={inputs.reserve_soc}")
        if active_peak_shaving:
            constraints.append("active_peak_shaving_triggered=true")
        if reserved_energy_events:
            constraints.append(f"future_peak_reserve_events={reserved_energy_events}")
        if unmet_shaving_slots:
            constraints.append(f"unmet_peak_shaving_slots={unmet_shaving_slots}")
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
