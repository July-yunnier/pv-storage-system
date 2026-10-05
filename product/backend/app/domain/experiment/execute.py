"""计划执行仿真：把 96 点计划动作放到实测光伏/负荷上回放。

计划是在预测曲线上算出来的；执行时面对的是真实（回放）曲线——
预测误差就在这里传导为 SOC 偏差和电费差异。
动作会被物理约束裁剪（功率上限、SOC 边界），不允许反送电时并网功率下限为 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.models import PricePoint, StorageConfig


@dataclass(frozen=True)
class ExecutionResult:
    grid_import_kwh: float
    cost_yuan: float
    peak_import_kw: float
    pv_self_use_kwh: float
    pv_total_kwh: float
    curtailment_kwh: float
    discharge_kwh: float
    grid_charge_kwh: float
    end_soc: float
    soc_violations: int
    soc_trace: list[float] = field(default_factory=list)


def execute_plan(
    actions: list[tuple[str, float]],
    actual_pv_kw: list[float],
    actual_load_kw: list[float],
    price_points: list[PricePoint],
    storage: StorageConfig,
    initial_soc: float,
    resolution_minutes: int = 15,
) -> ExecutionResult:
    """actions: 每槽 (action, power_kw)，action ∈ charge/discharge/idle。"""
    dt_hours = resolution_minutes / 60
    soc_energy = initial_soc * storage.capacity_kwh
    min_energy = storage.soc_min * storage.capacity_kwh
    max_energy = storage.soc_max * storage.capacity_kwh
    charge_eff = storage.charge_eff
    discharge_eff = storage.discharge_eff

    cost = 0.0
    grid_import_kwh = 0.0
    peak_import = 0.0
    pv_self_use = 0.0
    pv_total = 0.0
    curtailment = 0.0
    discharged = 0.0
    grid_charged = 0.0
    violations = 0
    trace: list[float] = []

    for (action, planned_kw), pv_kw, load_kw, price in zip(
        actions, actual_pv_kw, actual_load_kw, price_points
    ):
        headroom_kwh = max(max_energy - soc_energy, 0.0)
        available_kwh = max(soc_energy - min_energy, 0.0)

        charge_kw = discharge_kw = 0.0
        if action == "charge" and planned_kw > 0:
            charge_kw = min(planned_kw, storage.pcs_kw, headroom_kwh / charge_eff / dt_hours)
        elif action == "discharge" and planned_kw > 0:
            discharge_kw = min(planned_kw, storage.pcs_kw, available_kwh * discharge_eff / dt_hours)
            if not storage.allow_grid_export:
                # 防逆流：放电不超过净负荷
                discharge_kw = min(discharge_kw, max(load_kw - pv_kw, 0.0))

        soc_energy += charge_kw * dt_hours * charge_eff
        soc_energy -= discharge_kw * dt_hours / discharge_eff
        if soc_energy < min_energy - 1e-6 or soc_energy > max_energy + 1e-6:
            violations += 1
        soc_energy = min(max(soc_energy, min_energy), max_energy)

        net = load_kw - pv_kw
        grid_import = max(net + charge_kw - discharge_kw, 0.0)
        export = max(-(net + charge_kw - discharge_kw), 0.0)
        if not storage.allow_grid_export:
            curtailment += export * dt_hours

        cost += grid_import * dt_hours * price.price
        grid_import_kwh += grid_import * dt_hours
        peak_import = max(peak_import, grid_import)
        pv_total += max(pv_kw, 0.0) * dt_hours
        pv_self_use += max(min(pv_kw, load_kw + charge_kw), 0.0) * dt_hours
        discharged += discharge_kw * dt_hours
        if charge_kw > 0 and net >= 0:
            grid_charged += charge_kw * dt_hours
        trace.append(round(soc_energy / storage.capacity_kwh, 6))

    return ExecutionResult(
        grid_import_kwh=round(grid_import_kwh, 4),
        cost_yuan=round(cost, 4),
        peak_import_kw=round(peak_import, 4),
        pv_self_use_kwh=round(pv_self_use, 4),
        pv_total_kwh=round(pv_total, 4),
        curtailment_kwh=round(curtailment, 4),
        discharge_kwh=round(discharged, 4),
        grid_charge_kwh=round(grid_charged, 4),
        end_soc=round(soc_energy / storage.capacity_kwh, 6),
        soc_violations=violations,
        soc_trace=trace,
    )


def no_storage_baseline(
    actual_pv_kw: list[float],
    actual_load_kw: list[float],
    price_points: list[PricePoint],
    resolution_minutes: int = 15,
) -> ExecutionResult:
    """D0：不配储能的费用基准。"""
    idle = [("idle", 0.0)] * len(actual_pv_kw)
    zero_storage = StorageConfig(pcs_kw=1e-9, capacity_kwh=1e-9, soc_min=0.0, soc_max=1.0)
    return execute_plan(idle, actual_pv_kw, actual_load_kw, price_points, zero_storage, 0.0, resolution_minutes)
