"""规则式日调度：谷充峰放。

为什么先做规则式而不是直接上线性规划
------------------------------------
峰谷时段是**已知的**（政策给定），所以"谷时充电、峰时放电"天然接近最优解。
规则式几十行就能出可用的调度表，而且结果可解释——给企业讲方案时，
"我们在凌晨 0.35 元的谷时充电，傍晚 1.25 元的尖峰时放电"比一个 LP 求解器
输出容易理解得多。

等规则式跑通、收益模型验证过了，再升级成 LP（见 core.py）也不迟。
反过来做，会在还没有可信基准的情况下争论优化器参数，得不偿失。

效率处理
--------
储能效率必须**分单程计算**，不能直接乘 round_trip_eff：

    充电：存进电池的直流电量 = 交流电量 × one_way_eff
    放电：送到负荷的交流电量 = 电池直流电量 × one_way_eff

其中 one_way_eff = sqrt(round_trip_eff)。
混用两个口径会让收益高估约 10%。
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from ..models.schedule import DispatchResult, ScheduleEntry
from ..models.site import StorageConfig
from ..tariff.base import PricePoint

#: 触发充电的时段
CHARGE_TIERS = {"valley"}
#: 触发放电的时段
DISCHARGE_TIERS = {"sharp", "peak"}


def rule_based_dispatch(
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    prices: list[PricePoint],
    storage: StorageConfig,
    *,
    resolution_minutes: int = 15,
    allow_grid_charge: bool = True,
    allow_grid_export: bool = False,
    demand_aware: bool = True,
    initial_soc: float | None = None,
) -> DispatchResult:
    """按电价时段执行「谷充峰放」的规则式调度。

    参数
    ----
    load_kw, pv_kw
        长度相同的负荷与光伏出力序列（kW，均为正）。
    prices
        PricePoint 序列，长度需与上面两个一致，且已按时间排序。
    storage
        储能参数。容量固定（这是运行模式，不是规划模式）。
    allow_grid_charge
        是否允许从电网取电充电。**这是峰谷套利的前提**，关掉就只剩光伏余电充电。
    allow_grid_export
        是否允许余电倒送电网。国内工商业多为"自发自用、余电不上网"，
        默认 False，此时放电功率会被限制在净负荷以内。
    demand_aware
        是否启用需量感知。**强烈建议开启**。

        谷时若不加限制地满功率充电，充电负荷会叠加在原有负荷上，
        把并网点最大功率推到比不配储能时更高——省下的电度电费，
        回头会从基本电费（需量电费）里加倍吐出去。

        启用后，充电功率被约束在「不把并网功率推高到基线峰值之上」。
        实测影响：某示例站点满功率充电时峰值从 542 kW 升到 706 kW
        （+164 kW，按月 38 元/kW 计，一个月倒亏 6,232 元基本电费）。
    initial_soc
        初始 SOC，默认取 soc_min。

    返回
    ----
    DispatchResult
        含逐时段动作、SOC 轨迹、电费对比、峰值对比。
    """
    load = np.asarray(load_kw, dtype=float)
    pv = np.asarray(pv_kw, dtype=float)
    n = len(load)

    if len(pv) != n:
        raise ValueError(f"load_kw 长度 {n} 与 pv_kw 长度 {len(pv)} 不一致")
    if len(prices) != n:
        raise ValueError(f"prices 长度 {len(prices)} 与负荷序列长度 {n} 不一致")

    dt_h = resolution_minutes / 60.0
    eta = storage.one_way_eff
    cap = storage.capacity_kwh
    soc_min, soc_max = storage.soc_min, storage.soc_max
    pcs = storage.pcs_kw

    # 需量约束上限：基线（无储能）情形下的并网峰值。
    # 充电不得把瞬时并网功率推到这个值以上。
    if demand_aware:
        demand_cap = float(np.max(np.maximum(0.0, load - pv)))
    else:
        demand_cap = float("inf")

    soc = soc_min if initial_soc is None else float(initial_soc)

    # ---- 逐时段推进 ----
    entries: list[ScheduleEntry] = []
    charged = discharged = 0.0
    cost_base = cost_opt = 0.0
    peak_base = peak_opt = 0.0

    for i in range(n):
        pp = prices[i]
        price = pp.price
        tier = pp.tier
        net = load[i] - pv[i]              # 净负荷：正 = 需从电网买

        # 基线（不配储能）的取电与电费
        grid_base = max(0.0, net)
        cost_base += grid_base * price * dt_h
        peak_base = max(peak_base, grid_base)

        # ---- 决定本时段动作 ----
        power = 0.0                        # 约定：正 = 充电，负 = 放电

        if net < 0:
            # 光伏余电优先充入电池（自发自用），不产生购电成本
            room_kwh = (soc_max - soc) * cap
            max_charge_kw = min(pcs, room_kwh / (eta * dt_h) if dt_h > 0 else 0.0)
            power = min(-net, max_charge_kw)

        elif tier in CHARGE_TIERS and allow_grid_charge and soc < soc_max - 1e-9:
            # 谷时从电网充电
            room_kwh = (soc_max - soc) * cap
            max_charge_kw = room_kwh / (eta * dt_h) if dt_h > 0 else 0.0
            # ⚠️ 需量约束：充电功率不能把瞬时并网功率推高到基线峰值之上。
            #    少了这一条，谷时满功率充电会反而抬高最大需量，
            #    省下的电度电费从基本电费里加倍还回去。
            headroom_kw = max(0.0, demand_cap - net)
            power = min(pcs, max_charge_kw, headroom_kw)

        elif tier in DISCHARGE_TIERS and soc > soc_min + 1e-9:
            # 峰时放电供负荷
            avail_kwh = (soc - soc_min) * cap
            max_discharge_kw = avail_kwh * eta / dt_h if dt_h > 0 else 0.0
            limit = pcs if allow_grid_export else min(pcs, max(0.0, net))
            power = -min(limit, max_discharge_kw)

        # ---- 更新 SOC ----
        if power > 0:
            dc_kwh = power * dt_h * eta
            soc += dc_kwh / cap
            charged += power * dt_h
        elif power < 0:
            ac_kwh = -power * dt_h
            dc_kwh = ac_kwh / eta
            soc -= dc_kwh / cap
            discharged += ac_kwh

        soc = float(np.clip(soc, soc_min, soc_max))

        # 优化后的取电
        grid_opt = net + power            # power>0 充电 → 多买电；power<0 放电 → 少买电
        if not allow_grid_export:
            grid_opt = max(0.0, grid_opt)
        cost_opt += grid_opt * price * dt_h
        peak_opt = max(peak_opt, grid_opt)

        entries.append(ScheduleEntry(
            ts=pp.ts,
            action="charge" if power > 1e-9 else ("discharge" if power < -1e-9 else "idle"),
            power_kw=round(float(power), 3),
            soc_eod=round(soc, 4),
            tier=tier,
            price_yuan_per_kwh=round(price, 4),
        ))

    # 等效循环次数：以额定容量为基准
    cycles = discharged / cap if cap > 0 else 0.0

    return DispatchResult(
        entries=entries,
        baseline_cost_yuan=round(cost_base, 2),
        optimized_cost_yuan=round(cost_opt, 2),
        charged_kwh=round(charged, 2),
        discharged_kwh=round(discharged, 2),
        cycles_used=round(cycles, 3),
        peak_grid_kw=round(peak_opt, 2),
        baseline_peak_grid_kw=round(peak_base, 2),
    )


def make_day_arrays(
    load_profile: np.ndarray,
    pv_profile: np.ndarray,
    prices: list[PricePoint],
) -> tuple[np.ndarray, np.ndarray]:
    """把长度不同的两条曲线对齐到电价序列长度（简单截断/重复）。

    仅用于快速演示。正式流程应由预测模块直接产出对齐的序列。
    """
    n = len(prices)
    if len(load_profile) != n:
        load_profile = np.resize(load_profile, n)
    if len(pv_profile) != n:
        pv_profile = np.resize(pv_profile, n)
    return load_profile.astype(float), pv_profile.astype(float)


def synthetic_day(
    site,
    prices: list[PricePoint],
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一条典型的「光伏 + 两班制负荷」日内曲线，用于无实测数据时演示。

    ⚠️ 这是给 demo 用的合成数据，不要用于任何对外结论。
    """
    n = len(prices)
    h = np.array([p.ts.hour + p.ts.minute / 60.0 for p in prices])
    rng = np.random.RandomState(seed)

    # 光伏：钟形，峰值出现在 13:00 前后
    # ⚠️ 必须先把 sin 截断到非负再做分数次幂——np.where 会先算完两个分支，
    #    直接对负的 sin 求 1.35 次方会得到 NaN 并触发 RuntimeWarning。
    shape = np.clip(np.sin(np.pi * (h - 6) / 12), 0.0, None)
    pv_peak = site.pv.capacity_kwp * site.pv.derate_factor * 0.78
    pv = pv_peak * shape ** 1.35
    pv *= (0.88 + 0.12 * rng.rand(n))

    # 负荷：两班制（早班 7-12，晚班 13-20，夜间低谷）
    base = site.load.avg_kw * 0.55
    day = np.where((h >= 7) & (h < 12), site.load.avg_kw * 1.35,
          np.where((h >= 12) & (h < 13), site.load.avg_kw * 1.05,
          np.where((h >= 13) & (h < 20), site.load.avg_kw * 1.45,
                   base)))
    load = day * (0.96 + 0.08 * rng.rand(n))

    return load, pv
