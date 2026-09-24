"""储能容量规划（Sizing）：该配多大、几年回本。

与 dispatch.py 的分工
---------------------
    dispatch.py  容量给定 → 求每天怎么充放（运行问题）
    sizing.py    容量待求 → 求配多大最划算（规划问题）

**两者必须共用同一个调度内核。** 本模块评估任何候选容量时，都调用
`rule_based_dispatch()` 真跑一遍调度——不另写"简化收益公式"。

为什么这条约束不能破：如果规划和运行用两套算法，必然出现

    规划说：3 年回本
    实际运行：赚不到这个数

企业客户一眼就能看出矛盾，信任直接归零。

求解方式：离散扫描，不用 LP
--------------------------
容量候选是有限个（典型时长 1~4 h × 若干功率档），逐个真跑调度评估即可。
峰谷时段已知，规则式调度已接近最优，所以离散扫描的精度足够——
换来的是**零依赖**（不需要 pulp / scipy / cvxpy），克隆下来直接能跑。

三个容易算错的量
----------------
1. **需量电费**：按「当月最大需量」计费，不是按天累加。
   谷时满功率充电会把并网峰值推高，省下的电度电费从基本电费里加倍还回去。
   本模块统一用 `demand_aware=True`，并单独测算需量节省。

2. **全周期收益要用多场景加权**：只拿一个典型日 × 365 会失真。
   夏季光伏出力低（高温降效）而负荷高（空调），冬季相反。
   本模块接受一组 `DayScenario`，按各自代表的天数加权得年收益。

3. **循环寿命约束**：每天 2 次循环 × 365 天 × 10 年 = 7300 次，
   超过 6000 次的电池寿命 → 项目期内必须更换电池，这笔钱要计入现金流。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..models.site import Site, StorageConfig
from ..tariff.base import PricePoint
from .dispatch import rule_based_dispatch

#: 默认容量扫描档位（额定放电时长，小时）
DEFAULT_DURATIONS = (1.0, 2.0, 3.0, 4.0)

#: 默认功率档位（相对负荷峰值的比例）
DEFAULT_POWER_RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50)

#: 默认削峰目标档位（相对基线峰值）。
#:   1.00 = 不主动削峰（仅保证充电不恶化峰值）
#:   0.80 = 把峰值削到基线的 80%
#: 为什么必须扫描它：削峰目标越低，需量电费省得越多，
#: 但谷时充电的可用余量（headroom）也越小，套利收益会下降。
#: 两者存在权衡，最优值只能算出来，不能拍脑袋定。
DEFAULT_SHAVING_RATIOS = (1.0, 0.90, 0.80, 0.70, 0.60)


# ============================================================
#  输入模型
# ============================================================

@dataclass
class DayScenario:
    """一个典型日场景。

    为什么需要"多场景"：光伏和负荷都有季节性。
    夏季光伏出力低（组件温度高、效率下降）但空调用电高；
    冬季光伏低、取暖用电中等；春秋光伏最好、负荷最平。
    只用一个典型日外推全年，会把收益算偏。
    """

    name: str
    """场景名，如「夏季」「冬季」「春秋」。"""

    load_kw: np.ndarray
    """负荷曲线（kW）。"""

    pv_kw: np.ndarray
    """光伏出力曲线（kW）。"""

    prices: list[PricePoint]
    """该场景的电价序列，长度需与上面两条一致。"""

    days_per_year: float
    """该场景一年代表多少天。所有场景之和应为 365。"""

    def __post_init__(self) -> None:
        n = len(self.load_kw)
        if len(self.pv_kw) != n or len(self.prices) != n:
            raise ValueError(
                f"场景「{self.name}」三条序列长度不一致："
                f"load={len(self.load_kw)}, pv={len(self.pv_kw)}, "
                f"prices={len(self.prices)}"
            )


@dataclass(frozen=True)
class EconomicsConfig:
    """投资测算参数。

    ⚠️ 默认值是**行业参考值**，不是任何具体项目的报价。
    实际报价必须以设备厂商与集成商为准。
    """

    discount_rate: float = 0.08
    """折现率（资金成本）。"""

    project_years: int = 10
    """项目评估期。"""

    battery_price_yuan_per_kwh: float = 1100.0
    """储能系统综合单价（元/kWh），**含 PCS 与安装**。

    按 kWh 整体计价是国内工商业储能的常见报价方式，
    比拆成「电池单价 + PCS 单价 + 安装费」更贴近实际签约形式。
    """

    battery_life_years: int = 10
    """电池寿命（年）。小于 project_years 时需中途更换。"""

    battery_cycle_life: float = 6000.0
    """循环寿命（次）。用于校验项目期内的总循环次数。"""

    opex_ratio: float = 0.02
    """年运维成本占初始投资的比例。"""

    include_replacement: bool = True
    """是否把电池更换成本计入现金流。"""

    residual_value_ratio: float = 0.05
    """期末残值占初始投资的比例。"""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EconomicsConfig":
        """从 YAML 的 economics 段构造。缺字段用默认值。"""
        d = d or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ============================================================
#  输出模型
# ============================================================

@dataclass
class Recommendation:
    """推荐配置。"""

    power_kw: float
    capacity_kwh: float
    duration_hours: float
    chemistry: str = "LFP"
    round_trip_efficiency: float = 0.88
    depth_of_discharge: float = 0.90
    cycles_per_day: float = 0.0
    """达成该收益所需的日均等效循环次数。"""

    peak_shaving_target_ratio: float = 1.0
    """最优削峰目标（相对基线峰值）。0.8 表示把并网峰值削到原峰值的 80%。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "power_kw": round(self.power_kw, 1),
            "capacity_kwh": round(self.capacity_kwh, 1),
            "duration_hours": round(self.duration_hours, 2),
            "chemistry": self.chemistry,
            "round_trip_efficiency": self.round_trip_efficiency,
            "depth_of_discharge": self.depth_of_discharge,
            "cycles_per_day": round(self.cycles_per_day, 2),
            "peak_shaving_target_ratio": round(self.peak_shaving_target_ratio, 2),
        }


@dataclass
class Economics:
    """经济性指标。"""

    capex_yuan: float
    opex_yuan_per_year: float
    arbitrage_yuan: float
    """年峰谷套利收益（即调度算出的电度电费节省）。"""

    demand_saving_yuan: float
    """年需量电费节省。"""

    annual_benefit_yuan: float
    """年总收益 = 套利 + 需量节省。"""

    annual_net_yuan: float
    """年净现金流 = 年总收益 − 年运维。"""

    simple_payback_years: float
    irr: float
    npv_yuan: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "capex_yuan": round(self.capex_yuan, 0),
            "opex_yuan_per_year": round(self.opex_yuan_per_year, 0),
            "annual_benefit": {
                "arbitrage_yuan": round(self.arbitrage_yuan, 0),
                "demand_charge_saving_yuan": round(self.demand_saving_yuan, 0),
                "total_yuan": round(self.annual_benefit_yuan, 0),
            },
            "annual_net_yuan": round(self.annual_net_yuan, 0),
            "simple_payback_years": round(self.simple_payback_years, 2),
            "irr": round(self.irr, 4),
            "npv_10y_yuan": round(self.npv_yuan, 0),
        }


@dataclass
class SensitivityEntry:
    """敏感性分析的一行。"""

    variable: str
    change: str
    payback_years: float
    npv_yuan: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable": self.variable,
            "change": self.change,
            "payback_years": round(self.payback_years, 2),
            "npv_yuan": round(self.npv_yuan, 0),
        }


@dataclass
class CandidateResult:
    """一个候选容量的评估结果（供扫描明细表使用）。"""

    power_kw: float
    capacity_kwh: float
    duration_hours: float
    shaving_ratio: float
    """最优削峰目标（相对基线峰值）。1.0 表示不主动削峰。"""

    arbitrage_yuan: float
    demand_saving_yuan: float
    annual_benefit_yuan: float
    capex_yuan: float
    npv_yuan: float
    irr: float
    payback_years: float
    cycles_per_day: float
    peak_reduction_kw: float = 0.0
    """加权平均的削峰量（kW），用于展示「这套配置削掉多少需量」。"""

    feasible: bool = True
    note: str = ""


@dataclass
class SizingResult:
    """容量规划的完整输出。"""

    site_id: str
    scenario: str
    recommendation: Recommendation
    economics: Economics
    sensitivity: list[SensitivityEntry] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    candidates: list[CandidateResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转为可直接 JSON 序列化的字典（对齐设计文档 5.1 节）。"""
        return {
            "site_id": self.site_id,
            "scenario": self.scenario,
            "recommendation": self.recommendation.to_dict(),
            "economics": self.economics.to_dict(),
            "sensitivity": [s.to_dict() for s in self.sensitivity],
            "assumptions": self.assumptions,
            "notes": self.notes,
        }

    def summary(self) -> str:
        r, e = self.recommendation, self.economics
        lines = [
            f"推荐配置  {r.power_kw:,.0f} kW / {r.capacity_kwh:,.0f} kWh "
            f"({r.duration_hours:.1f} h, {r.chemistry})",
            f"投资      {e.capex_yuan:,.0f} 元   "
            f"年运维 {e.opex_yuan_per_year:,.0f} 元",
            f"年收益    套利 {e.arbitrage_yuan:,.0f} + "
            f"需量 {e.demand_saving_yuan:,.0f} "
            f"= {e.annual_benefit_yuan:,.0f} 元",
            f"年净现金流 {e.annual_net_yuan:,.0f} 元",
            f"回收期    {e.simple_payback_years:.2f} 年   "
            f"IRR {e.irr:.1%}   NPV {e.npv_yuan:,.0f} 元",
            f"日均循环   {r.cycles_per_day:.2f} 次/天",
        ]
        return "\n".join(lines)


# ============================================================
#  财务计算
# ============================================================

def _npv(rate: float, cashflows: list[float]) -> float:
    """净现值。cashflows[0] 为期初支出（负值）。"""
    return sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cashflows))


def _irr(cashflows: list[float]) -> float:
    """内部收益率，二分法求 NPV=0 的折现率。

    不用牛顿法：现金流可能有多次符号变化（中途换电池产生负项），
    牛顿法在这种情形下容易不收敛，二分法虽然慢一点但稳定。
    """
    lo, hi = -0.9, 5.0
    f_lo, f_hi = _npv(lo, cashflows), _npv(hi, cashflows)

    # 全周期都在赚钱（f_lo>0 说明折现率 -90% 时仍为正）→ 收益率为无穷大
    if f_lo < 0 or f_hi > 0:
        return float("nan")

    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = _npv(mid, cashflows)
        if abs(f_mid) < 1e-6:
            return mid
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _build_cashflows(
    capex: float,
    annual_net: float,
    econ: EconomicsConfig,
) -> list[float]:
    """构造全周期现金流。

    t=0 期初投资；t=1..N 年净收益；期末加残值。
    若电池寿命短于项目期且启用更换，在寿命到期年扣一笔更换成本
    （扣除残值抵扣）。
    """
    n = econ.project_years
    flows = [-capex] + [annual_net] * n

    if econ.include_replacement and 0 < econ.battery_life_years < n:
        # 更换成本 = 初始投资 × (1 − 旧电池残值率)
        replacement = capex * (1.0 - econ.residual_value_ratio)
        for yr in range(econ.battery_life_years, n + 1, econ.battery_life_years):
            flows[yr] -= replacement

    flows[-1] += capex * econ.residual_value_ratio
    return flows


# ============================================================
#  候选生成
# ============================================================

def default_candidates(
    site: Site,
    *,
    durations: tuple[float, ...] = DEFAULT_DURATIONS,
    power_ratios: tuple[float, ...] = DEFAULT_POWER_RATIOS,
) -> list[tuple[float, float]]:
    """按负荷规模自动生成候选 (pcs_kw, capacity_kwh) 组合。

    功率档位以负荷峰值为基准取比例，而不是拍脑袋定绝对值——
    1 MW 的厂和 100 kW 的厂，合理配置差一个数量级。
    """
    ref = site.load.peak_kw
    grid: list[float] = []
    for r in power_ratios:
        p = round(ref * r / 50.0) * 50.0      # 取整到 50 kW，避免出现 183.7 kW 这种数
        if p > 0 and p not in grid:
            grid.append(p)

    return [(p, p * d) for p in grid for d in durations]


def _scale_price_spread(
    prices: list[PricePoint],
    factor: float,
) -> list[PricePoint]:
    """按比例放大/缩小电价的「峰谷差」，用于敏感性分析。

    做法：以**平段**价格为锚点，把各时段价格向锚点收拢或推远。

        new = anchor + (old − anchor) × factor

    factor=0.8 即「峰谷价差缩小 20%」，而**平段价格水平不变**。
    这正是敏感性分析要检验的「价差风险」。

    为什么不用「整体乘 0.8」：那检验的是「电价水平下降」，
    是另一个问题——而且它同时压低了充电成本和放电收入，
    对套利收益的影响方向不明确，起不到敏感性检验的作用。

    为什么锚点选平段而不是中位数：平段是峰谷的天然分界，
    也是政策文件里唯一"基准"性质的档位。中位数可能落在高峰档上，
    那样缩放会连平段价格一起抬升，口径就乱了。
    """
    flat_prices = [p.price for p in prices if p.tier == "flat"]
    anchor = (float(np.median(flat_prices)) if flat_prices
              else float(np.median([p.price for p in prices])))

    out: list[PricePoint] = []
    for p in prices:
        new_price = anchor + (p.price - anchor) * factor
        # 电能量价格同步缩放，保持「到户价 = 能量价 + 输配 + 基金」的构成关系
        new_energy = anchor + (p.energy_price - anchor) * factor
        out.append(replace(p,
                           price=round(new_price, 6),
                           energy_price=round(new_energy, 6)))
    return out


# ============================================================
#  单次评估
# ============================================================

def _evaluate_candidate(
    template: StorageConfig,
    pcs_kw: float,
    capacity_kwh: float,
    scenarios: list[DayScenario],
    *,
    demand_charge_price: float,
    resolution_minutes: int,
    allow_grid_charge: bool,
    allow_grid_export: bool,
    shaving_ratio: float,
) -> tuple[float, float, float, float]:
    """对给定容量 + 削峰目标跑全部场景，返回加权年化结果：

        (套利收益, 需量节省, 日均循环, 加权平均削峰量)
    """
    storage = replace(template, pcs_kw=pcs_kw, capacity_kwh=capacity_kwh)

    arb_year = dem_year = 0.0
    cycles_w = peak_red_w = 0.0

    for sc in scenarios:
        net = sc.load_kw - sc.pv_kw
        base_peak = float(np.max(np.clip(net, 0.0, None)))
        target = base_peak * shaving_ratio

        res = rule_based_dispatch(
            sc.load_kw, sc.pv_kw, sc.prices, storage,
            resolution_minutes=resolution_minutes,
            allow_grid_charge=allow_grid_charge,
            allow_grid_export=allow_grid_export,
            demand_aware=True,              # ⚠️ 硬性要求，见模块 docstring
            peak_target_kw=target,
        )
        w = sc.days_per_year
        arb_year += res.savings_yuan * w                     # 电度电费节省
        # 需量电费按月计费：单场景日的削峰量 × 单价 × 12 个月
        dem_year += res.peak_reduction_kw * demand_charge_price * 12.0 * (w / 365.0)
        cycles_w += res.cycles_used * w
        peak_red_w += res.peak_reduction_kw * w

    total_days = sum(sc.days_per_year for sc in scenarios) or 1.0
    return (arb_year, dem_year, cycles_w / total_days, peak_red_w / total_days)


def _evaluate_full(
    template: StorageConfig,
    pcs_kw: float,
    capacity_kwh: float,
    scenarios: list[DayScenario],
    econ: EconomicsConfig,
    *,
    demand_charge_price: float,
    resolution_minutes: int,
    allow_grid_charge: bool,
    allow_grid_export: bool,
    shaving_ratios: tuple[float, ...],
) -> CandidateResult:
    """评估一个候选——**同时对多个削峰目标求最优**，再算完整经济性。

    为什么这里要再套一层循环：削峰目标不是给定的，而是规划要决策的。
    capex 只取决于容量，与削峰目标无关，所以「年总收益最大」的方案
    就是「NPV 最大」的方案。
    """
    best: tuple[float, float, float, float, float] | None = None
    for ratio in shaving_ratios:
        arb, dem, cyc, peak_red = _evaluate_candidate(
            template, pcs_kw, capacity_kwh, scenarios,
            demand_charge_price=demand_charge_price,
            resolution_minutes=resolution_minutes,
            allow_grid_charge=allow_grid_charge,
            allow_grid_export=allow_grid_export,
            shaving_ratio=ratio,
        )
        if best is None or (arb + dem) > (best[0] + best[1]):
            best = (arb, dem, cyc, peak_red, ratio)

    assert best is not None
    arb, dem, cycles, peak_red, ratio = best

    capex = capacity_kwh * econ.battery_price_yuan_per_kwh
    opex = capex * econ.opex_ratio
    annual_benefit = arb + dem
    annual_net = annual_benefit - opex

    flows = _build_cashflows(capex, annual_net, econ)
    npv = _npv(econ.discount_rate, flows)
    irr = _irr(flows)
    payback = capex / annual_net if annual_net > 0 else float("inf")

    return CandidateResult(
        power_kw=pcs_kw,
        capacity_kwh=capacity_kwh,
        duration_hours=capacity_kwh / pcs_kw if pcs_kw > 0 else 0.0,
        shaving_ratio=ratio,
        arbitrage_yuan=arb,
        demand_saving_yuan=dem,
        annual_benefit_yuan=annual_benefit,
        capex_yuan=capex,
        npv_yuan=npv,
        irr=irr,
        payback_years=payback,
        cycles_per_day=cycles,
        peak_reduction_kw=peak_red,
    )


# ============================================================
#  主入口
# ============================================================

def plan_capacity(
    scenarios: list[DayScenario],
    site: Site,
    econ: EconomicsConfig,
    *,
    demand_charge_price: float = 0.0,
    resolution_minutes: int = 15,
    allow_grid_charge: bool = True,
    allow_grid_export: bool = False,
    candidates: list[tuple[float, float]] | None = None,
    max_payback_years: float = 8.0,
    shaving_ratios: tuple[float, ...] = DEFAULT_SHAVING_RATIOS,
    sensitivity_spread_changes: tuple[float, ...] = (0.8, 1.2),
) -> SizingResult:
    """容量规划主入口：扫描候选，返回 NPV 最优方案。

    参数
    ----
    scenarios
        典型日场景列表，天数之和应为 365。年收益由各场景加权得到。
    site
        站点档案。`site.storage` 作为参数模板（效率、SOC 区间等），
        其 `pcs_kw` / `capacity_kwh` 会被候选值**覆盖**。
    econ
        投资测算参数。
    demand_charge_price
        需量电费单价（元/kW·月）。为 0 表示不考虑需量电费。
    max_payback_years
        回收期上限。超过此值的方案标记为不可行——企业通常不接受
        回收期过长的投资，即使 NPV 为正。
    shaving_ratios
        削峰目标候选（相对各场景的基线峰值）。**规划的另一半决策变量**。
        每个容量候选都会对全部目标试算一遍，取年收益最大者。
        默认从 1.0（不削峰）试到 0.6（削到基线的六成）。
    sensitivity_spread_changes
        敏感性分析要检验的价差变化比例，默认 ±20%。

    返回
    ----
    SizingResult，含推荐配置、经济性、敏感性分析、扫描明细。
    """
    if not scenarios:
        raise ValueError("scenarios 不能为空")
    if site.storage is None:
        raise ValueError("site.storage 缺失：规划需要一份参数模板（效率/SOC 区间等）")

    total_days = sum(sc.days_per_year for sc in scenarios)
    if abs(total_days - 365.0) > 1.0:
        raise ValueError(
            f"场景天数之和应为 365，当前为 {total_days:.0f}。"
            f"年收益靠天数加权，天数不对整个测算就失真。"
        )

    template = site.storage
    cands = candidates or default_candidates(site)

    # ---- 扫描 ----
    evaluated: list[CandidateResult] = []
    for pcs_kw, cap_kwh in cands:
        r = _evaluate_full(
            template, pcs_kw, cap_kwh, scenarios, econ,
            demand_charge_price=demand_charge_price,
            resolution_minutes=resolution_minutes,
            allow_grid_charge=allow_grid_charge,
            allow_grid_export=allow_grid_export,
            shaving_ratios=shaving_ratios,
        )
        if not np.isfinite(r.payback_years) or r.payback_years > max_payback_years:
            r.feasible = False
            r.note = (f"回收期 {r.payback_years:.1f} 年超过上限 "
                      f"{max_payback_years:.0f} 年") if np.isfinite(r.payback_years) else "年净收益非正"
        evaluated.append(r)

    feasible = [r for r in evaluated if r.feasible]
    if not feasible:
        raise ValueError(
            f"没有任何候选方案满足回收期 ≤ {max_payback_years:.0f} 年。"
            f"可能原因：峰谷价差过小 / 需量电费未计入 / 电池单价过高。"
            f"请检查 config 中的电价与 economics 参数。"
        )

    # NPV 最大的可行方案
    best = max(feasible, key=lambda r: r.npv_yuan)

    # ---- 敏感性分析 ----
    base_arb = best.arbitrage_yuan
    base_dem = best.demand_saving_yuan
    sensitivity: list[SensitivityEntry] = []

    for factor in sensitivity_spread_changes:
        pct = int(round((factor - 1.0) * 100))
        label = f"{pct:+d}%"
        scaled = [
            DayScenario(
                name=sc.name,
                load_kw=sc.load_kw,
                pv_kw=sc.pv_kw,
                prices=_scale_price_spread(sc.prices, factor),
                days_per_year=sc.days_per_year,
            )
            for sc in scenarios
        ]
        s = _evaluate_full(
            template, best.power_kw, best.capacity_kwh, scaled, econ,
            demand_charge_price=demand_charge_price,
            resolution_minutes=resolution_minutes,
            allow_grid_charge=allow_grid_charge,
            allow_grid_export=allow_grid_export,
            shaving_ratios=(best.shaving_ratio,),
        )
        sensitivity.append(SensitivityEntry(
            variable="价差", change=label,
            payback_years=s.payback_years, npv_yuan=s.npv_yuan,
        ))

    # 循环寿命检验
    cyc_total = best.cycles_per_day * 365.0 * econ.project_years
    if cyc_total > econ.battery_cycle_life:
        life_years = econ.battery_cycle_life / max(best.cycles_per_day * 365.0, 1e-9)
        replacement_note = (
            f"项目期总循环 {cyc_total:,.0f} 次 > 电池循环寿命 "
            f"{econ.battery_cycle_life:,.0f} 次，"
            f"按当前充放强度电池约 {life_years:.1f} 年后需更换"
        )
        if econ.include_replacement:
            replacement_note += "（更换成本已计入现金流）"
        else:
            replacement_note += "（⚠️ 当前未计入更换成本，NPV 偏乐观）"
    else:
        replacement_note = (
            f"项目期总循环 {cyc_total:,.0f} 次 < 电池循环寿命 "
            f"{econ.battery_cycle_life:,.0f} 次，无更换需求"
        )

    notes = [replacement_note]

    # 削峰目标说明
    if demand_charge_price > 0:
        if best.shaving_ratio >= 0.999:
            notes.append(
                "削峰目标扫描结果为「不主动削峰」（1.00）——"
                "说明本站在当前配置下，削峰的需量收益抵不上充电余量收紧带来的套利损失"
            )
        else:
            notes.append(
                f"最优削峰目标 {best.shaving_ratio:.0%}（相对基线峰值），"
                f"加权平均削峰 {best.peak_reduction_kw:,.0f} kW"
            )

    # 若最优方案处于扫描边界，提示扩大范围
    max_p = max(r.power_kw for r in evaluated)
    max_c = max(r.capacity_kwh for r in evaluated)
    if best.power_kw >= max_p:
        notes.append(f"⚠️ 最优功率落在扫描上界 {max_p:.0f} kW，"
                     f"建议扩大候选范围后重算")
    if best.capacity_kwh >= max_c:
        notes.append(f"⚠️ 最优容量落在扫描上界 {max_c:.0f} kWh，"
                     f"建议扩大候选范围后重算")

    # ---- 组装结果 ----
    rec = Recommendation(
        power_kw=best.power_kw,
        capacity_kwh=best.capacity_kwh,
        duration_hours=best.duration_hours,
        chemistry=template.chemistry,
        round_trip_efficiency=template.round_trip_eff,
        depth_of_discharge=template.depth_of_discharge,
        cycles_per_day=best.cycles_per_day,
        peak_shaving_target_ratio=best.shaving_ratio,
    )

    capex = best.capacity_kwh * econ.battery_price_yuan_per_kwh
    opex = capex * econ.opex_ratio
    annual_net = best.annual_benefit_yuan - opex
    ec = Economics(
        capex_yuan=capex,
        opex_yuan_per_year=opex,
        arbitrage_yuan=base_arb,
        demand_saving_yuan=base_dem,
        annual_benefit_yuan=best.annual_benefit_yuan,
        annual_net_yuan=annual_net,
        simple_payback_years=best.payback_years,
        irr=best.irr,
        npv_yuan=best.npv_yuan,
    )

    assumptions = {
        "battery_price_yuan_per_kwh": econ.battery_price_yuan_per_kwh,
        "battery_price_note": "按 kWh 综合单价（含 PCS 与安装）",
        "battery_life_years": econ.battery_life_years,
        "battery_cycle_life": econ.battery_cycle_life,
        "discount_rate": econ.discount_rate,
        "project_years": econ.project_years,
        "opex_ratio": econ.opex_ratio,
        "residual_value_ratio": econ.residual_value_ratio,
        "demand_charge_price_yuan_per_kw_month": demand_charge_price,
        "demand_charge_note": "按当月最大需量计费；年节省 = 日削峰量 × 单价 × 12",
        "scenario_note": "年收益由多场景按天数加权，非单日 × 365",
        "resolution_minutes": resolution_minutes,
        "allow_grid_charge": allow_grid_charge,
        "allow_grid_export": allow_grid_export,
        "demand_aware": True,
        "peak_shaving_target_ratio": round(best.shaving_ratio, 2),
        "peak_shaving_note": (
            "削峰目标相对基线峰值（1.0 = 不主动削峰）。"
            "规划阶段已对 " + "/".join(f"{r:.2f}" for r in shaving_ratios) +
            " 多档试算，取年收益最大者"
        ),
    }

    return SizingResult(
        site_id=site.id,
        scenario="峰谷套利" + (" + 需量管理" if demand_charge_price > 0 else ""),
        recommendation=rec,
        economics=ec,
        sensitivity=sensitivity,
        assumptions=assumptions,
        candidates=evaluated,
        notes=notes,
    )
