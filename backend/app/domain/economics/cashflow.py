"""储能项目经济性计算：CAPEX/OPEX/现金流/NPV/IRR/回本。stdlib only。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinancialParams(BaseModel):
    battery_price_per_kwh: float = Field(default=1200.0, ge=0)
    pcs_price_per_kw: float = Field(default=300.0, ge=0)
    install_ratio: float = Field(default=0.10, ge=0)
    opex_ratio: float = Field(default=0.02, ge=0)
    project_years: int = Field(default=15, ge=1)
    battery_life_years: int = Field(default=10, ge=1)
    replacement_cost_ratio: float = Field(default=0.60, ge=0)
    discount_rate: float = Field(default=0.08, ge=-0.99)
    residual_ratio: float = Field(default=0.05, ge=0)


def capex_breakdown(
    params: FinancialParams, capacity_kwh: float, pcs_kw: float
) -> tuple[float, float]:
    """返回 (总 CAPEX, 其中电池部分)。安装费按比例摊到两部分。"""
    battery = params.battery_price_per_kwh * capacity_kwh * (1 + params.install_ratio)
    pcs = params.pcs_price_per_kw * pcs_kw * (1 + params.install_ratio)
    return battery + pcs, battery


def annual_opex(params: FinancialParams, capex_total: float) -> float:
    return params.opex_ratio * capex_total


def cash_flows(
    params: FinancialParams,
    capex_total: float,
    battery_capex: float,
    annual_net: float,
) -> list[float]:
    """flows[t] 为第 t 年末现金流，flows[0] 为初始投资（负）。

    第 battery_life_years 年末更换电池；第 project_years 年末回收残值。
    """
    flows = [-capex_total] + [annual_net] * params.project_years
    if params.battery_life_years <= params.project_years:
        flows[params.battery_life_years] -= params.replacement_cost_ratio * battery_capex
    flows[params.project_years] += params.residual_ratio * capex_total
    return flows


def npv(rate: float, flows: list[float]) -> float:
    return sum(flow / (1 + rate) ** t for t, flow in enumerate(flows))


def irr(flows: list[float], lo: float = -0.99, hi: float = 10.0) -> float | None:
    """二分法求使 NPV=0 的折现率；区间内无根返回 None。"""
    f_lo, f_hi = npv(lo, flows), npv(hi, flows)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid, flows)
        if abs(f_mid) < 1e-10:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def payback_years(capex_total: float, annual_net: float) -> float | None:
    """简单回本年限；年净收益非正时无法回本，返回 None。"""
    if annual_net <= 0:
        return None
    return capex_total / annual_net
