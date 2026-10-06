"""容量规划：对候选 (pcs, duration) 组合用同一个调度引擎跑代表日，再折算全年。

原则：Sizing 不凭公式，年化 = 代表日平均日收益 × 365（已知简化，写进 assumptions）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.economics.cashflow import (
    FinancialParams,
    annual_opex,
    capex_breakdown,
    cash_flows,
    irr,
    npv,
    payback_years,
)
from app.domain.models import MeasurementRead, SiteRead, TariffRead, TariffTier
from app.domain.sizing.result import (
    SensitivityEntry,
    SizingCandidateResult,
    SizingResult,
    SizingSensitivity,
)
from app.domain.tariff import TouProvider
from app.infrastructure.db import _zone

ANNUALIZATION_NOTE = "典型日算术平均×365，未考虑季节差异"
SENSITIVITY_DELTAS = (-0.2, 0.0, 0.2)


@dataclass(frozen=True)
class Candidate:
    pcs_kw: float
    duration_h: float
    capacity_kwh: float


@dataclass(frozen=True)
class RepresentativeDay:
    date: date
    pv_values: list[float]
    load_values: list[float]


def generate_candidates(pcs_list: list[float], duration_list: list[float]) -> list[Candidate]:
    candidates = [
        Candidate(pcs_kw=pcs, duration_h=duration, capacity_kwh=pcs * duration)
        for pcs in pcs_list
        for duration in duration_list
        if pcs * duration > 0
    ]
    if not candidates:
        raise ValueError("candidate grid produced no valid (pcs, duration) combination")
    return candidates


def _day_inputs(
    site: SiteRead,
    day: RepresentativeDay,
    provider: TouProvider,
    initial_soc: float,
) -> DispatchInputs:
    tz = _zone(site.location.timezone)
    day_start = datetime.combine(day.date, time.min, tzinfo=tz)

    def series(data_type: str, values: list[float]) -> list[MeasurementRead]:
        return [
            MeasurementRead(
                data_type=data_type,  # type: ignore[arg-type]
                ts=day_start + timedelta(minutes=15 * slot),
                value=value,
                unit="kW",
                quality="measured",
                source="sizing",
            )
            for slot, value in enumerate(values)
        ]

    pv_points = series("pv_power", day.pv_values)
    return DispatchInputs(
        site=site,
        target_date=day.date,
        initial_soc=initial_soc,
        pv_points=pv_points,
        load_points=series("load_power", day.load_values),
        price_points=provider.points(day_start, periods=len(day.pv_values), resolution_minutes=15),
    )


def _candidate_site(site: SiteRead, candidate: Candidate) -> SiteRead:
    storage = site.storage.model_copy(
        update={"pcs_kw": candidate.pcs_kw, "capacity_kwh": candidate.capacity_kwh}
    )
    return site.model_copy(update={"storage": storage})


def _annual_benefit(
    site: SiteRead,
    days: list[RepresentativeDay],
    provider: TouProvider,
    candidate: Candidate,
    initial_soc: float,
) -> tuple[float, float]:
    """逐代表日跑引擎，返回 (年化收益, 平均每日等效满循环次数)。"""
    engine = RuleDispatchEngine()
    candidate_site = _candidate_site(site, candidate)
    benefits = []
    cycles = []
    for day in days:
        result = engine.run(_day_inputs(candidate_site, day, provider, initial_soc))
        benefits.append(result.expected_benefit_yuan)
        cycles.append(result.equivalent_full_cycles)
    daily_mean = sum(benefits) / len(benefits)
    return daily_mean * 365, sum(cycles) / len(cycles)


def _economics(
    params: FinancialParams, candidate: Candidate, annual_benefit: float, cycles: float
) -> SizingCandidateResult:
    capex_total, battery_capex = capex_breakdown(params, candidate.capacity_kwh, candidate.pcs_kw)
    opex = annual_opex(params, capex_total)
    annual_net = annual_benefit - opex
    flows = cash_flows(params, capex_total, battery_capex, annual_net)
    return SizingCandidateResult(
        pcs_kw=candidate.pcs_kw,
        capacity_kwh=candidate.capacity_kwh,
        duration_h=candidate.duration_h,
        capex_yuan=round(capex_total, 2),
        annual_benefit_yuan=round(annual_benefit, 2),
        annual_opex_yuan=round(opex, 2),
        payback_years=_round_or_none(payback_years(capex_total, annual_net)),
        npv_yuan=round(npv(params.discount_rate, flows), 2),
        irr=_round_or_none(irr(flows)),
        equivalent_full_cycles_per_day=round(cycles, 6),
    )


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _scale_price_spread(tariff: TariffRead, delta: float) -> TariffRead:
    """tier 电价围绕均价缩放 (1+delta)，峰谷价差随之变化。"""
    prices = [tier.price for tier in tariff.tiers]
    mean_price = sum(prices) / len(prices)
    scaled = [
        TariffTier(name=tier.name, price=round(mean_price + (tier.price - mean_price) * (1 + delta), 6), hours=tier.hours)
        for tier in tariff.tiers
    ]
    return tariff.model_copy(update={"tiers": scaled})


def _sensitivity(
    site: SiteRead,
    days: list[RepresentativeDay],
    tariff: TariffRead,
    params: FinancialParams,
    recommended: SizingCandidateResult,
    initial_soc: float,
) -> SizingSensitivity:
    candidate = Candidate(
        pcs_kw=recommended.pcs_kw,
        duration_h=recommended.duration_h,
        capacity_kwh=recommended.capacity_kwh,
    )
    price_entries: list[SensitivityEntry] = []
    for delta in SENSITIVITY_DELTAS:
        provider = TouProvider(_scale_price_spread(tariff, delta))
        benefit, cycles = _annual_benefit(site, days, provider, candidate, initial_soc)
        result = _economics(params, candidate, benefit, cycles)
        price_entries.append(
            SensitivityEntry(
                delta=delta,
                annual_benefit_yuan=result.annual_benefit_yuan,
                payback_years=result.payback_years,
                npv_yuan=result.npv_yuan,
            )
        )
    battery_entries: list[SensitivityEntry] = []
    for delta in SENSITIVITY_DELTAS:
        varied = params.model_copy(
            update={"battery_price_per_kwh": params.battery_price_per_kwh * (1 + delta)}
        )
        result = _economics(varied, candidate, recommended.annual_benefit_yuan, recommended.equivalent_full_cycles_per_day)
        battery_entries.append(
            SensitivityEntry(
                delta=delta,
                annual_benefit_yuan=result.annual_benefit_yuan,
                payback_years=result.payback_years,
                npv_yuan=result.npv_yuan,
            )
        )
    return SizingSensitivity(price_spread=price_entries, battery_price=battery_entries)


def build_assumptions(params: FinancialParams, representative_days: int, initial_soc: float) -> dict:
    return {
        "annualization": ANNUALIZATION_NOTE,
        "representative_days": representative_days,
        "initial_soc": initial_soc,
        "strategy": "rule_v1",
        **params.model_dump(),
    }


def evaluate_sizing(
    site: SiteRead,
    days: list[RepresentativeDay],
    tariff: TariffRead,
    candidates: list[Candidate],
    params: FinancialParams,
    initial_soc: float = 0.5,
) -> SizingResult:
    provider = TouProvider(tariff)
    results = [
        _economics(params, candidate, *_annual_benefit(site, days, provider, candidate, initial_soc))
        for candidate in candidates
    ]
    profitable = [r for r in results if r.npv_yuan > 0]
    recommended = max(profitable, key=lambda r: r.npv_yuan) if profitable else None
    notes = "" if recommended else "所有候选组合的 NPV 均不大于 0，不建议按当前假设投资"
    sensitivity = (
        _sensitivity(site, days, tariff, params, recommended, initial_soc)
        if recommended
        else SizingSensitivity()
    )
    return SizingResult(
        site_id=site.site_id,
        representative_dates=[day.date for day in days],
        recommended=recommended,
        candidates=results,
        sensitivity=sensitivity,
        assumptions=build_assumptions(params, len(days), initial_soc),
        notes=notes,
    )
