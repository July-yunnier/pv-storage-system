"""Sizing 结果 DTO。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


class SizingCandidateResult(BaseModel):
    pcs_kw: float
    capacity_kwh: float
    duration_h: float
    capex_yuan: float
    annual_benefit_yuan: float
    annual_opex_yuan: float
    payback_years: float | None
    npv_yuan: float
    irr: float | None
    equivalent_full_cycles_per_day: float


class SensitivityEntry(BaseModel):
    delta: float
    annual_benefit_yuan: float
    payback_years: float | None
    npv_yuan: float


class SizingSensitivity(BaseModel):
    price_spread: list[SensitivityEntry] = []
    battery_price: list[SensitivityEntry] = []


class SizingResult(BaseModel):
    site_id: str
    representative_dates: list[date]
    recommended: SizingCandidateResult | None
    candidates: list[SizingCandidateResult]
    sensitivity: SizingSensitivity
    assumptions: dict[str, Any]
    notes: str = ""


class SizingResponse(SizingResult):
    run_id: str
    created_at: datetime | None = None


class SizingRunDetail(SizingResponse):
    input_digest: str


class SizingRunSummary(BaseModel):
    run_id: str
    site_id: str
    created_at: datetime
    recommended_pcs_kw: float | None
    recommended_capacity_kwh: float | None
    npv_yuan: float | None
    payback_years: float | None
