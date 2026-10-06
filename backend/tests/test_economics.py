import pytest

from app.domain.economics.cashflow import (
    FinancialParams,
    annual_opex,
    capex_breakdown,
    cash_flows,
    irr,
    npv,
    payback_years,
)


def test_capex_breakdown() -> None:
    params = FinancialParams()  # 1200/kWh、300/kW、安装 10%
    total, battery = capex_breakdown(params, capacity_kwh=1000, pcs_kw=500)
    assert total == pytest.approx((1200 * 1000 + 300 * 500) * 1.1)
    assert battery == pytest.approx(1200 * 1000 * 1.1)


def test_npv_zero_rate_is_cashflow_sum() -> None:
    flows = cash_flows(FinancialParams(), capex_total=1_000_000, battery_capex=700_000, annual_net=100_000)
    assert npv(0.0, flows) == pytest.approx(sum(flows))


def test_irr_known_value() -> None:
    # -1000 三年后 1331 → IRR = 10%
    assert irr([-1000.0, 0.0, 0.0, 1331.0]) == pytest.approx(0.10, abs=1e-4)


def test_irr_no_root_returns_none() -> None:
    assert irr([-1000.0, -100.0, -100.0]) is None


def test_replacement_year_cashflow() -> None:
    params = FinancialParams(battery_life_years=10, project_years=15, replacement_cost_ratio=0.6, residual_ratio=0.05)
    flows = cash_flows(params, capex_total=1_000_000, battery_capex=800_000, annual_net=100_000)
    assert len(flows) == 16
    assert flows[10] == pytest.approx(100_000 - 0.6 * 800_000)
    assert flows[15] == pytest.approx(100_000 + 0.05 * 1_000_000)
    assert flows[0] == -1_000_000


def test_replacement_beyond_project_skipped() -> None:
    params = FinancialParams(battery_life_years=20, project_years=15)
    flows = cash_flows(params, capex_total=1_000_000, battery_capex=800_000, annual_net=100_000)
    assert all(flows[y] == 100_000 for y in range(1, 15))


def test_annual_opex_and_payback() -> None:
    params = FinancialParams(opex_ratio=0.02)
    assert annual_opex(params, 1_000_000) == pytest.approx(20_000)
    assert payback_years(1_000_000, 80_000) == pytest.approx(12.5)
    assert payback_years(1_000_000, 0) is None
    assert payback_years(1_000_000, -5000) is None
