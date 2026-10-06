from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.domain.dispatch import DispatchInputs, RuleDispatchEngine
from app.domain.models import (
    LoadConfig,
    MeasurementRead,
    PVConfig,
    SiteLocation,
    SiteRead,
    StorageConfig,
    TariffCreate,
    TariffRead,
)
from app.domain.tariff import TouProvider


@pytest.fixture()
def dispatch_inputs() -> DispatchInputs:
    now = datetime.now(timezone.utc)
    site = SiteRead(
        site_id="S-TEST",
        name="测试站点",
        location=SiteLocation(lat=32.2, lon=118.7),
        pv=PVConfig(capacity_kwp=1000),
        load=LoadConfig(peak_kw=800),
        storage=StorageConfig(
            pcs_kw=500,
            capacity_kwh=1000,
            round_trip_eff=0.88,
            soc_min=0.10,
            soc_max=0.95,
        ),
        tariff_id="T-TEST",
        created_at=now,
        updated_at=now,
    )
    tariff = TariffRead(
        **TariffCreate(
            tariff_id="T-TEST",
            name="测试电价",
            transmission_and_fund=0.21,
            tiers=[
                {"name": "peak", "price": 1.0, "hours": ["08:00-20:00"]},
                {"name": "valley", "price": 0.3, "hours": ["20:00-08:00"]},
            ],
        ).model_dump(),
        created_at=now,
        updated_at=now,
    )
    start = datetime(2026, 9, 20, tzinfo=timezone(timedelta(hours=8)))
    pv = [
        MeasurementRead(
            data_type="pv_power",
            ts=start + timedelta(minutes=15 * index),
            value=0,
            unit="kW",
            source="test",
        )
        for index in range(96)
    ]
    load = [
        MeasurementRead(
            data_type="load_power",
            ts=start + timedelta(minutes=15 * index),
            value=800,
            unit="kW",
            source="test",
        )
        for index in range(96)
    ]
    prices = TouProvider(tariff).points(start, 96)
    return DispatchInputs(
        site=site,
        target_date=start.date(),
        initial_soc=0.5,
        pv_points=pv,
        load_points=load,
        price_points=prices,
    )


def test_rule_dispatch_respects_soc_and_reduces_cost(dispatch_inputs: DispatchInputs) -> None:
    result = RuleDispatchEngine().run(dispatch_inputs)
    assert len(result.schedule) == 96
    assert result.expected_benefit_yuan > 0
    assert any(point.action == "charge" for point in result.schedule)
    assert any(point.action == "discharge" for point in result.schedule)


def test_soc_stays_within_limits(dispatch_inputs: DispatchInputs) -> None:
    result = RuleDispatchEngine().run(dispatch_inputs)
    storage = dispatch_inputs.site.storage
    for point in result.schedule:
        assert storage.soc_min <= point.soc_start <= storage.soc_max
        assert storage.soc_min <= point.soc_end <= storage.soc_max


def test_power_within_pcs_limit(dispatch_inputs: DispatchInputs) -> None:
    result = RuleDispatchEngine().run(dispatch_inputs)
    pcs_kw = dispatch_inputs.site.storage.pcs_kw
    assert all(point.power_kw <= pcs_kw for point in result.schedule)


def test_no_grid_export_when_disabled(dispatch_inputs: DispatchInputs) -> None:
    assert dispatch_inputs.site.storage.allow_grid_export is False
    result = RuleDispatchEngine().run(dispatch_inputs)
    assert all(point.grid_export_kw == 0 for point in result.schedule)


def test_dispatch_is_deterministic(dispatch_inputs: DispatchInputs) -> None:
    first = RuleDispatchEngine().run(dispatch_inputs)
    second = RuleDispatchEngine().run(dispatch_inputs)
    assert first.model_dump() == second.model_dump()


def test_round_trip_efficiency_and_energy_ledger(dispatch_inputs: DispatchInputs) -> None:
    storage = dispatch_inputs.site.storage
    assert storage.charge_eff == pytest.approx(storage.round_trip_eff**0.5)
    assert storage.discharge_eff == pytest.approx(storage.round_trip_eff**0.5)
    result = RuleDispatchEngine().run(dispatch_inputs)
    dt_hours = dispatch_inputs.resolution_minutes / 60
    capacity = storage.capacity_kwh
    for point in result.schedule:
        if point.action == "charge":
            expected_kwh = point.power_kw * dt_hours * storage.charge_eff
        elif point.action == "discharge":
            expected_kwh = -point.power_kw * dt_hours / storage.discharge_eff
        else:
            expected_kwh = 0.0
        actual_kwh = (point.soc_end - point.soc_start) * capacity
        assert actual_kwh == pytest.approx(expected_kwh, abs=2e-3)


def test_none_value_point_raises(dispatch_inputs: DispatchInputs) -> None:
    dispatch_inputs.pv_points[10].value = None
    with pytest.raises(ValueError, match="缺失点"):
        RuleDispatchEngine().run(dispatch_inputs)
    dispatch_inputs.pv_points[10].value = 0
    dispatch_inputs.load_points[20].value = None
    with pytest.raises(ValueError, match="缺失点"):
        RuleDispatchEngine().run(dispatch_inputs)


def test_pv_surplus_charges_battery(dispatch_inputs: DispatchInputs) -> None:
    # 10:00-14:00 光伏 900 > 负荷 800，余电 100 kW 充电
    for index in range(40, 56):
        dispatch_inputs.pv_points[index].value = 900
    result = RuleDispatchEngine().run(dispatch_inputs)
    surplus_points = result.schedule[40:56]
    assert all(point.action == "charge" for point in surplus_points)
    assert all(point.power_kw == pytest.approx(100) for point in surplus_points)
    assert all(point.grid_import_kw == 0 for point in surplus_points)
    assert all(point.pv_curtailment_kw == 0 for point in surplus_points)


def _with_demand_limit(dispatch_inputs: DispatchInputs, limit: float) -> DispatchInputs:
    site = dispatch_inputs.site.model_copy(update={"demand_limit_kw": limit})
    return replace(dispatch_inputs, site=site)


def _with_load(dispatch_inputs: DispatchInputs, value: float) -> DispatchInputs:
    for point in dispatch_inputs.load_points:
        point.value = value
    return dispatch_inputs


def test_demand_limit_caps_valley_grid_charging(dispatch_inputs: DispatchInputs) -> None:
    inputs = _with_demand_limit(_with_load(dispatch_inputs, 400), 600)
    result = RuleDispatchEngine().run(inputs)
    assert "demand_limit_kw=600" in result.constraints_applied
    assert all(point.grid_import_kw <= 600 for point in result.schedule)
    # 谷段确实在充电，且被需量截断到不超过 200 kW（400 负荷 + 200 充电 = 600 上限）
    valley_charges = [p for p in result.schedule if p.action == "charge" and p.tier == "valley"]
    assert valley_charges
    assert all(point.power_kw <= 200 + 1e-6 for point in valley_charges)
    assert any(point.power_kw == pytest.approx(200) for point in valley_charges)


def test_no_demand_limit_keeps_previous_behavior(dispatch_inputs: DispatchInputs) -> None:
    inputs = _with_load(dispatch_inputs, 400)
    result = RuleDispatchEngine().run(inputs)
    assert not any("demand_limit_kw" in item for item in result.constraints_applied)
    # 谷段满 PCS 功率充电，电网功率可超过 600
    valley_charges = [p for p in result.schedule if p.action == "charge" and p.tier == "valley"]
    assert valley_charges
    assert any(point.grid_import_kw > 600 for point in result.schedule)
    assert all(point.power_kw == pytest.approx(500) for point in valley_charges[:3])


def test_pv_surplus_charging_ignores_demand_limit(dispatch_inputs: DispatchInputs) -> None:
    for index in range(40, 56):
        dispatch_inputs.pv_points[index].value = 900
    inputs = _with_demand_limit(dispatch_inputs, 100)
    result = RuleDispatchEngine().run(inputs)
    surplus_points = result.schedule[40:56]
    # 余电充电不进电网，即使需量上限只有 100 也照常充
    assert all(point.action == "charge" for point in surplus_points)
    assert all(point.power_kw == pytest.approx(100) for point in surplus_points)
    assert all(point.grid_import_kw == 0 for point in surplus_points)


def test_forecast_mode_actively_shaves_above_target_in_flat_period(dispatch_inputs: DispatchInputs) -> None:
    """Forecast mode may shave a demand spike even when the tariff tier is flat."""
    flat_tariff = TariffRead(
        **TariffCreate(
            tariff_id="T-FLAT",
            name="全平段测试电价",
            tiers=[{"name": "flat", "price": 0.8, "hours": ["00:00-23:45", "23:45-00:00"]}],
        ).model_dump(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    prices = TouProvider(flat_tariff).points(dispatch_inputs.pv_points[0].ts, 96)
    forecasts = [
        MeasurementRead(
            data_type="load_power", ts=p.ts, value=1000 if i == 20 else 400,
            unit="kW", source="forecast"
        ) for i, p in enumerate(dispatch_inputs.load_points)
    ]
    inputs = replace(
        dispatch_inputs,
        price_points=prices,
        load_points=[p.model_copy(update={"value": 1000 if i == 20 else 400}) for i, p in enumerate(dispatch_inputs.load_points)],
        forecast_load_points=forecasts,
        forecast_price_points=prices,
        peak_shaving_target_kw=600,
    )
    result = RuleDispatchEngine().run(inputs)
    spike = result.schedule[20]
    assert spike.action == "discharge"
    assert spike.power_kw == pytest.approx(400, abs=1e-6)
    assert spike.grid_import_kw == pytest.approx(600, abs=1e-6)
    assert "active_peak_shaving_triggered=true" in result.constraints_applied


def test_forecast_mode_reserves_soc_for_future_peak(dispatch_inputs: DispatchInputs) -> None:
    """A future predicted peak prevents an earlier period from draining reserve."""
    forecasts = [
        MeasurementRead(
            data_type="load_power", ts=p.ts, value=1200 if i == 45 else 800,
            unit="kW", source="forecast"
        ) for i, p in enumerate(dispatch_inputs.load_points)
    ]
    inputs = replace(
        dispatch_inputs,
        initial_soc=0.80,
        load_points=[p.model_copy(update={"value": 1200 if i == 45 else 800}) for i, p in enumerate(dispatch_inputs.load_points)],
        forecast_load_points=forecasts,
        forecast_price_points=dispatch_inputs.price_points,
        reserve_soc=0.50,
        peak_shaving_target_kw=1000,
        lookahead_hours=4,
    )
    result = RuleDispatchEngine().run(inputs)
    # Earlier peak is held at or above the configured reserve, and the later
    # spike is still eligible for a discharge action.
    assert result.schedule[40].soc_end >= 0.50 - 1e-6
    assert any(p.action == "discharge" for p in result.schedule[45:50])
    assert "reserve_soc=0.5" in result.constraints_applied


def test_price_spread_must_cover_round_trip_losses(dispatch_inputs: DispatchInputs) -> None:
    prices = [p.model_copy(update={"tier": None, "price": 1.0 if i == 0 else 1.01}) for i, p in enumerate(dispatch_inputs.price_points)]
    inputs = replace(dispatch_inputs, forecast_price_points=prices, price_points=prices)
    result = RuleDispatchEngine().run(inputs)
    assert result.schedule[0].action != "charge"


def test_future_measured_pv_cannot_change_current_decision(dispatch_inputs: DispatchInputs) -> None:
    forecasts = [p.model_copy(update={"value": 1200 if i == 1 else 400}) for i, p in enumerate(dispatch_inputs.load_points)]
    inputs = replace(dispatch_inputs, forecast_load_points=forecasts, peak_shaving_target_kw=600)
    before = RuleDispatchEngine().run(inputs).schedule[0]
    changed_pv = [p.model_copy(update={"value": 9999 if i > 0 else 0}) for i, p in enumerate(inputs.pv_points)]
    after = RuleDispatchEngine().run(replace(inputs, pv_points=changed_pv)).schedule[0]
    assert before.model_dump() == after.model_dump()


def test_reserve_covers_multiple_later_spike_intervals(dispatch_inputs: DispatchInputs) -> None:
    prices = [p.model_copy(update={"tier": "peak", "price": 1}) for p in dispatch_inputs.price_points]
    load = [p.model_copy(update={"value": 1000 if 3 <= i <= 5 else 700}) for i, p in enumerate(dispatch_inputs.load_points)]
    inputs = replace(dispatch_inputs, price_points=prices, load_points=load, forecast_load_points=load,
                     peak_shaving_target_kw=600, initial_soc=.7)
    result = RuleDispatchEngine().run(inputs)
    reserve_energy = 3*400*.25/inputs.site.storage.discharge_eff
    assert result.schedule[0].soc_end*1000 >= 100+reserve_energy-1e-3
