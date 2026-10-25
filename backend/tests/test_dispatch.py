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


@pytest.mark.parametrize("initial_soc", [0.5, 0.9, 0.95])
@pytest.mark.parametrize("allow_export", [False, True])
def test_pv_surplus_energy_balance(
    dispatch_inputs: DispatchInputs, initial_soc: float, allow_export: bool,
) -> None:
    storage = dispatch_inputs.site.storage.model_copy(update={"allow_grid_export": allow_export})
    inputs = replace(
        dispatch_inputs,
        site=dispatch_inputs.site.model_copy(update={"storage": storage}),
        initial_soc=initial_soc,
        pv_points=[dispatch_inputs.pv_points[0].model_copy(update={"value": 1000})],
        load_points=[dispatch_inputs.load_points[0].model_copy(update={"value": 100})],
        price_points=dispatch_inputs.price_points[:1],
    )
    point = RuleDispatchEngine().run(inputs).schedule[0]
    charge = min(500, (storage.soc_max - initial_soc) * 1000 / storage.charge_eff / 0.25)
    remainder = 900 - charge
    assert point.charge_from_pv_kw == pytest.approx(charge)
    assert point.pv_curtailment_kw == pytest.approx(0 if allow_export else remainder)
    assert point.grid_export_kw == pytest.approx(remainder if allow_export else 0)
    assert point.grid_import_kw == 0
    assert point.pv_power_kw == pytest.approx(
        point.load_power_kw + point.charge_from_pv_kw + point.grid_export_kw + point.pv_curtailment_kw
    )


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
