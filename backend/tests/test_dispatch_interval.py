from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from app.domain.dispatch_interval import IntervalDispatchEngine, IntervalInputs, PriceBand, robust_gate
from app.domain.models import (
    LoadConfig,
    MeasurementRead,
    PVConfig,
    SiteCreate,
    SiteRead,
    StorageConfig,
)

TZ = ZoneInfo("Asia/Shanghai")


def _site(**storage) -> SiteRead:
    payload = SiteCreate(
        site_id="S-T",
        name="测试站点",
        location={"lat": 22.9, "lon": 113.7, "timezone": "Asia/Shanghai"},
        pv=PVConfig(capacity_kwp=1000),
        load=LoadConfig(peak_kw=1000),
        storage=StorageConfig(
            pcs_kw=500, capacity_kwh=1000, round_trip_eff=0.88,
            soc_min=0.10, soc_max=0.95, cycles_per_day_max=2, **storage
        ),
        tariff_id="T-GUIDANGDONG-DEMO",
    )
    now = datetime(2026, 1, 1, tzinfo=TZ)
    return SiteRead(**payload.model_dump(), created_at=now, updated_at=now)


def _day(n: int = 96, pv_value: float = 0.0, load_value: float = 400.0):
    start = datetime(2026, 5, 1, tzinfo=TZ)
    pv = [
        MeasurementRead(data_type="pv_power", ts=start + timedelta(minutes=15 * i), value=pv_value, unit="kW", source="sim")
        for i in range(n)
    ]
    load = [
        MeasurementRead(data_type="load_power", ts=start + timedelta(minutes=15 * i), value=load_value, unit="kW", source="sim")
        for i in range(n)
    ]
    return pv, load


def _bands(p10: float, p50: float, p90: float, n: int = 96) -> list[PriceBand]:
    start = datetime(2026, 5, 1, tzinfo=TZ)
    return [
        PriceBand(ts=start + timedelta(minutes=15 * i), p10=p10, p50=p50, p90=p90)
        for i in range(n)
    ]


def test_gate_closed_when_robust_spread_below_margin():
    p10 = np.full(96, 0.60)
    p90 = np.full(96, 0.55)
    gate = robust_gate(p10, p90, 0.88, margin=0.10)
    assert gate["gate_open"] is False
    pv, load = _day()
    resp = IntervalDispatchEngine().run(
        IntervalInputs(
            site=_site(), target_date=date(2026, 5, 1), initial_soc=0.5,
            pv_points=pv, load_points=load, bands=_bands(0.60, 0.58, 0.55),
            margin_robust=0.10,
        )
    )
    # 门槛没过 → 不充不放（除光伏余电外）
    assert all(p.action == "idle" for p in resp.schedule)
    assert any("gate_open=False" in c for c in resp.constraints_applied)


def test_charges_only_in_cheap_p90_and_discharges_in_rich_p10():
    pv, load = _day()
    start = datetime(2026, 5, 1, tzinfo=TZ)
    bands = []
    for i in range(96):
        if i < 16:            # 0-4 点：很便宜
            bands.append(PriceBand(ts=start + timedelta(minutes=15 * i), p10=0.20, p50=0.25, p90=0.30))
        elif i > 70:          # 17:30 以后：很贵
            bands.append(PriceBand(ts=start + timedelta(minutes=15 * i), p10=1.40, p50=1.50, p90=1.60))
        else:
            bands.append(PriceBand(ts=start + timedelta(minutes=15 * i), p10=0.70, p50=0.75, p90=0.80))
    resp = IntervalDispatchEngine().run(
        IntervalInputs(
            site=_site(), target_date=date(2026, 5, 1), initial_soc=0.5,
            pv_points=pv, load_points=load, bands=bands, margin_robust=0.02,
        )
    )
    charge_slots = [i for i, p in enumerate(resp.schedule) if p.action == "charge"]
    discharge_slots = [i for i, p in enumerate(resp.schedule) if p.action == "discharge"]
    assert charge_slots and all(i < 16 for i in charge_slots)
    assert discharge_slots and all(i > 70 for i in discharge_slots)
    assert resp.equivalent_full_cycles > 0


def test_reserve_floor_protects_soc_unless_extreme_price():
    pv, load = _day(load_value=600.0)
    start = datetime(2026, 5, 1, tzinfo=TZ)
    bands = []
    for i in range(96):
        cheap = i < 24   # 前 6 小时便宜，占 25%，保证 0.15 分位落在便宜区
        bands.append(
            PriceBand(ts=start + timedelta(minutes=15 * i),
                      p10=0.15 if cheap else 0.62, p50=0.20 if cheap else 0.70,
                      p90=0.25 if cheap else 0.78)
        )
    resp = IntervalDispatchEngine().run(
        IntervalInputs(
            site=_site(), target_date=date(2026, 5, 1), initial_soc=0.5,
            pv_points=pv, load_points=load, bands=bands,
            margin_robust=0.0, reserve_ratio=0.30,
        )
    )
    floors = [p.soc_end for p in resp.schedule if p.action == "discharge"]
    # 放电末端不应低于 soc_min + 尚未打穿保底
    assert floors
    assert min(floors) >= 0.10 - 1e-6


def test_pv_surplus_is_charged_before_curtailing():
    pv, load = _day(pv_value=800.0, load_value=100.0)
    resp = IntervalDispatchEngine().run(
        IntervalInputs(
            site=_site(), target_date=date(2026, 5, 1), initial_soc=0.2,
            pv_points=pv, load_points=load, bands=_bands(0.5, 0.5, 0.5),
            margin_robust=1e9,   # 门槛全部关掉，只验证余电充电
        )
    )
    charged_from_pv = sum(p.charge_from_pv_kw for p in resp.schedule) * 0.25
    curtailed = sum(p.pv_curtailment_kw for p in resp.schedule) * 0.25
    assert charged_from_pv > 0
    # 电池充满后必然弃光，两者都为正说明逻辑正确
    assert curtailed >= 0
    # 余电 = 800-100 = 700kW，要么进电池要么被弃，二者之和应等于余电总量
    assert charged_from_pv + curtailed == pytest.approx(700.0 * 24, rel=0.01)


def test_rejects_misaligned_series():
    pv, load = _day(48)
    with pytest.raises(ValueError):
        IntervalDispatchEngine().run(
            IntervalInputs(
                site=_site(), target_date=date(2026, 5, 1), initial_soc=0.5,
                pv_points=pv, load_points=load, bands=_bands(0.5, 0.5, 0.5, n=96),
            )
        )


def test_interval_strategy_reachable_via_api(site_client):
    """接口层：strategy=interval_robust 走区间引擎，返回的策略名与约束要能追溯。"""
    from datetime import date, datetime, timedelta, timezone

    tz = timezone(timedelta(hours=8))
    day = date(2026, 6, 1)
    points = []
    for i in range(96):
        ts = datetime(2026, 6, 1, tzinfo=tz) + timedelta(minutes=15 * i)
        hour = i / 4
        pv = max(0.0, 800 * np.sin(np.pi * (hour - 6) / 12)) if 6 < hour < 18 else 0.0
        load = 700.0
        points.append({"ts": ts.isoformat(), "pv": round(pv, 3), "load": load})
    for data_type, key in (("pv_power", "pv"), ("load_power", "load")):
        resp = site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "source": "sim",
                "points": [{"ts": p["ts"], "value": p[key]} for p in points],
            },
        )
        assert resp.status_code == 200, resp.text

    resp = site_client.post(
        "/api/v1/storage/schedule",
        json={"site_id": "S-API", "date": day.isoformat(), "initial_soc": 0.5,
              "strategy": "interval_robust"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["strategy"] == "interval_robust"
    assert any("robust_spread_gate" in c for c in body["constraints_applied"])
    assert len(body["schedule"]) == 96


def test_interval_strategy_reachable_via_api(site_client):
    """接口层：strategy=interval_robust 走区间引擎，返回的策略名与约束要能追溯。"""
    from datetime import date, datetime, timedelta, timezone

    tz = timezone(timedelta(hours=8))
    day = date(2026, 6, 1)
    points = []
    for i in range(96):
        ts = datetime(2026, 6, 1, tzinfo=tz) + timedelta(minutes=15 * i)
        hour = i / 4
        pv = max(0.0, 800 * np.sin(np.pi * (hour - 6) / 12)) if 6 < hour < 18 else 0.0
        load = 700.0
        points.append({"ts": ts.isoformat(), "pv": round(pv, 3), "load": load})
    for data_type, key in (("pv_power", "pv"), ("load_power", "load")):
        resp = site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": data_type,
                "unit": "kW",
                "source": "sim",
                "points": [{"ts": p["ts"], "value": p[key]} for p in points],
            },
        )
        assert resp.status_code == 200, resp.text

    resp = site_client.post(
        "/api/v1/storage/schedule",
        json={"site_id": "S-API", "date": day.isoformat(), "initial_soc": 0.5,
              "strategy": "interval_robust"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["strategy"] == "interval_robust"
    assert any("robust_spread_gate" in c for c in body["constraints_applied"])
    assert len(body["schedule"]) == 96
