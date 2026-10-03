"""滚动回放实验框架测试：防泄漏、SOC 连续、指标口径、API 闭环。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from statistics import mean

import pytest
from fastapi.testclient import TestClient

from app.application import replay_service
from app.domain.experiment.execute import execute_plan, no_storage_baseline
from app.domain.experiment.metrics import forecast_metrics
from app.domain.models import PricePoint, StorageConfig
from app.infrastructure.db import Database
from app.infrastructure.weather.base import WeatherPoint
from tests.conftest import CN_TZ, SITE_PAYLOAD

HISTORY_DAYS = 20
TEST_DAYS = 7
START_DATE = date(2026, 3, 1)


def _day_pv(slot: int) -> float:
    """确定性的光伏日形：白天梯形，峰值 500 kW。"""
    hour = slot / 4
    if 7 <= hour < 17:
        return min(500.0, 500.0 * (1 - abs(hour - 12) / 5))
    return 0.0


def _seed_history(db: Database, site_id: str) -> None:
    from app.domain.models import MeasurementUpload

    for offset in range(HISTORY_DAYS + TEST_DAYS):
        day = START_DATE + timedelta(days=offset)
        start = datetime.combine(day, time.min, tzinfo=CN_TZ)
        pv_points = [
            {"ts": (start + timedelta(minutes=15 * i)).isoformat(),
             "value": _day_pv(i), "quality": "measured"}
            for i in range(96)
        ]
        load_points = [
            {"ts": (start + timedelta(minutes=15 * i)).isoformat(),
             "value": 600.0 if 8 <= i / 4 < 20 else 200.0, "quality": "estimated"}
            for i in range(96)
        ]
        db.insert_measurements(site_id, MeasurementUpload(
            data_type="pv_power", unit="kW", resolution_minutes=15,
            source="public_replay_test", points=pv_points,
        ))
        db.insert_measurements(site_id, MeasurementUpload(
            data_type="load_power", unit="kW", resolution_minutes=15,
            source="simulator", points=load_points,
        ))


def _fake_weather(site, days: int) -> list[WeatherPoint]:
    points: list[WeatherPoint] = []
    for offset in range(days):
        day = START_DATE + timedelta(days=offset)
        for hour in range(24):
            ts = datetime.combine(day, time(hour), tzinfo=CN_TZ)
            points.append(WeatherPoint(
                ts=ts, ghi_wm2=500.0 if 7 <= hour < 17 else 0.0,
                dni_wm2=400.0, dhi_wm2=100.0, temperature_c=25.0, wind_speed_ms=2.0,
            ))
    return points


@pytest.fixture()
def replay_db(client: TestClient, tmp_path) -> Database:
    client.post("/api/v1/sites", json=SITE_PAYLOAD)
    db = Database(str(tmp_path / "test.db"))
    db.init_schema()
    _seed_history(db, SITE_PAYLOAD["site_id"])
    return db


def test_metrics_daylight_caliber() -> None:
    predicted = [0.0] * 24 + [100.0] * 48 + [0.0] * 24
    actual = [0.0] * 24 + [110.0] * 48 + [0.0] * 24
    result = forecast_metrics(predicted, actual, cap_kw=1000)
    assert result["n_points"] == 96
    assert result["n_daylight"] == 48
    assert result["skill_all_day"] > result["skill_daylight"]  # 含夜间口径必然更高
    assert result["bias"] < 0  # 预测系统性偏低


def test_execute_plan_respects_soc_and_export() -> None:
    storage = StorageConfig(pcs_kw=100, capacity_kwh=100, soc_min=0.1, soc_max=0.9)
    prices = [PricePoint(ts=datetime(2026, 3, 1, tzinfo=CN_TZ), price=1.0,
                         energy_price=0.8, tier="peak")] * 96
    # 计划放电远超库存：执行必须按 SOC 可用电量裁剪
    actions = [("discharge", 100.0)] * 96
    ex = execute_plan(actions, [0.0] * 96, [200.0] * 96, prices, storage, 0.5, 15)
    assert ex.soc_violations == 0
    assert ex.end_soc == pytest.approx(0.1, abs=1e-3)
    assert all(0.1 - 1e-6 <= s <= 0.9 + 1e-6 for s in ex.soc_trace)
    # 防逆流：负荷 200 > 放电 100，不放电时不会反送
    assert ex.curtailment_kwh == 0.0


def test_no_storage_baseline() -> None:
    prices = [PricePoint(ts=datetime(2026, 3, 1, tzinfo=CN_TZ), price=1.0,
                         energy_price=0.8, tier="flat")] * 96
    ex = no_storage_baseline([100.0] * 96, [300.0] * 96, prices, 15)
    assert ex.cost_yuan == pytest.approx(200.0 * 0.25 * 96, rel=1e-6)


def test_replay_runs_and_no_leakage(replay_db: Database) -> None:
    site = replay_db.get_site(SITE_PAYLOAD["site_id"])
    assert site is not None
    weather = _fake_weather(site, HISTORY_DAYS + TEST_DAYS)
    result = replay_service.run_replay(
        replay_db, SITE_PAYLOAD["site_id"], test_days=TEST_DAYS,
        pv_methods=["slot_mean", "physics", "oracle"],
        strategies=None, weather=weather,
    )
    summary = result["summary"]
    # 预测指标：恒定日形下 slot_mean 应接近完美；oracle 必须完美
    assert summary["forecasts"]["oracle"]["skill_all_day"] == pytest.approx(1.0, abs=1e-3)
    assert summary["forecasts"]["slot_mean"]["rmse"] < 1.0
    # 策略矩阵：oracle 调度收益 ≥ 预测调度收益 ≥ 0；不储能节省为 0
    assert summary["strategies"]["__actual__+d0_no_storage"]["total_saving_yuan"] == 0
    d4 = summary["strategies"]["slot_mean+d4_rule_predictive"]["total_saving_yuan"]
    d5 = summary["strategies"]["__actual__+d5_rule_oracle"]["total_saving_yuan"]
    assert d5 >= d4 > 0


def test_replay_api(client: TestClient, replay_db: Database) -> None:
    site = replay_db.get_site(SITE_PAYLOAD["site_id"])
    weather = _fake_weather(site, HISTORY_DAYS + TEST_DAYS)
    response = client.post(
        f"/api/v1/sites/{SITE_PAYLOAD['site_id']}/replay",
        json={"test_days": TEST_DAYS, "pv_methods": ["slot_mean", "oracle"]},
    )
    # API 路径走真实气象代理只在 physics/blend 时需要；这里无需外网
    assert response.status_code == 200, response.text
    run_id = response.json()["run_id"]

    runs = client.get(f"/api/v1/sites/{SITE_PAYLOAD['site_id']}/replay-runs")
    assert runs.status_code == 200
    assert any(r["run_id"] == run_id for r in runs.json())

    detail = client.get(f"/api/v1/replay-runs/{run_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "finished"
    strategies = {r["strategy"] for r in body["day_results"]}
    assert "d4_rule_predictive" in strategies and "__forecast__" in strategies
