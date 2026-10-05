"""负荷高级模型特征构造层测试：形状、缺口填补、服务接线、真实工件端到端。"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.application import forecast_service
from app.application.load_features import build_advanced_load_inputs
from app.domain.forecast.result import ForecastModel, ForecastPoint, TargetForecast
from app.domain.models import MeasurementUpload, SiteCreate
from app.infrastructure.db import Database
from tests.conftest import CN_TZ, day_points

TODAY = datetime.now(CN_TZ).date()
TARGET = TODAY + timedelta(days=1)
HISTORY_DAYS = 30


def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "features.db"))
    db.create_site(
        SiteCreate(
            site_id="S-FEAT",
            name="特征测试站点",
            location={"lat": 22.94, "lon": 113.75, "timezone": "Asia/Shanghai"},
            pv={"capacity_kwp": 1000},
            load={"peak_kw": 800},
            storage={"pcs_kw": 500, "capacity_kwh": 1000},
        )
    )
    return db


def _seed_load(db: Database, site_id: str, days: int, value: float = 600.0) -> None:
    for offset in range(days, 0, -1):
        day = TODAY - timedelta(days=offset)
        base = datetime.combine(day, time.min, tzinfo=CN_TZ)
        db.insert_measurements(
            site_id,
            MeasurementUpload(
                data_type="load_power",
                points=[
                    {"ts": (base + timedelta(minutes=15 * i)).isoformat(),
                     "value": value + i % 4}
                    for i in range(96)
                ],
            ),
        )


def test_builder_shapes_and_mock_weather(tmp_path) -> None:
    db = _make_db(tmp_path)
    _seed_load(db, "S-FEAT", HISTORY_DAYS)
    site = db.get_site("S-FEAT")
    inputs = build_advanced_load_inputs(db, site, TARGET)

    assert len(inputs.history_features) == HISTORY_DAYS * 96
    assert all(len(row) == 6 for row in inputs.history_features)
    assert len(inputs.history_mes_labels) == HISTORY_DAYS * 96
    assert len(inputs.target_features) == 96
    assert all(len(row) == 5 for row in inputs.target_features)
    assert len(inputs.target_mes_labels) == 96
    # mock 气象：恒温 25°C、恒湿 60%
    assert {round(row[1], 1) for row in inputs.history_features} == {25.0}
    assert {round(row[2], 1) for row in inputs.history_features} == {60.0}
    assert {round(row[0], 1) for row in inputs.target_features} == {25.0}
    assert {round(row[1], 1) for row in inputs.target_features} == {60.0}
    # 历史首行负荷 = 种子值（600 + slot%4）
    assert inputs.history_features[0][0] == pytest.approx(600.0)


def test_builder_gap_fill_with_partial_history(tmp_path) -> None:
    db = _make_db(tmp_path)
    _seed_load(db, "S-FEAT", 10, value=500.0)  # 只有 10 天，其余 20 天补缺
    site = db.get_site("S-FEAT")
    inputs = build_advanced_load_inputs(db, site, TARGET)
    assert len(inputs.history_features) == HISTORY_DAYS * 96
    # 无数据天用同刻均值填补：slot0 均值 = 500
    oldest_day_slot0 = inputs.history_features[0][0]
    assert oldest_day_slot0 == pytest.approx(500.0)


def test_builder_requires_history(tmp_path) -> None:
    db = _make_db(tmp_path)
    site = db.get_site("S-FEAT")
    with pytest.raises(ValueError, match="负荷历史"):
        build_advanced_load_inputs(db, site, TARGET)


def test_builder_is_workday_and_labels(tmp_path) -> None:
    db = _make_db(tmp_path)
    _seed_load(db, "S-FEAT", 5)
    site = db.get_site("S-FEAT")
    inputs = build_advanced_load_inputs(db, site, TARGET)
    expected_workday = 1.0 if TARGET.weekday() < 5 else 0.0
    assert {row[4] for row in inputs.target_features} == {expected_workday}  # is_workday 列
    expected_label = 1 if TARGET.weekday() < 5 else 7  # 白天 day_shift / 全天 weekend_base
    midday_label = inputs.target_mes_labels[12 * 4]
    assert midday_label == expected_label


def _stub_forecast(target_date: date) -> TargetForecast:
    base = datetime.combine(target_date, time.min, tzinfo=CN_TZ)
    return TargetForecast(
        target="load_power",
        unit="kW",
        model=ForecastModel(name="stub_advanced", version="v0"),
        points=[
            ForecastPoint(ts=base + timedelta(minutes=15 * i), value=700.0)
            for i in range(96)
        ],
    )


def test_api_advanced_auto_builds_inputs(site_client: TestClient, monkeypatch) -> None:
    """不显式传 advanced_load_input：服务端自动构造特征后调模型。"""
    captured: dict = {}

    def stub_predict(inputs, target_date, tz, model_path=None):
        captured["history_len"] = len(inputs.history_features)
        captured["target_len"] = len(inputs.target_features)
        return _stub_forecast(target_date)

    monkeypatch.setattr(forecast_service, "predict_from_artifact", stub_predict)
    for offset in range(5, 0, -1):
        day = TODAY - timedelta(days=offset)
        response = site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": "load_power",
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(datetime.combine(day, time.min, tzinfo=CN_TZ), 600),
            },
        )
        assert response.status_code == 200, response.text

    response = site_client.post(
        "/api/v1/forecast",
        json={
            "site_id": "S-API",
            "targets": ["load_power"],
            "horizon_hours": 24,
            "options": {"load_model": "advanced", "include_uncertainty": False},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["model"]["name"] == "stub_advanced"
    assert len(result["points"]) == 96
    assert captured["history_len"] == HISTORY_DAYS * 96
    assert captured["target_len"] == 96


def test_api_advanced_with_real_artifact(site_client: TestClient, monkeypatch) -> None:
    """真实工件端到端：需要 torch 与 LOAD_MODEL_PATH 工件，缺环境时跳过。"""
    pytest.importorskip("torch")
    artifact = (
        Path(__file__).resolve().parents[1]
        / "data" / "training_artifacts" / "load_mes_attention.pt"
    )
    if not artifact.is_file():
        pytest.skip("模型工件未随环境提供")
    monkeypatch.setenv("LOAD_MODEL_PATH", str(artifact))

    for offset in range(5, 0, -1):
        day = TODAY - timedelta(days=offset)
        response = site_client.post(
            "/api/v1/sites/S-API/measurements",
            json={
                "data_type": "load_power",
                "unit": "kW",
                "resolution_minutes": 15,
                "points": day_points(datetime.combine(day, time.min, tzinfo=CN_TZ), 600),
            },
        )
        assert response.status_code == 200, response.text

    response = site_client.post(
        "/api/v1/forecast",
        json={
            "site_id": "S-API",
            "targets": ["load_power"],
            "horizon_hours": 24,
            "options": {"load_model": "advanced", "include_uncertainty": False},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["model"]["name"] == "load_mes_multitask_attention"
    values = [p["value"] for p in result["points"]]
    assert len(values) == 96
    assert all(v >= 0 for v in values)
