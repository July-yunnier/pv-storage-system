"""站点训练入口的校准保护与工件到预测接口的回归验证。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.domain.forecast.price_features import PriceSeries
from app.domain.forecast.price_model import LgbmQuantileForecaster
from scripts import train_price_model


@pytest.mark.parametrize(
    "cli_args,valid_hours,expected_days",
    [
        ([], 90 * 24, 45),  # 恰好达标时不应关闭默认校准
        ([], 90 * 24 - 1, 0),
        (["--cal-days", "60"], 120 * 24, 60),
        (["--cal-days", "60"], 100 * 24, 0),
        (["--cal-days", "0"], 50 * 24, 0),
    ],
)
def test_training_calibration_uses_valid_hours_in_window(
    monkeypatch, tmp_path, capsys, cli_args, valid_hours, expected_days
):
    # 数据库首尾跨度足够长，但实际窗口中其余小时缺失，不能用跨度判断校准资格。
    ts = pd.date_range("2026-01-01", periods=121 * 24, freq="h", tz="Asia/Shanghai")
    price = np.full(len(ts), np.nan)
    price[:valid_hours] = 0.5
    price[-1] = np.inf  # 非有限值也不能算有效历史
    series = PriceSeries(ts=ts, price=price)
    site = SimpleNamespace(location=SimpleNamespace(timezone="Asia/Shanghai"), tariff_id="T-TEST")
    db = SimpleNamespace(
        get_site=lambda _: site,
        get_tariff=lambda _: object(),
        spot_price_bounds=lambda _: (ts[0].to_pydatetime(), ts[-1].to_pydatetime()),
    )
    received = {}

    class RecordingModel:
        def fit(self, series, **kwargs):
            received.update(kwargs)
            return self

        def save(self, target):
            target.mkdir(parents=True)
            (target / "manifest.json").write_text(json.dumps({"columns": []}), encoding="utf-8")

    monkeypatch.setattr(train_price_model, "Database", lambda _: db)
    monkeypatch.setattr(train_price_model, "build_price_series", lambda *args: (series, None, None))
    monkeypatch.setattr(train_price_model, "LgbmQuantileForecaster", RecordingModel)
    monkeypatch.setenv("PRICE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["train_price_model.py", "--site", "S-TEST", *cli_args])

    assert train_price_model.main() == 0
    assert received["cal_days"] == expected_days
    manifest = json.loads((tmp_path / "S-TEST" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["valid_price_hours"] == valid_hours
    assert manifest["cal_days"] == expected_days
    assert datetime.fromisoformat(manifest["trained_at"]).utcoffset() == timedelta(0)
    requested = manifest["cal_days_requested"]
    stderr = capsys.readouterr().err
    if requested > 0 and expected_days == 0:
        assert "disabling calibration (cal_days=0)" in stderr
    else:
        assert "disabling calibration" not in stderr


def test_negative_cal_days_rejected_before_database_access(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_price_model.py", "--site", "S-TEST", "--cal-days", "-1"])

    def unexpected_database(_):
        pytest.fail("invalid calibration argument must be rejected before opening the database")

    monkeypatch.setattr(train_price_model, "Database", unexpected_database)
    with pytest.raises(SystemExit) as exc:
        train_price_model.main()
    assert exc.value.code == 2
    assert "--cal-days must be >= 0" in capsys.readouterr().err


def test_trained_calibration_reaches_forecast_api_and_legacy_artifact_loads(
    site_client, monkeypatch, tmp_path
):
    lightgbm = pytest.importorskip("lightgbm")

    # 工件加载的 Booster 默认使用全部 OpenMP 线程；测试限定 CPU 预算，覆盖路由线程推理。
    predict = lightgbm.Booster.predict

    def predict_single_thread(self, *args, **kwargs):
        kwargs["num_threads"] = 1
        return predict(self, *args, **kwargs)

    monkeypatch.setattr(lightgbm.Booster, "predict", predict_single_thread)
    db = site_client.app.state.db
    tz = timezone(timedelta(hours=8))
    end = datetime.now(tz).replace(minute=0, second=0, microsecond=0)
    hours = 110 * 24
    rng = np.random.default_rng(42)
    values = 0.5 + 0.08 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    # 合成测试场景：后 45 天波动增加，确保可观测到非零外扩量；不是市场效果验证。
    noise = np.where(np.arange(hours) < 65 * 24, 0.005, 0.15)
    values += rng.normal(size=hours) * noise
    response = site_client.post(
        "/api/v1/sites/S-API/price/spot-prices",
        json={
            "source": "pytest_synthetic",
            "points": [
                {
                    "ts": (end - timedelta(hours=hours - index)).isoformat(),
                    "price_cny_per_kwh": float(value),
                    "quality": "measured",
                }
                for index, value in enumerate(values)
            ],
        },
    )
    assert response.status_code == 200, response.text
    monkeypatch.setenv("PRICE_MODEL_DIR", str(tmp_path / "price_models"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_price_model.py", "--site", "S-API", "--db", db.path, "--rounds", "15"],
    )
    assert train_price_model.main() == 0
    artifact = tmp_path / "price_models" / "S-API"
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cal_days"] == manifest["cal_days_requested"] == 45
    assert manifest["calibration"]["n_cal"] > 0
    calibrated_model = LgbmQuantileForecaster.load(artifact)
    assert calibrated_model.calibrator is not None

    params = {"days": 1, "model": "lgbm", "uncertainty": True}
    calibrated_response = site_client.get("/api/v1/sites/S-API/price/forecast", params=params)
    assert calibrated_response.status_code == 200, calibrated_response.text
    calibrated = calibrated_response.json()
    assert calibrated["model"] == "price_lgbm_quantile"
    assert "保形校准" in calibrated["method_note"]

    # 同一组 LightGBM 权重去掉校准字段，既隔离校准效应，也覆盖旧工件兼容路径。
    legacy_manifest = {key: value for key, value in manifest.items() if key != "calibration"}
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    try:
        assert LgbmQuantileForecaster.load(artifact).calibrator is None
        raw_response = site_client.get("/api/v1/sites/S-API/price/forecast", params=params)
        assert raw_response.status_code == 200, raw_response.text
        raw = raw_response.json()
    finally:
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert raw["model"] == "price_lgbm_quantile"
    assert len(calibrated["points"]) == len(raw["points"]) == 96
    raw_width = np.mean([point["p90"] - point["p10"] for point in raw["points"]])
    calibrated_width = np.mean([point["p90"] - point["p10"] for point in calibrated["points"]])
    assert calibrated_width > raw_width * 1.25
    for before, after in zip(raw["points"], calibrated["points"]):
        assert after["p50"] == pytest.approx(before["p50"])
        assert after["p10"] <= before["p10"] <= before["p50"] <= before["p90"] <= after["p90"]
        assert datetime.fromisoformat(after["ts"]).utcoffset() == timedelta(hours=8)
    assert datetime.fromisoformat(calibrated["issued_at"]).utcoffset() == timedelta(hours=8)
    print(f"synthetic API interval width (CNY/kWh): {raw_width:.6f} -> {calibrated_width:.6f}")
