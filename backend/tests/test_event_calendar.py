from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.domain.forecast.event_calendar import EventCalendar, MarketEvent, load_calendar

REPO_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "events.guangdong.yaml"


def _calendar() -> EventCalendar:
    return EventCalendar(
        events=[
            MarketEvent(kind="maintenance", start=date(2026, 3, 2), end=date(2026, 3, 18), severity=0.8),
            MarketEvent(kind="cold_wave", start=date(2026, 12, 14), end=date(2026, 12, 19), severity=0.9),
        ],
        name="test",
    )


def test_features_shape_and_flags():
    cal = _calendar()
    ts = pd.date_range("2026-03-01", periods=24 * 3, freq="h")
    feats = cal.features(ts)
    assert set(feats) >= {"event_pressure", "in_maintenance", "days_to_maintenance", "cold_wave"}
    for value in feats.values():
        assert len(value) == len(ts)
    # 3-02 起进入检修
    assert feats["in_maintenance"][24] == 1.0
    assert feats["in_maintenance"][0] == 0.0
    # 检修前一天，距下一次检修应为 1 天
    assert feats["days_to_maintenance"][0] == 1.0
    assert feats["event_pressure"][24] > 0


def test_days_to_maintenance_zero_inside_and_nan_when_far():
    cal = _calendar()
    ts = pd.date_range("2026-03-05", periods=24, freq="h")
    feats = cal.features(ts)
    assert (feats["days_to_maintenance"] == 0.0).all()
    far = cal.features(pd.date_range("2026-08-01", periods=24, freq="h"))
    assert np.isnan(far["days_to_maintenance"]).all()


def test_cold_wave_severity_is_preserved():
    cal = _calendar()
    feats = cal.features(pd.date_range("2026-12-15", periods=24, freq="h"))
    assert (feats["cold_wave"] == 0.9).all()
    assert (feats["heat_wave"] == 0.0).all()


def test_no_calendar_returns_nan_columns():
    from app.domain.forecast.price_features import PriceSeries, build_samples

    ts = pd.date_range("2026-01-01", periods=400, freq="h")
    rng = np.random.default_rng(1)
    series = PriceSeries(ts=ts, price=60 + rng.normal(0, 3, 400))
    x, _, _, _ = build_samples(series, origins=np.arange(200, 260), horizon=6, min_origin=0)
    assert "event_pressure" in x.columns
    assert x["event_pressure"].isna().all()


def test_event_features_flow_into_design_matrix():
    from app.domain.forecast.price_features import PriceSeries, build_samples

    cal = _calendar()
    ts = pd.date_range("2026-02-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(2)
    series = PriceSeries(
        ts=ts,
        price=60 + rng.normal(0, 3, len(ts)),
        event=cal.features(ts),
    )
    x, _, target_ts, _ = build_samples(series, origins=np.arange(200, 300), horizon=6, min_origin=0)
    assert x["event_pressure"].notna().any()
    assert x["fuel_index_norm"].isna().all()  # 未配燃料序列
    assert set(target_ts.minute) == {0}


@pytest.mark.skipif(not REPO_CONFIG.exists(), reason="示例事件配置不存在")
def test_example_config_loads_and_matches_features():
    calendar = load_calendar(REPO_CONFIG)
    assert calendar is not None
    summary = calendar.summary()
    assert summary["events"] >= 3
    assert "maintenance" in summary["by_kind"]
    feats = calendar.features(pd.date_range("2026-07-20", periods=48, freq="h"))
    assert feats["heat_wave"].max() > 0
    assert np.isfinite(feats["fuel_index_norm"]).any()


def test_unknown_kind_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "name: bad\nevents:\n  - kind: earthquake\n    start: 2026-01-01\n    end: 2026-01-02\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        EventCalendar.from_yaml(path)
