from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.forecast.price_ingest import aggregate_hourly, detect_unit, normalize_spot


def _spot_frame(n: int = 96, base: float = 380.0) -> pd.DataFrame:
    ts = pd.date_range("2026-03-01 00:00", periods=n, freq="15min")
    prices = base + 60 * np.sin(np.arange(n) / 8.0)
    return pd.DataFrame({"时间": ts.astype(str), "出清价": prices})


def test_detect_unit_by_magnitude():
    assert detect_unit(np.array([380.0, 420.0]))[0] == "CNY/MWh"
    assert abs(detect_unit(np.array([380.0, 420.0]))[1] - 0.001) < 1e-12
    assert detect_unit(np.array([0.38, 0.42]))[0] == "CNY/kWh"
    assert detect_unit(np.array([0.38, 0.42]))[1] == 1.0


def test_normalize_converts_unit_and_sets_timezone():
    frame, report = normalize_spot(_spot_frame(), tz="Asia/Shanghai")
    assert report.unit_detected == "CNY/MWh"
    assert report.scale_applied == 0.001
    assert abs(frame["price"].mean() - 0.38) < 0.05
    assert str(frame["ts"].dt.tz) == "Asia/Shanghai"
    assert report.rows_out == report.rows_in


def test_negative_prices_are_kept():
    frame = _spot_frame(48)
    frame.loc[10, "出清价"] = -15.0   # 元/MWh 的负电价
    clean, report = normalize_spot(frame)
    assert report.negative_points == 1
    assert clean["price"].min() < 0
    assert any("negative" in w for w in report.warnings)


def test_duplicate_timestamps_removed_keeping_last():
    frame = _spot_frame(24)
    dup = frame.iloc[[5]].copy()
    dup["出清价"] = 999.0
    merged = pd.concat([frame, dup], ignore_index=True)
    clean, report = normalize_spot(merged)
    assert report.duplicates_removed == 1
    # 时间序排序后保留同一时间戳的最后一条
    target = pd.Timestamp(frame.loc[5, "时间"]).tz_localize("Asia/Shanghai")
    kept = clean.loc[clean["ts"] == target, "price"]
    assert len(kept) == 1
    assert abs(float(kept.iloc[0]) - 0.999) < 1e-9


def test_hourly_aggregation_is_mean_not_endpoint():
    frame, _ = normalize_spot(_spot_frame(8), unit="CNY/kWh")
    # 8 个 15 分钟点 = 2 小时
    hourly = aggregate_hourly(frame)
    assert len(hourly) == 2
    first_hour = frame["price"].iloc[:4].mean()
    assert abs(hourly["price"].iloc[0] - first_hour) < 1e-9


def test_missing_column_raises_clear_error():
    frame = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    try:
        normalize_spot(frame)
    except ValueError as exc:
        assert "timestamp" in str(exc) or "price" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_explicit_unit_override():
    frame, report = normalize_spot(_spot_frame(16), unit="CNY/kWh")
    assert report.scale_applied == 1.0
    assert frame["price"].mean() > 300  # 没有做换算
