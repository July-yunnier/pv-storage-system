"""MQ-RNN 主线移植的测试：结构、边界遮罩、防泄漏、冒烟训练。

防泄漏是协议红线：predict 只允许读 origin 及之前的数据，fit 由调用方截断，
这里用"篡改未来数据，预测必须逐点不变"守住。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from app.domain.forecast.price_mqrnn import (  # noqa: E402
    FEATURE_NAMES,
    LOOKBACK,
    WEEK_LAG,
    MqRnn,
    calendar_features,
    observed_features,
)


def _synthetic(hours: int = 1100, seed: int = 7) -> tuple[pd.DatetimeIndex, np.ndarray]:
    ts = pd.date_range("2013-01-01", periods=hours, freq="h")
    hour = ts.hour.to_numpy(dtype="float64")
    dow = ts.dayofweek.to_numpy(dtype="float64")
    base = 40 + 12 * np.sin(2 * np.pi * hour / 24) + 6 * np.sin(2 * np.pi * dow / 7)
    noise = np.random.default_rng(seed).normal(0, 2, hours)
    return ts, base + noise


def test_calendar_features_shape_and_range():
    ts = pd.date_range("2013-03-01", periods=48, freq="h")
    feats = calendar_features(ts)
    assert feats.shape == (48, len(FEATURE_NAMES))
    assert np.all(np.abs(feats[:, :6]) <= 1.0 + 1e-9)
    assert set(np.unique(feats[:, 6])).issubset({0.0, 1.0})


def test_observed_features_lag_alignment():
    ts = pd.date_range("2013-01-01", periods=400, freq="h")
    price = np.arange(400, dtype="float64")
    feats = observed_features(ts, price)
    assert feats.shape == (400 - WEEK_LAG, 3 + len(FEATURE_NAMES))
    # 第 0 行对应原序列第 168 点：日滞后=price[144]，周滞后=price[0]
    assert feats[0, 0] == 168
    assert feats[0, 1] == 144
    assert feats[0, 2] == 0


def test_blocks_mask_boundary_targets():
    ts, price = _synthetic(hours=600)
    model = MqRnn(horizon=24)
    enc, fut, y, mask = model._blocks(ts, price)
    n = len(price) - WEEK_LAG
    assert enc.shape[1] == LOOKBACK
    assert fut.shape[2:] == (24, len(FEATURE_NAMES))
    # 最后一个块：越界目标必须被 mask 掉，且参与损失的位置 target 都在界内
    last_mask = mask[-1]
    start = (len(mask) - 1) * LOOKBACK
    for j, pos in enumerate(range(start, start + LOOKBACK)):
        for k in range(24):
            expected = (pos < n - 1) and (pos + k + 1 < n)
            assert bool(last_mask[j, k]) == expected
    # 被遮罩的标签不参与损失：改动遮罩区标签，损失不变
    pred = torch.zeros(y.shape[0], y.shape[1], 24, 3)
    t1 = torch.as_tensor(y.copy())
    loss1 = model._pinball_masked(pred, t1, torch.as_tensor(mask))
    y2 = y.copy()
    y2[~mask] = 1e6
    loss2 = model._pinball_masked(pred, torch.as_tensor(y2), torch.as_tensor(mask))
    assert float(loss1) == pytest.approx(float(loss2))


def test_fit_predict_smoke_and_quantile_order():
    ts, price = _synthetic()
    model = MqRnn(horizon=24)
    model.fit(ts, price, None, horizon=24, max_epochs=2, val_days=3, seed=42)
    origin = len(ts) - 25
    out = model.predict(ts, price, None, origin)
    assert len(out.ts) == 24
    assert out.ts[0] == ts[origin] + pd.Timedelta("1h")
    assert np.all(out.p10 <= out.p50 + 1e-9)
    assert np.all(out.p50 <= out.p90 + 1e-9)
    assert np.isfinite(model.best_val_pinball)


def test_predict_does_not_look_past_origin():
    """篡改 origin 之后的全部价格，预测必须逐点不变（防泄漏红线）。"""
    ts, price = _synthetic()
    model = MqRnn(horizon=24)
    model.fit(ts, price, None, horizon=24, max_epochs=2, val_days=3, seed=42)
    origin = len(ts) - 200
    before = model.predict(ts, price, None, origin)
    poisoned = price.copy()
    poisoned[origin + 1 :] = 9999.0
    after = model.predict(ts, poisoned, None, origin)
    np.testing.assert_array_equal(before.p50, after.p50)
    np.testing.assert_array_equal(before.p10, after.p10)
    np.testing.assert_array_equal(before.p90, after.p90)


def test_fit_rejects_non_finite():
    ts, price = _synthetic()
    price[500] = np.nan
    model = MqRnn(horizon=24)
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(ts, price, None, max_epochs=1, val_days=3)


def test_registry_entry():
    from app.application.algorithm_registry import get

    entry = get("price_mqrnn")
    assert entry is not None and entry.target == "price"
    assert entry.kind == "ml"
