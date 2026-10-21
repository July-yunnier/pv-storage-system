from __future__ import annotations

import numpy as np

from app.domain.forecast.price_calibration import ConformalCalibrator, calibrate_output


def _toy(n: int = 2000, seed: int = 3, width: float = 8.0, err_std: float = 12.0):
    """构造"区间系统性偏窄"的场景：中心给对，但宽度不够。"""
    rng = np.random.default_rng(seed)
    center = rng.normal(100, 20, n)
    y = center + rng.normal(0, err_std, n)   # 实际误差远大于区间半宽
    p10 = center - width
    p90 = center + width
    days = np.tile(np.arange(1, 25), n // 24 + 1)[:n]
    return y, p10, p90, days


def test_calibration_widens_interval_to_nominal_coverage():
    y, p10, p90, days = _toy()
    split = 1400
    cal = ConformalCalibrator(alpha=0.2).fit(y[:split], p10[:split], p90[:split], days[:split])
    lo, hi = cal.apply(p10[split:], p90[split:], days[split:])
    coverage = float(np.mean((y[split:] >= lo) & (y[split:] <= hi)))
    # 保形校准只保证边际覆盖，留 4 个点容差
    assert 0.76 <= coverage <= 0.86, coverage
    assert cal.coverage_before < 0.6
    assert (lo <= p10[split:]).all() and (hi >= p90[split:]).all()


def test_calibration_never_shrinks_interval():
    y, p10, p90, days = _toy(seed=11)
    cal = ConformalCalibrator(alpha=0.2).fit(y, p10, p90, days)
    assert (cal.lo >= 0).all() and (cal.hi >= 0).all()


def test_per_horizon_offsets_differ():
    rng = np.random.default_rng(5)
    days = np.tile(np.arange(1, 25), 80)
    y = rng.normal(0, 1, days.size)
    center = y + rng.normal(0, 0.6, days.size)   # 预测中心本身带误差
    # 远端（h 越大）区间越窄 → 需要的外扩量应随 horizon 递增
    width = 0.2 + (24 - days) / 24.0 * 2.0
    p10 = center - width
    p90 = center + width
    cal = ConformalCalibrator().fit(y, p10, p90, days)
    assert cal.lo[-1] > cal.lo[0]


def test_apply_rejects_horizon_out_of_range():
    y, p10, p90, days = _toy()
    cal = ConformalCalibrator().fit(y, p10, p90, days)
    try:
        cal.apply(np.array([1.0]), np.array([2.0]), np.array([99]))
    except ValueError as exc:
        assert "horizon" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_roundtrip_dict():
    y, p10, p90, days = _toy()
    cal = ConformalCalibrator().fit(y[:1400], p10[:1400], p90[:1400], days[:1400])
    clone = ConformalCalibrator.from_dict(cal.to_dict())
    a = cal.apply(p10[1400:], p90[1400:], days[1400:])
    b = clone.apply(p10[1400:], p90[1400:], days[1400:])
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_calibrate_output_helper():
    y, p10, p50, p90 = None, None, None, None
    rng = np.random.default_rng(9)
    y = rng.normal(50, 10, 600)
    p50 = y + rng.normal(0, 1, 600)
    p10, p90 = p50 - 5, p50 + 5
    days = np.tile(np.arange(1, 25), 25)
    cal, lo, hi = calibrate_output(y, p10, p50, p90, days)
    assert cal.fitted and lo.shape == p10.shape
