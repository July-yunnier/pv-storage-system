from __future__ import annotations

import numpy as np

from app.domain.forecast.price_metrics import (
    coverage,
    mae,
    peak_valley_hit,
    pinball,
    skill,
    summarize,
)


def test_pinball_matches_manual_formula():
    actual = np.asarray([10.0, 20.0])
    q = np.asarray([[8.0, 9.0, 11.0], [22.0, 21.0, 19.0]])
    levels = np.asarray([0.1, 0.5, 0.9])
    error = actual.reshape(-1, 1) - q
    expected = np.mean(np.maximum(levels * error, (levels - 1) * error))
    assert np.isclose(pinball(actual, q), expected)
    # 完美预测的 pinball 为 0
    assert pinball(actual, np.repeat(actual.reshape(-1, 1), 3, axis=1)) == 0.0


def test_coverage_and_skill():
    actual = np.asarray([1.0, 2.0, 3.0, 4.0])
    lo = np.asarray([0.0, 1.5, 2.5, 5.0])
    hi = np.asarray([1.5, 2.5, 3.5, 6.0])
    assert coverage(actual, lo, hi) == 0.75
    perfect = actual.copy()
    baseline = np.asarray([2.0, 2.0, 2.0, 2.0])
    assert skill(actual, perfect, baseline) == 1.0
    assert skill(actual, baseline, baseline) == 0.0


def test_peak_hit_and_summary_keys():
    actual = np.asarray([5.0, 9.0, 3.0, 1.0])
    predicted = np.asarray([4.0, 8.0, 2.0, 0.5])
    assert peak_valley_hit(actual, predicted, top_k=1) == 1.0
    stats = summarize(actual, predicted)
    for key in ("mae", "rmse", "pinball", "picp80", "peak_hit"):
        assert key in stats
    assert stats["picp80"] == 0.0  # 点预测退化为零宽区间，几乎不可能命中
    assert mae(actual, predicted) > 0
