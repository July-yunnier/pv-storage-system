"""特征工程的防泄漏测试。

电价预测最容易犯的错不是模型不够强，而是特征里混进了发报时刻之后的信息。
这里用"改未来、看特征是否变化"的方式把这条红线钉死。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.domain.forecast.price_features import (
    FEATURE_COLUMNS,
    PriceSeries,
    build_samples,
    select_active_columns,
    to_matrix,
)


def _series(n: int = 400, seed: int = 7) -> PriceSeries:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    price = 50 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 2, n)
    return PriceSeries(
        ts=ts,
        price=price,
        load=1000 + rng.normal(0, 50, n),
        temperature=15 + 8 * np.sin(np.arange(n) * 2 * np.pi / 24),
        tou_price=np.tile([0.3] * 8 + [0.9] * 4 + [0.6] * 12, n // 24 + 1)[:n],
    )


def test_features_do_not_look_past_origin():
    base = _series()
    origins = np.asarray([200])
    x1, y1, ts1, _ = build_samples(base, origins=origins, horizon=24, min_origin=0)

    tampered_price = base.price.copy()
    tampered_price[201:] += 500.0  # 发报时刻之后的真实价格全部被改掉
    tampered = PriceSeries(
        ts=base.ts,
        price=tampered_price,
        load=base.load,
        temperature=base.temperature,
        tou_price=base.tou_price,
    )
    x2, y2, _, _ = build_samples(tampered, origins=origins, horizon=24, min_origin=0)

    # 特征必须逐列完全一致（未来价格不许进入任何滞后/滚动统计）
    pd.testing.assert_frame_equal(x1, x2)
    # 标签当然会变——这说明改的确实是"未来"
    assert not np.allclose(y1, y2)


def test_horizon_row_count_and_alignment():
    base = _series()
    x, y, target_ts, origin_ts = build_samples(base, origins=np.asarray([200]), horizon=24, min_origin=0)
    assert len(x) == 24
    assert len(y) == 24
    assert list(x["horizon"]) == [float(h) for h in range(1, 25)]
    assert set(x.columns) == set(FEATURE_COLUMNS)
    assert target_ts[0] == base.ts[201]
    assert origin_ts[0] == base.ts[200]


def test_active_columns_drop_all_nan():
    base = _series()
    # 去掉目录电价与温度 → 对应列应被判定为"不可用"
    sparse = PriceSeries(ts=base.ts, price=base.price)
    x, _, _, _ = build_samples(sparse, origins=np.arange(200, 260), horizon=6, min_origin=0)
    columns = select_active_columns(x)
    assert "tou_price_target" not in columns
    assert "temp_target" not in columns
    assert "price_lag_w1" in columns
    matrix = to_matrix(x, columns)
    assert matrix.shape == (len(x), len(columns))


def test_week_lag_requires_full_window():
    base = _series()
    # min_origin=167 时，origin=100 必须被剔除，只剩 origin=200 的 2 个步长
    x, _, _, origins = build_samples(base, origins=np.asarray([100, 200]), horizon=2, min_origin=167)
    assert len(x) == 2
    assert set(origins) == {base.ts[200]}
