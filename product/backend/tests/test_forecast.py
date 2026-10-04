import math
from datetime import date, datetime, time, timedelta

import pytest

from app.domain.forecast.load import predict_load_intraday, predict_load_profile
from app.domain.forecast.load_advanced import OnlineResidualKalman
from app.domain.forecast.price import forecast_price_tou
from app.domain.forecast.pv import predict_pv_diurnal
from app.domain.models import TariffCreate, TariffRead
from tests.conftest import CN_TZ

TARGET = date(2026, 9, 20)
CAP = 1000.0


def make_history(days: int, func, start: date = date(2026, 9, 8)) -> list[tuple[datetime, float]]:
    """生成 days 天的 15 分钟历史，func(day_index, slot) -> value。"""
    history = []
    for day_index in range(days):
        day = start + timedelta(days=day_index)
        base = datetime.combine(day, time.min, tzinfo=CN_TZ)
        for slot in range(96):
            history.append((base + timedelta(minutes=15 * slot), func(day_index, slot)))
    return history


def pv_pattern(day_index: int, slot: int) -> float:
    hour = slot / 4
    if not 6 <= hour <= 18:
        return 0.0
    return CAP * math.sin(math.pi * (hour - 6) / 12) * (0.8 + 0.05 * (day_index % 3))


def test_pv_deterministic() -> None:
    history = make_history(12, pv_pattern)
    first = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    second = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    assert first.model_dump() == second.model_dump()


def test_pv_nighttime_is_zero() -> None:
    history = make_history(12, pv_pattern)
    forecast = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    assert len(forecast.points) == 96
    night = [p for p in forecast.points if p.ts.hour < 6 or p.ts.hour >= 19]
    assert night
    assert all(point.value == 0 for point in night)
    # 正午接近容量量级
    noon = forecast.points[48]
    assert noon.value > 0.7 * CAP


def test_pv_cap_clip() -> None:
    history = make_history(5, lambda d, s: 2 * CAP)
    forecast = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    assert all(point.value <= CAP for point in forecast.points)
    assert all(point.value == CAP for point in forecast.points)


def test_pv_insufficient_history_raises() -> None:
    history = make_history(2, pv_pattern)
    with pytest.raises(ValueError, match="至少 3 天"):
        predict_pv_diurnal(history, TARGET, CAP, CN_TZ)


def test_pv_accuracy_caliber_complete() -> None:
    history = make_history(12, pv_pattern)
    forecast = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    accuracy = forecast.accuracy
    assert accuracy is not None
    assert accuracy.metric == "one_minus_rmse_over_cap"
    assert 0 < accuracy.value <= 1
    assert accuracy.caliber.normalization == "cap"
    assert accuracy.caliber.cap_kw == CAP
    assert accuracy.caliber.includes_nighttime is True
    assert accuracy.caliber.baseline == "persistence"
    assert accuracy.evaluated_on == "backtest:last_12d"
    assert "12 天" in forecast.notes


def test_pv_uses_fewer_days_when_history_shorter() -> None:
    history = make_history(5, pv_pattern)
    forecast = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    assert forecast.accuracy is not None
    assert forecast.accuracy.evaluated_on == "backtest:last_5d"
    assert "5 天" in forecast.notes


def test_pv_uncertainty_bounds() -> None:
    history = make_history(12, pv_pattern)
    forecast = predict_pv_diurnal(history, TARGET, CAP, CN_TZ)
    assert forecast.p10 is not None and forecast.p90 is not None
    assert len(forecast.p10) == len(forecast.p90) == 96
    for point, lo, hi in zip(forecast.points, forecast.p10, forecast.p90):
        assert lo <= point.value <= hi
        assert 0 <= lo and hi <= CAP


def test_load_diurnal_mean_known_series() -> None:
    # 每槽位恒定值 = slot 序号 → 预测应精确等于槽位序号
    history = make_history(5, lambda d, s: float(s))
    forecast = predict_load_profile(history, TARGET, CN_TZ)
    for slot, point in enumerate(forecast.points):
        assert point.value == pytest.approx(slot)
    assert forecast.accuracy is not None
    assert forecast.accuracy.value == pytest.approx(1.0)
    assert forecast.accuracy.caliber.normalization == "mean_power"
    assert forecast.accuracy.caliber.cap_kw == pytest.approx(47.5)
    assert forecast.accuracy.caliber.baseline == "diurnal_persistence"


def test_load_floor_zero() -> None:
    history = make_history(5, lambda d, s: -10.0)
    forecast = predict_load_profile(history, TARGET, CN_TZ)
    assert all(point.value == 0 for point in forecast.points)


def test_load_insufficient_history_raises() -> None:
    with pytest.raises(ValueError, match="至少 3 天"):
        predict_load_profile(make_history(2, lambda d, s: 100.0), TARGET, CN_TZ)


def test_load_intraday_uses_today_residual_and_decays() -> None:
    history = make_history(5, lambda d, s: 100.0, start=date(2026, 9, 15))
    issued = datetime(2026, 9, 20, 2, 30, tzinfo=CN_TZ)
    today_start = datetime.combine(TARGET, time.min, tzinfo=CN_TZ)
    history.extend(
        (today_start + timedelta(minutes=15 * slot), 110.0)
        for slot in range(11)
    )
    result = predict_load_intraday(history, issued, CN_TZ, decay=0.9)
    assert result.model.name == "load_intraday_kalman"
    assert result.points[0].ts.hour == 2 and result.points[0].ts.minute == 45
    assert result.points[0].value > 100.0
    assert result.points[1].value < result.points[0].value


def test_online_kalman_never_creates_negative_load() -> None:
    kf = OnlineResidualKalman(decay=0.9)
    kf.update(-100.0)
    corrected = kf.correct_remaining([10.0, 10.0, 10.0], observed_slot=0)
    assert corrected[0] == 10.0
    assert corrected[1:] == [0.0, 0.0]


def test_price_tou_policy() -> None:
    now = datetime.now(CN_TZ)
    tariff = TariffRead(
        **TariffCreate(
            tariff_id="T-F",
            name="测试电价",
            transmission_and_fund=0.21,
            tiers=[
                {"name": "peak", "price": 1.0, "hours": ["08:00-20:00"]},
                {"name": "valley", "price": 0.3, "hours": ["20:00-08:00"]},
            ],
        ).model_dump(),
        created_at=now,
        updated_at=now,
    )
    forecast = forecast_price_tou(tariff, TARGET, days=2, tz=CN_TZ)
    assert len(forecast.points) == 192
    assert forecast.model.model_dump() == {"name": "tou_policy", "version": "v1"}
    assert forecast.notes == "政策给定，无需预测"
    assert forecast.accuracy is None
    assert forecast.p10 == forecast.p90 == [p.value for p in forecast.points]
    assert forecast.points[0].value == pytest.approx(0.51)
    assert forecast.points[32].value == pytest.approx(1.21)
