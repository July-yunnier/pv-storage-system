from datetime import datetime, timedelta, timezone

import pytest

from app.domain.models import TariffCreate, TariffRead
from app.domain.tariff import TouProvider

TZ = timezone(timedelta(hours=8))


def make_tariff(tiers: list[dict]) -> TariffRead:
    now = datetime.now(timezone.utc)
    return TariffRead(
        **TariffCreate(
            tariff_id="T-TOU",
            name="峰谷测试电价",
            transmission_and_fund=0.21,
            tiers=tiers,
        ).model_dump(),
        created_at=now,
        updated_at=now,
    )


@pytest.fixture()
def jiangsu_like() -> TouProvider:
    return TouProvider(
        make_tariff(
            [
                {"name": "sharp", "price": 1.25, "hours": ["17:00-19:00"]},
                {"name": "peak", "price": 0.95, "hours": ["13:00-17:00", "19:00-20:00"]},
                {"name": "flat", "price": 0.65, "hours": ["07:00-13:00", "20:00-23:00"]},
                {"name": "valley", "price": 0.35, "hours": ["23:00-07:00"]},
            ]
        )
    )


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 20, hour, minute, tzinfo=TZ)


def test_boundary_moments_hit_exactly_one_tier(jiangsu_like: TouProvider) -> None:
    assert jiangsu_like.price_at(at(7, 0)).tier == "flat"
    assert jiangsu_like.price_at(at(6, 45)).tier == "valley"
    assert jiangsu_like.price_at(at(13, 0)).tier == "peak"
    assert jiangsu_like.price_at(at(17, 0)).tier == "sharp"
    assert jiangsu_like.price_at(at(19, 0)).tier == "peak"
    assert jiangsu_like.price_at(at(20, 0)).tier == "flat"
    assert jiangsu_like.price_at(at(23, 0)).tier == "valley"
    assert jiangsu_like.price_at(at(22, 45)).tier == "flat"


def test_cross_midnight_range_matches_late_night_and_early_morning(
    jiangsu_like: TouProvider,
) -> None:
    assert jiangsu_like.price_at(at(23, 30)).tier == "valley"
    assert jiangsu_like.price_at(at(0, 0)).tier == "valley"
    assert jiangsu_like.price_at(at(2, 15)).tier == "valley"
    assert jiangsu_like.price_at(at(6, 59)).tier == "valley"
    assert jiangsu_like.price_at(at(2, 0)).price == pytest.approx(0.35 + 0.21)


def test_gap_in_tiers_raises() -> None:
    provider = TouProvider(
        make_tariff([{"name": "peak", "price": 1.0, "hours": ["08:00-20:00"]}])
    )
    with pytest.raises(ValueError, match="exactly one tier"):
        provider.price_at(at(3, 0))
    # 覆盖时段内仍然正常
    assert provider.price_at(at(12, 0)).tier == "peak"


def test_non_tou_mode_rejected() -> None:
    now = datetime.now(timezone.utc)
    tariff = TariffRead(
        **TariffCreate(
            tariff_id="T-SPOT",
            name="现货电价",
            mode="spot",
            tiers=[{"name": "flat", "price": 0.5, "hours": ["00:00-23:45"]}],
        ).model_dump(),
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(ValueError, match="mode"):
        TouProvider(tariff)
