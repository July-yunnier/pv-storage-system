from datetime import date, datetime

import pytest

from app.domain.policy_tariff import (
    PolicyTariffCandidate,
    candidate_to_tariff,
    validate_policy_candidate,
)
from app.domain.tariff import TouProvider
from tests.conftest import CN_TZ


def valid_candidate() -> PolicyTariffCandidate:
    return PolicyTariffCandidate(
        policy_name="测试政策",
        base_price=0.48,
        transmission_and_fund=0.21,
        effective_from=date(2026, 10, 1),
        rules=[
            {"start_time": "00:00", "end_time": "08:00", "type": "valley", "multiplier": 0.5},
            {"start_time": "08:00", "end_time": "11:00", "type": "peak", "multiplier": 1.7},
            {"start_time": "11:00", "end_time": "12:00", "type": "tip", "multiplier": 2.2},
            {"start_time": "12:00", "end_time": "17:00", "type": "flat", "multiplier": 1.0},
            {"start_time": "17:00", "end_time": "22:00", "type": "peak", "multiplier": 1.7},
            {"start_time": "22:00", "end_time": "24:00", "type": "valley", "multiplier": 0.5},
        ],
    )


def test_candidate_validates_and_tip_normalizes_to_sharp() -> None:
    candidate = valid_candidate()
    assert validate_policy_candidate(candidate, date(2026, 10, 4)) == []
    assert candidate.rules[2].type == "sharp"

    tariff = candidate_to_tariff(candidate, tariff_id="T-POLICY")
    provider = TouProvider(
        tariff.model_copy(
            update={
                "tariff_id": "T-POLICY",
                "created_at": datetime.now(CN_TZ),
                "updated_at": datetime.now(CN_TZ),
            }
        )
    )
    provider.validate_day()


def test_candidate_rejects_gap_and_overlap() -> None:
    candidate = valid_candidate()
    candidate.rules[1].start_time = "08:15"
    issues = validate_policy_candidate(candidate)
    assert any("08:00 匹配到 0 条" in issue for issue in issues)

    candidate = valid_candidate()
    candidate.rules[1].start_time = "07:45"
    issues = validate_policy_candidate(candidate)
    assert any("07:45 匹配到 2 条" in issue for issue in issues)


def test_candidate_rejects_missing_price_and_wrong_effective_date() -> None:
    candidate = valid_candidate().model_copy(update={"base_price": None})
    issues = validate_policy_candidate(candidate, date(2026, 9, 30))
    assert any("base_price" in issue for issue in issues)
    assert any("早于政策生效日" in issue for issue in issues)
    with pytest.raises(ValueError, match="未通过校验"):
        candidate_to_tariff(candidate)


def test_seasonal_policy_not_silently_saved_as_annual_tariff() -> None:
    candidate = valid_candidate().model_copy(update={"applicable_months": [7, 8, 9]})
    issues = validate_policy_candidate(candidate, date(2026, 8, 1))
    assert any("按月份自动切换" in issue for issue in issues)
