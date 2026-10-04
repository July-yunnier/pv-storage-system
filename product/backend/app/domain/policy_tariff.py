"""电价政策的结构化候选、严格校验与系统价目转换。

LLM 只负责生成候选结构。本模块不依赖 LLM，并在候选进入调度系统前完成：
- 时间格式、全天覆盖和重叠校验；
- 生效日期与适用月份校验；
- 电能量附加项与容量/需量费用分离；
- 转换为现有 ``TariffCreate`` 模型。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.models import TariffCreate, TierName

SLOT_MINUTES = 15
SLOTS_PER_DAY = 96


class PolicyEvidence(BaseModel):
    page: int | None = Field(default=None, ge=1)
    quote: str = Field(min_length=1)
    field: str = Field(min_length=1)


class PolicyTimeRule(BaseModel):
    start_time: str = Field(description="HH:MM；允许结束时间使用 24:00")
    end_time: str = Field(description="HH:MM；允许结束时间使用 24:00")
    type: TierName
    multiplier: float = Field(ge=0)
    energy_addition: float = Field(
        default=0.0,
        ge=0,
        description="仅限按 kWh 收取的附加电费，不得填写容量费或需量费",
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_tip(cls, value: str) -> str:
        # 政策材料常用 tip 表示尖峰，系统内部统一使用 sharp。
        return "sharp" if value == "tip" else value


class PolicyTariffCandidate(BaseModel):
    policy_name: str = Field(min_length=1, max_length=200)
    policy_document_no: str | None = None
    currency: Literal["CNY"] = "CNY"
    base_price: float | None = Field(default=None, ge=0)
    transmission_and_fund: float = Field(default=0.0, ge=0)
    published_date: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    applicable_months: list[int] = Field(default_factory=lambda: list(range(1, 13)))
    day_types: list[Literal["all", "workday", "weekend", "holiday"]] = Field(
        default_factory=lambda: ["all"]
    )
    voltage_level: str | None = None
    customer_category: str | None = None
    capacity_charge_yuan_per_kva_month: float | None = Field(default=None, ge=0)
    demand_charge_yuan_per_kw_month: float | None = Field(default=None, ge=0)
    rules: list[PolicyTimeRule] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    evidence: list[PolicyEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_metadata(self) -> "PolicyTariffCandidate":
        if len(set(self.applicable_months)) != len(self.applicable_months):
            raise ValueError("applicable_months must not contain duplicates")
        if any(month < 1 or month > 12 for month in self.applicable_months):
            raise ValueError("applicable_months must be between 1 and 12")
        if self.effective_from and self.effective_to:
            if self.effective_to < self.effective_from:
                raise ValueError("effective_to must not be before effective_from")
        return self


class PolicyParseResponse(BaseModel):
    candidate: PolicyTariffCandidate
    validation_issues: list[str]
    requires_human_confirmation: bool = True


class PolicyConfirmRequest(BaseModel):
    candidate: PolicyTariffCandidate
    tariff_id: str | None = None
    target_date: date | None = None
    confirmed_by: str = Field(min_length=1, description="人工复核人或团队成员标识")


def _minute_of_day(value: str, *, allow_24: bool) -> int:
    try:
        hour_text, minute_text = value.split(":")
        hour, minute = int(hour_text), int(minute_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid time: {value!r}, expected HH:MM") from exc
    if hour == 24 and minute == 0 and allow_24:
        return 24 * 60
    if not 0 <= hour <= 23 or minute not in (0, 15, 30, 45):
        raise ValueError(f"invalid 15-minute boundary: {value!r}")
    return hour * 60 + minute


def _rule_contains(rule: PolicyTimeRule, minute: int) -> bool:
    start = _minute_of_day(rule.start_time, allow_24=False)
    end = _minute_of_day(rule.end_time, allow_24=True)
    if end == 24 * 60:
        return start <= minute < end
    if start == end:
        return False
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


def validate_policy_candidate(
    candidate: PolicyTariffCandidate, target_date: date | None = None
) -> list[str]:
    """返回所有阻止候选进入调度系统的问题，不静默填补政策缺口。"""
    issues = list(candidate.issues)
    if candidate.base_price is None:
        issues.append("政策未确认基础电能量价格 base_price")
    if not candidate.rules:
        issues.append("政策未提取到分时规则")
    if sorted(candidate.applicable_months) != list(range(1, 13)):
        issues.append("当前系统价目尚不支持按月份自动切换，不能把季节性候选保存为全年价目")
    if candidate.day_types != ["all"]:
        issues.append("当前系统价目尚不支持工作日/周末/节假日自动切换")

    for index, rule in enumerate(candidate.rules, start=1):
        try:
            start = _minute_of_day(rule.start_time, allow_24=False)
            end = _minute_of_day(rule.end_time, allow_24=True)
            if end != 24 * 60 and start == end:
                issues.append(f"第 {index} 条规则起止时间相同")
        except ValueError as exc:
            issues.append(f"第 {index} 条规则：{exc}")

    if candidate.rules:
        for slot in range(SLOTS_PER_DAY):
            minute = slot * SLOT_MINUTES
            try:
                matches = [rule for rule in candidate.rules if _rule_contains(rule, minute)]
            except ValueError:
                break
            if len(matches) != 1:
                hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
                issues.append(f"{hhmm} 匹配到 {len(matches)} 条规则，应恰好为 1 条")

    if target_date is not None:
        if candidate.effective_from and target_date < candidate.effective_from:
            issues.append(f"目标日期 {target_date} 早于政策生效日 {candidate.effective_from}")
        if candidate.effective_to and target_date > candidate.effective_to:
            issues.append(f"目标日期 {target_date} 晚于政策失效日 {candidate.effective_to}")
        if target_date.month not in candidate.applicable_months:
            issues.append(f"政策不适用于目标月份 {target_date.month}")

    # 保序去重，方便前端逐项展示。
    return list(dict.fromkeys(issues))


def _system_time(value: str) -> str:
    """现有 TouProvider 用 00:00 表示跨午夜规则的日末边界。"""
    return "00:00" if value == "24:00" else value


def _system_ranges(rule: PolicyTimeRule) -> list[str]:
    if rule.start_time == "00:00" and rule.end_time == "24:00":
        # 现有 Provider 禁止 00:00-00:00；用两个连续区间精确表达全天。
        return ["00:00-23:45", "23:45-00:00"]
    return [f"{_system_time(rule.start_time)}-{_system_time(rule.end_time)}"]


def candidate_to_tariff(
    candidate: PolicyTariffCandidate,
    *,
    tariff_id: str | None = None,
    target_date: date | None = None,
) -> TariffCreate:
    issues = validate_policy_candidate(candidate, target_date)
    if issues:
        raise ValueError("政策候选未通过校验：" + "；".join(issues))
    assert candidate.base_price is not None

    grouped: dict[TierName, dict[str, object]] = {}
    for rule in candidate.rules:
        price = round(candidate.base_price * rule.multiplier + rule.energy_addition, 6)
        hour_ranges = _system_ranges(rule)
        if rule.type in grouped:
            previous_price = grouped[rule.type]["price"]
            if previous_price != price:
                raise ValueError(
                    f"同一 {rule.type} 档存在不同电价 {previous_price} 与 {price}，"
                    "当前系统需拆分价目模型后才能保存"
                )
            grouped[rule.type]["hours"].extend(hour_ranges)  # type: ignore[union-attr]
        else:
            grouped[rule.type] = {"price": price, "hours": hour_ranges}

    tiers = [
        {"name": name, "price": data["price"], "hours": data["hours"]}
        for name, data in grouped.items()
    ]
    return TariffCreate(
        tariff_id=tariff_id,
        name=candidate.policy_name,
        mode="tou",
        transmission_and_fund=candidate.transmission_and_fund,
        tiers=tiers,
    )
