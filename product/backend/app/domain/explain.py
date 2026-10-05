"""计划说明生成器：把 96 点调度计划聚合成连续时段段，配人话解释。

纯模板规则（不依赖 LLM）：动作分类 + 档位 + SOC 变化拼句。
后续 LLM 工作包可在此基础上润色，输入输出契约不变。
"""

from __future__ import annotations

from datetime import timedelta
from statistics import mean

from app.domain.models import DispatchPoint, PlanSegment, TierName

TIER_LABEL: dict[TierName, str] = {
    "sharp": "尖峰",
    "peak": "高峰",
    "flat": "平段",
    "valley": "低谷",
}


def _point_action(p: DispatchPoint) -> str | None:
    """动作四分：charge_pv / charge_grid / discharge；其余（含静置）返回 None。"""
    if p.action == "charge":
        # 新记录直接读拆分字段；旧记录（拆分字段为 0）按光伏是否盈余回退推断
        if p.charge_from_pv_kw > 0 or (p.charge_from_grid_kw == 0 and p.pv_power_kw > p.load_power_kw):
            return "charge_pv"
        return "charge_grid"
    if p.action == "discharge":
        return "discharge"
    return None


def _segment_text(
    action: str,
    tier: TierName | None,
    start: str,
    end: str,
    avg_power: float,
    avg_price: float,
    avg_pv: float,
    soc_from: float,
    soc_to: float,
) -> str:
    span = f"{start}–{end}"
    soc = f"SOC {soc_from:.0f}% → {soc_to:.0f}%"
    tier_label = TIER_LABEL.get(tier, "") if tier else ""
    price = f"{tier_label}段（均价 {avg_price:.2f} 元/kWh）" if tier else ""
    if action == "charge_grid":
        return f"{span} {price}：电网充电 {avg_power:.0f} kW，{soc}，低谷囤便宜电"
    if action == "charge_pv":
        return (
            f"{span}：光伏预测 {avg_pv:.0f} kW 超过负荷，余电直充 {avg_power:.0f} kW"
            f"（不花电费），{soc}"
        )
    return f"{span} {price}：放电 {avg_power:.0f} kW 替代电网购电削峰，{soc}"


def explain_schedule(schedule: list[DispatchPoint]) -> list[PlanSegment]:
    """把 96 点计划聚合为分时段说明；仅保留充电/放电段，静置时段省略。"""
    segments: list[tuple[str, TierName | None, list[DispatchPoint]]] = []
    for p in schedule:
        action = _point_action(p)
        if action is None:
            continue
        if segments and segments[-1][0] == action and segments[-1][1] == p.tier:
            segments[-1][2].append(p)
        else:
            segments.append((action, p.tier, [p]))

    result: list[PlanSegment] = []
    for action, tier, points in segments:
        start = points[0].ts.strftime("%H:%M")
        end = (points[-1].ts + timedelta(minutes=15)).strftime("%H:%M")
        avg_power = mean(p.power_kw for p in points)
        avg_price = mean(p.price for p in points)
        avg_pv = mean(p.pv_power_kw for p in points)
        soc_from = points[0].soc_start * 100
        soc_to = points[-1].soc_end * 100
        result.append(
            PlanSegment(
                start=start,
                end=end,
                action=action,  # type: ignore[arg-type]
                tier=tier,
                avg_power_kw=round(avg_power, 1),
                text=_segment_text(
                    action, tier, start, end, avg_power, avg_price, avg_pv, soc_from, soc_to
                ),
            )
        )
    return result
