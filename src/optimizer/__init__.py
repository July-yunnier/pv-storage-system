"""优化内核：日调度（dispatch）与容量规划（sizing）。

两者本质是**同一个优化问题**，只是时间尺度与决策变量不同：

|          | dispatch（运行）      | sizing（规划）        |
|----------|----------------------|----------------------|
| 触发     | 每日滚动             | 一次性               |
| 决策变量 | 每时段充放电功率     | **容量 kWh / 功率 kW** |
| 目标     | 当日电费最小         | 全生命周期 NPV 最大  |
| 额外约束 | SOC 上下限、循环次数 | 投资预算、场地面积    |

因此：

    core.py      目标函数 + 约束构造（两者共用，确保口径一致）
    dispatch.py  容量固定，只决策功率
    sizing.py    决策变量含容量

⚠️ 不要为两者各写一套优化逻辑。
那会导致「规划说三年回本、运行却赚不到」的口径不一致，企业客户一眼就能看出来。

实现顺序建议
------------
1. ✅ **规则式 dispatch**（已完成，见 dispatch.py）：谷时充、峰时放，按 SOC 上下限截断，
   并带**需量感知约束**（充电不得推高并网峰值）+ **削峰优先**（`peak_target_kw`）。
   峰谷时段已知，规则式天然接近最优，几十行就能出可用调度表。
2. ✅ **sizing**（已完成，见 sizing.py）：离散扫描「容量 × 削峰目标」，
   对每个候选真跑调度算收益，按全周期 NPV 选优。零依赖（不需要 LP 求解器）。
3. ⏳ 升级为 **LP dispatch**：目标 = 电费最小化，约束 = SOC / 功率 / 循环次数 / 需量。

⚠️ 规则式已经暴露出两个真实缺陷（值得写进报告）：
   ① 不加需量约束时，谷时满功率充电会把并网峰值推到比不配储能更高
      （示例站点实测 542 → 706 kW），省下的电度电费从基本电费里加倍还回去。
   ② 只按电价时段放电、不为削峰预留电量时，储能在中午净负荷还很低时就放空
      （实测 13:00-16:00 把 SOC 从 95% 放到 17%），等傍晚真正的高峰
      （650~684 kW）来了无电可放 —— 削峰量为 0，需量电费一分钱没省。

   这两个坑 LP 版本能自然规避（把需量写进目标函数），
   规则式则靠 `peak_target_kw` 显式处理：充电不超过目标 + 为削峰预留电量。
"""
from __future__ import annotations

from .dispatch import make_day_arrays, rule_based_dispatch, synthetic_day
from .sizing import (
    DEFAULT_SHAVING_RATIOS,
    CandidateResult,
    DayScenario,
    Economics,
    EconomicsConfig,
    Recommendation,
    SensitivityEntry,
    SizingResult,
    default_candidates,
    plan_capacity,
)

__all__ = [
    # dispatch
    "rule_based_dispatch",
    "synthetic_day",
    "make_day_arrays",
    # sizing
    "plan_capacity",
    "default_candidates",
    "DayScenario",
    "EconomicsConfig",
    "SizingResult",
    "Recommendation",
    "Economics",
    "SensitivityEntry",
    "CandidateResult",
    "DEFAULT_SHAVING_RATIOS",
]
