"""预测精度与运行指标。纯函数，stdlib only。

口径约定（比赛报告必须随指标声明）：
- skill = 1 − RMSE/Cap，含夜间口径会显著抬高数值，白天口径单独报告
- 白天口径：实测或预测任一 > 2% Cap 的槽位纳入统计
"""

from __future__ import annotations

from math import sqrt
from statistics import mean


def rmse(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return sqrt(mean([(p - a) ** 2 for p, a in pairs]))


def mae(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    return mean([abs(p - a) for p, a in pairs])


def bias(pairs: list[tuple[float, float]]) -> float:
    """mean(pred − actual)：正=系统性偏高。"""
    if not pairs:
        return 0.0
    return mean([p - a for p, a in pairs])


def forecast_metrics(
    predicted: list[float],
    actual: list[float],
    cap_kw: float,
) -> dict:
    """全天与白天两套口径的精度指标。"""
    pairs_all = list(zip(predicted, actual))
    pairs_day = [
        (p, a) for p, a in pairs_all if p > 0.02 * cap_kw or a > 0.02 * cap_kw
    ]
    result = {
        "rmse": round(rmse(pairs_all), 4),
        "mae": round(mae(pairs_all), 4),
        "bias": round(bias(pairs_all), 4),
        "skill_all_day": round(1 - rmse(pairs_all) / cap_kw, 4) if cap_kw else None,
        "n_points": len(pairs_all),
    }
    if pairs_day:
        result["rmse_daylight"] = round(rmse(pairs_day), 4)
        result["skill_daylight"] = round(1 - rmse(pairs_day) / cap_kw, 4) if cap_kw else None
        result["n_daylight"] = len(pairs_day)
    else:
        result["rmse_daylight"] = None
        result["skill_daylight"] = None
        result["n_daylight"] = 0
    return result
