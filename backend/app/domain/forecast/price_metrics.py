"""电价预测专用精度口径。纯函数，numpy only。

电价和光伏/负荷的精度不能共用一套口径：
- 功率预测的 skill = 1 − RMSE/Cap 有物理上限做分母；电价没有 Cap，必须用
  **相对基准的 skill**（vs 昨日延续 / vs 上周同刻），否则数字不可比。
- 储能在意的不是均值误差，而是**峰谷时段的排序与分位数**：套利收益取决于
  高价时段有没有被识别出来，所以必须报分位数损失与区间覆盖率。
"""

from __future__ import annotations

import numpy as np

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def _clean(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mats = [np.asarray(a, dtype="float64").ravel() for a in arrays]
    mask = np.ones(mats[0].size, dtype=bool)
    for m in mats:
        mask &= np.isfinite(m)
    return tuple(m[mask] for m in mats)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(np.abs(p - a))) if a.size else float("nan")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = _clean(actual, predicted)
    return float(np.sqrt(np.mean((p - a) ** 2))) if a.size else float("nan")


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = _clean(actual, predicted)
    return float(np.mean(p - a)) if a.size else float("nan")


def mape(actual: np.ndarray, predicted: np.ndarray, floor: float = 1.0) -> float:
    """电价可能接近 0，分母加 floor 防止爆炸；仅作辅助指标。"""
    a, p = _clean(actual, predicted)
    if not a.size:
        return float("nan")
    return float(np.mean(np.abs(p - a) / np.maximum(np.abs(a), floor)))


def pinball(actual: np.ndarray, quantile_pred: np.ndarray, quantiles=DEFAULT_QUANTILES) -> float:
    """分位数（pinball）损失：actual 形状 (n,)，quantile_pred 形状 (n, q)。"""
    a = np.asarray(actual, dtype="float64").ravel()
    q = np.asarray(quantile_pred, dtype="float64")
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.shape[0] != a.size or q.shape[1] != len(quantiles):
        raise ValueError("quantile_pred shape must be (n, len(quantiles))")
    mask = np.isfinite(a) & np.all(np.isfinite(q), axis=1)
    if not mask.any():
        return float("nan")
    a, q = a[mask], q[mask]
    levels = np.asarray(quantiles, dtype="float64").reshape(1, -1)
    error = a.reshape(-1, 1) - q
    return float(np.mean(np.maximum(levels * error, (levels - 1.0) * error)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """区间覆盖率 PICP：理想值等于名义区间宽度（如 P10-P90 → 0.8）。"""
    a, lo, hi = _clean(actual, lower, upper)
    return float(np.mean((a >= lo) & (a <= hi))) if a.size else float("nan")


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lo, hi = _clean(lower, upper)
    return float(np.mean(hi - lo)) if lo.size else float("nan")


def winkler_score(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2
) -> float:
    """Winkler/区间得分：同时惩罚过宽与未覆盖，越小越好。alpha=0.2 对应 80% 区间。"""
    a, lo, hi = _clean(actual, lower, upper)
    if not a.size:
        return float("nan")
    width = hi - lo
    penalty = np.where(a < lo, 2.0 * (lo - a) / alpha, 0.0) + np.where(
        a > hi, 2.0 * (a - hi) / alpha, 0.0
    )
    return float(np.mean(width + penalty))


def skill(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> float:
    """相对基准的 MAE 技巧分：1 − MAE_model/MAE_baseline。>0 才算赢过基准。"""
    base = mae(actual, baseline)
    if not np.isfinite(base) or base == 0:
        return float("nan")
    return float(1.0 - mae(actual, predicted) / base)


def peak_valley_hit(actual: np.ndarray, predicted: np.ndarray, top_k: int = 4) -> float:
    """峰值时段命中率：实际最高的 top_k 小时里，预测也排进 top_k 的比例。

    储能套利真正吃的是这个——均价再准，峰谷排错就赚不到钱。
    """
    a, p = _clean(actual, predicted)
    if a.size < top_k:
        return float("nan")
    true_peak = set(np.argsort(-a)[:top_k].tolist())
    pred_peak = set(np.argsort(-p)[:top_k].tolist())
    return float(len(true_peak & pred_peak) / top_k)


def summarize(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray | None = None,
    quantile_pred: np.ndarray | None = None,
    quantiles=DEFAULT_QUANTILES,
) -> dict[str, float]:
    """一次算全一套指标。quantile_pred 为 None 时退化为点预测（p10=p50=p90）。"""
    point = np.asarray(predicted, dtype="float64")
    q = quantile_pred
    if q is None:
        q = np.repeat(point.reshape(-1, 1), len(quantiles), axis=1)
    q = np.asarray(q, dtype="float64")
    median_pos = int(np.argmin(np.abs(np.asarray(quantiles) - 0.5)))
    center = q[:, median_pos]

    out = {
        "mae": mae(actual, center),
        "rmse": rmse(actual, center),
        "bias": bias(actual, center),
        "mape": mape(actual, center),
        "pinball": pinball(actual, q, quantiles),
        "picp80": coverage(actual, q[:, 0], q[:, -1]),
        "width80": interval_width(q[:, 0], q[:, -1]),
        "winkler80": winkler_score(actual, q[:, 0], q[:, -1], alpha=0.2),
        "peak_hit": peak_valley_hit(actual, center),
        "n_points": int(np.isfinite(np.asarray(actual, dtype="float64")).sum()),
    }
    if baseline is not None:
        out["mae_baseline"] = mae(actual, baseline)
        out["skill"] = skill(actual, center, baseline)
    return out
