"""分位数区间校准：把名义 80% 区间拉回实测 80%。

为什么需要单独一层：
LightGBM 三个分位数是**各自独立训练**的，没有任何机制保证 P10–P90 真的覆盖
80% 的样本。GEFCom2014-P 上实测 PICP80 = 0.686，即区间系统性偏窄——价格分布
右偏、尖峰厚尾，独立分位回归会低估尾部离散度。区间偏窄的后果不是"看起来不准"，
而是**下游调度按错误的置信度做决策**：以为只有 20% 概率突破 P90，实际是 40%。

方法用保形预测（Conformalized Quantile Regression, CQR）：
- 在训练窗口尾部留一段**校准集**（不参与拟合）；
- 计算不符合度 `E = max(p10 − y, y − p90)`，只有落在区间外时才为正；
- 取 E 的 (1−α) 分位数 q，把区间对称外扩 q：`p10' = p10 − q`，`p90' = p90 + q`；
- 只要校准集与测试集可交换，覆盖率就有有限样本保证（≥ 1−α），不依赖分布假设。

按**提前小时**分别校准：24 步的误差结构不同（1 步和 24 步的宽度差一倍以上），
用同一个 q 会让近端过宽、远端过窄。

代价要说清楚：q 通常为正，区间变宽后 pinball 损失会上升、Winkler 得分可能变差。
这不是"变得更准"，而是**把说过的 80% 兑现**。取舍必须摆在汇报里，不能只报 PICP。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ALPHA = 0.2  # 80% 区间


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """有限样本保形分位数：取第 ceil((n+1)(1−α)) 小的不符合度。

    直接用 `np.quantile(scores, 1-alpha)` 会低估——样本量有限时它给出的覆盖
    只是渐近的。标准做法见 Angelopoulos & Bates (2021) 的 split conformal。
    """
    values = np.sort(np.asarray(scores, dtype="float64"))
    n = values.size
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(values[k - 1])


@dataclass
class ConformalCalibrator:
    """按提前小时（horizon）分别学习的保形外扩量。"""

    alpha: float = DEFAULT_ALPHA
    lo: np.ndarray | None = None      # 每个 horizon 的下侧外扩量
    hi: np.ndarray | None = None      # 每个 horizon 的上侧外扩量
    n_cal: int = 0
    coverage_before: float = float("nan")
    coverage_after: float = float("nan")

    @property
    def fitted(self) -> bool:
        return self.lo is not None and self.hi is not None

    def fit(self, y: np.ndarray, p10: np.ndarray, p90: np.ndarray, horizons: np.ndarray) -> "ConformalCalibrator":
        """y / p10 / p90 / horizons 等长。horizons 为 1..H 的整数。"""
        y = np.asarray(y, dtype="float64")
        p10 = np.asarray(p10, dtype="float64")
        p90 = np.asarray(p90, dtype="float64")
        h = np.asarray(horizons, dtype=int)
        if not (len(y) == len(p10) == len(p90) == len(h)):
            raise ValueError("y / p10 / p90 / horizons must have equal length")
        mask = np.isfinite(y) & np.isfinite(p10) & np.isfinite(p90)
        if mask.sum() < 24:
            raise ValueError(f"calibration set too small: {int(mask.sum())}")
        y, p10, p90, h = y[mask], p10[mask], p90[mask], h[mask]

        self.coverage_before = float(np.mean((y >= p10) & (y <= p90)))
        steps = int(h.max())
        lo = np.zeros(steps)
        hi = np.zeros(steps)
        # 不符合度：落在区间外才为正；区间越窄、越容易为正
        score = np.maximum(p10 - y, y - p90)
        # 全样本分位数：每步只有几十个点时，单步估计方差很大，取二者较大值兜底
        q_global = _conformal_quantile(score, self.alpha)
        for step in range(1, steps + 1):
            sample = score[h == step]
            q_step = _conformal_quantile(sample, self.alpha) if sample.size >= 12 else q_global
            q = max(q_step, q_global)
            lo[step - 1] = max(q, 0.0)   # 只外扩不收缩：收缩会把已有覆盖率打回去
            hi[step - 1] = max(q, 0.0)

        self.lo, self.hi = lo, hi
        self.n_cal = int(mask.sum())
        widened_lo = p10 - lo[h - 1]
        widened_hi = p90 + hi[h - 1]
        self.coverage_after = float(np.mean((y >= widened_lo) & (y <= widened_hi)))
        return self

    def apply(self, p10: np.ndarray, p90: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        h = np.asarray(horizons, dtype=int)
        if h.min() < 1 or h.max() > len(self.lo):
            raise ValueError(f"horizon out of calibrated range: [{h.min()}, {h.max()}]")
        return (
            np.asarray(p10, dtype="float64") - self.lo[h - 1],
            np.asarray(p90, dtype="float64") + self.hi[h - 1],
        )

    # ---- 工件 ----
    def to_dict(self) -> dict:
        if not self.fitted:
            raise RuntimeError("calibrator is not fitted")
        return {
            "alpha": self.alpha,
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "n_cal": self.n_cal,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ConformalCalibrator":
        return cls(
            alpha=float(payload["alpha"]),
            lo=np.asarray(payload["lo"], dtype="float64"),
            hi=np.asarray(payload["hi"], dtype="float64"),
            n_cal=int(payload.get("n_cal", 0)),
            coverage_before=float(payload.get("coverage_before", float("nan"))),
            coverage_after=float(payload.get("coverage_after", float("nan"))),
        )


def calibrate_output(
    y: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    horizons: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[ConformalCalibrator, np.ndarray, np.ndarray]:
    """拟合并应用校准，返回 (校准器, 新 p10, 新 p90)。"""
    cal = ConformalCalibrator(alpha=alpha).fit(y, p10, p90, horizons)
    new_lo, new_hi = cal.apply(p10, p90, horizons)
    return cal, new_lo, new_hi
