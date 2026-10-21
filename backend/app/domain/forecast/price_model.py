"""电价预测模型层：分位数 LightGBM（生产）+ 同刻分位数画像（始终可用的兜底）。

为什么是分位数而不是点预测：储能套利是**带风险的决策**——同一个 p50 曲线，
若 p10/p90 张开得厉害，最优充放电深度就该收缩。只给一条确定性曲线，调度器
无法区分"确定的价差"和"赌出来的价差"。

两条路径的分工与现有光伏/负荷模块一致：
- `lgbm_quantile`：需要 `requirements-forecast.txt` 与已训练工件，精度优先；
- `price_diurnal_quantile`：纯 numpy，冷启动/无依赖时可用，语义可解释。

模型一律工作在**小时粒度**：现货与分时电价在小时内都是常数，15 分钟只是
把同一小时的值重复 4 次，插值反而会造出不存在的价格。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain.forecast.price_calibration import ConformalCalibrator
from app.domain.forecast.price_features import (
    FEATURE_COLUMNS,
    PriceSeries,
    build_samples,
    column_medians,
    select_active_columns,
    to_matrix,
)

QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
MIN_LGBM_TRAIN_ROWS = 500


@dataclass
class PriceForecastOutput:
    """小时粒度的分位数预测结果。"""

    ts: pd.DatetimeIndex
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    method: str
    version: str
    notes: str = ""
    components: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.ts)
        for name in ("p10", "p50", "p90"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length must match ts")
        if np.any(self.p10 > self.p50 + 1e-9) or np.any(self.p50 > self.p90 + 1e-9):
            raise ValueError("quantiles must be ordered p10 <= p50 <= p90")


def _order(median: np.ndarray, spread_lo: np.ndarray, spread_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把三个独立输出压成有序分位数：中位数不动，两侧 spread 取绝对值。"""
    lo = median - np.abs(spread_lo)
    hi = median + np.abs(spread_hi)
    return lo, median, hi


class LgbmQuantileForecaster:
    """LightGBM 分位数回归（三模型，逐分位独立训练）。"""

    name = "price_lgbm_quantile"
    version = "v1"

    def __init__(self) -> None:
        self.models: dict[float, object] = {}
        self.columns: list[str] = []
        self.medians: dict[str, float] = {}
        self.horizon: int = 24
        self.fitted = False
        self.calibrator: ConformalCalibrator | None = None

    # ---- 训练 ----
    def fit(
        self,
        series: PriceSeries,
        horizon: int = 24,
        origins: np.ndarray | None = None,
        min_origin: int = 167,
        num_boost_round: int = 400,
        learning_rate: float = 0.04,
        num_leaves: int = 63,
        min_data_in_leaf: int = 40,
        feature_fraction: float = 0.85,
        bagging_fraction: float = 0.85,
        bagging_freq: int = 1,
        seed: int = 42,
        cal_days: int = 0,
        verbose: bool = False,
    ) -> "LgbmQuantileForecaster":
        """训练三分位模型。

        `cal_days > 0` 时，把训练窗口尾部若干天留作**校准集**（不参与拟合），
        用于保形校准区间。留出会损失一点训练样本，换来覆盖率有保证的区间——
        下游调度要靠这个区间做决策，宁可牺牲一点拟合。
        """
        from lightgbm import LGBMRegressor

        origins = np.arange(len(series.ts)) if origins is None else np.asarray(origins, dtype=int)
        origins = origins[(origins >= min_origin) & (origins + horizon < len(series.ts))]

        cal_origins = np.asarray([], dtype=int)
        train_origins = origins
        if cal_days > 0 and len(origins) > 0:
            cut = origins[-1] - cal_days * 24
            train_origins = origins[origins <= cut]
            cal_origins = origins[origins > cut]
            if len(train_origins) < 200 or len(cal_origins) < 24:
                raise ValueError(
                    f"calibration split too small: train={len(train_origins)} cal={len(cal_origins)}"
                )

        x, y, _, _ = build_samples(
            series, origins=train_origins, horizon=horizon, min_origin=min_origin, require_target=True
        )
        if len(x) < MIN_LGBM_TRAIN_ROWS:
            raise ValueError(
                f"insufficient training rows for price model: {len(x)} < {MIN_LGBM_TRAIN_ROWS}"
            )
        self.columns = select_active_columns(x)
        self.medians = column_medians(x, self.columns)
        matrix = to_matrix(x, self.columns)
        self.horizon = horizon
        self.models = {}
        for level in QUANTILES:
            model = LGBMRegressor(
                objective="quantile",
                alpha=level,
                n_estimators=num_boost_round,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_data_in_leaf,
                colsample_bytree=feature_fraction,
                subsample=bagging_fraction,
                subsample_freq=bagging_freq,
                random_state=seed,
                n_jobs=1,
                verbose=-1 if not verbose else 0,
            )
            model.fit(matrix, y)
            self.models[level] = model
        self.fitted = True

        if cal_days > 0 and len(cal_origins):
            self.calibrator = self._fit_calibrator(series, cal_origins, horizon)
        return self

    def _fit_calibrator(
        self, series: PriceSeries, cal_origins: np.ndarray, horizon: int
    ) -> ConformalCalibrator:
        """在校准 origin 上取未校准的 P10/P90 与真实价格，学习外扩量。"""
        x, y, _, _ = build_samples(
            series, origins=cal_origins, horizon=horizon, min_origin=0, require_target=True
        )
        if len(x) < 24:
            raise ValueError("calibration samples insufficient")
        matrix = to_matrix(x, self.columns, self.medians)
        raw = {level: self.models[level].predict(matrix) for level in QUANTILES}
        p10, _, p90 = _order(raw[0.5], raw[0.5] - raw[0.1], raw[0.9] - raw[0.5])
        return ConformalCalibrator().fit(y, p10, p90, x["horizon"].to_numpy(dtype=int))

    # ---- 推理 ----
    def predict(
        self,
        series: PriceSeries,
        issue_idx: int,
        horizon: int | None = None,
        apply_calibration: bool = True,
    ) -> PriceForecastOutput:
        """`apply_calibration=False` 用于消融：同一模型输出未校准区间做对照。"""
        if not self.fitted:
            raise RuntimeError("price model is not fitted")
        horizon = horizon or self.horizon
        x, _, target_ts, _ = build_samples(
            series,
            origins=np.asarray([issue_idx]),
            horizon=horizon,
            min_origin=0,
            require_target=False,
        )
        if len(x) == 0:
            raise ValueError("no forecast rows: issue index too close to series end")
        matrix = to_matrix(x, self.columns, self.medians)
        raw = {level: self.models[level].predict(matrix) for level in QUANTILES}
        p10, p50, p90 = _order(raw[0.5], raw[0.5] - raw[0.1], raw[0.9] - raw[0.5])
        notes = "LightGBM 分位数回归，direct multi-horizon（特征锚定发报时刻，不用递归）"
        if apply_calibration and self.calibrator is not None:
            steps = x["horizon"].to_numpy(dtype=int)
            p10, p90 = self.calibrator.apply(p10, p90, steps)
            notes += (
                f"；区间经保形校准（α={self.calibrator.alpha:.2f}，"
                f"校准集 {self.calibrator.n_cal} 点，"
                f"校准前覆盖 {self.calibrator.coverage_before:.3f}）"
            )
        return PriceForecastOutput(
            ts=target_ts,
            p10=p10,
            p50=p50,
            p90=p90,
            method=self.name,
            version=self.version,
            notes=notes,
        )

    # ---- 工件 ----
    def save(self, directory: str | Path) -> Path:
        """落盘工件。

        注意：LightGBM 的 `save_model` 走 C++ 的窄字符 fopen，在 Windows 上遇到
        中文路径会报 "not available for writes"。因此统一用 `model_to_string()`
        交给 Python 写文件，读取侧对称地用 `Booster(model_str=...)`。
        """
        if not self.fitted:
            raise RuntimeError("cannot save an unfitted price model")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for level, model in self.models.items():
            text = model.booster_.model_to_string()
            (target / f"q{int(level * 100):02d}.txt").write_text(text, encoding="utf-8")
        manifest = {
            "name": self.name,
            "version": self.version,
            "horizon": self.horizon,
            "quantiles": list(QUANTILES),
            "columns": self.columns,
            "medians": self.medians,
            "feature_columns": list(FEATURE_COLUMNS),
            "calibration": self.calibrator.to_dict() if self.calibrator is not None else None,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return target

    @classmethod
    def load(cls, directory: str | Path) -> "LgbmQuantileForecaster":
        from lightgbm import Booster

        source = Path(directory)
        manifest_path = source / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"price model manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = cls()
        model.columns = list(manifest["columns"])
        model.medians = {k: float(v) for k, v in manifest["medians"].items()}
        model.horizon = int(manifest["horizon"])
        model.version = str(manifest.get("version", "v1"))
        model.models = {
            float(level): Booster(
                model_str=(source / f"q{int(float(level) * 100):02d}.txt").read_text(encoding="utf-8")
            )
            for level in manifest["quantiles"]
        }
        payload = manifest.get("calibration")
        model.calibrator = ConformalCalibrator.from_dict(payload) if payload else None
        model.fitted = True
        return model


# --------------------------------------------------------------------------------------
# 兜底：同刻分位数画像（纯 numpy，无外部依赖）
# --------------------------------------------------------------------------------------


def _day_type(ts: pd.Timestamp) -> str:
    from app.domain.forecast import price_calendar as cal

    day = ts.date()
    if cal.is_holiday(day):
        return "holiday"
    return "workday" if cal.is_workday(day) else "weekend"


def diurnal_quantile_forecast(
    series: PriceSeries,
    issue_idx: int,
    horizon: int = 24,
    min_days: int = 3,
    window_days: int = 14,
) -> PriceForecastOutput:
    """按 (日类型, 小时) 的历史经验分位数出曲线。

    有目录电价时预测的是**价差**（市场价 − 目录价），再叠加到目标小时的目录价上。
    这样做的好处：政策调价、季节尖峰、节假日深谷会被目录价自动承接，模型只需
    学"市场在政策底价之上加了多少"，样本量要求从"至少一年"降到"至少两周"。

    **只用最近 `window_days` 天**：电价水平不平稳（燃料、来水、机组结构都会整体
    抬升或压低价格带），用半年样本的经验分位数会在 regime shift 后系统性偏低。
    这一点在 120 天演示序列上非常明显（8—9 月 0.3 元、12 月 0.9 元）。

    历史不足以覆盖节假日的，退化为全体同期样本；再不足则整段用中位数。
    这是"没有模型也能出可解释曲线"的最低保障，不是精度方案。
    """
    window_hours = max(window_days, min_days) * 24
    start = max(0, issue_idx + 1 - window_hours)
    known = series.price[start : issue_idx + 1]
    known_ts = series.ts[start : issue_idx + 1]
    target_ts = series.ts[issue_idx + 1 : issue_idx + 1 + horizon]
    if len(target_ts) == 0:
        raise ValueError("issue index too close to series end")

    valid = np.isfinite(known)
    if valid.sum() < 24:
        raise ValueError("not enough history for diurnal price profile")

    hours = np.asarray([t.hour for t in known_ts])[valid]
    days = np.asarray([_day_type(t) for t in known_ts])[valid]
    values = known[valid]

    tou_known = None
    if series.tou_price is not None:
        candidate = series.tou_price[start : issue_idx + 1][valid]
        if np.isfinite(candidate).mean() > 0.9:
            tou_known = candidate
    if tou_known is not None:
        values = values - tou_known  # 学价差

    p10 = np.full(len(target_ts), np.nan)
    p50 = np.full(len(target_ts), np.nan)
    p90 = np.full(len(target_ts), np.nan)
    fallback_level = float(np.median(values))

    for pos, ts in enumerate(target_ts):
        dtype = _day_type(ts)
        mask = (hours == ts.hour) & (days == dtype)
        if mask.sum() < min_days:
            mask = hours == ts.hour
        if mask.sum() < min_days:
            mask = np.ones_like(values, dtype=bool)
        sample = values[mask]
        if sample.size < 3:
            sample = values
        p10[pos], p50[pos], p90[pos] = np.percentile(sample, [10, 50, 90])

    if not np.isfinite(p50).all():
        p10 = np.nan_to_num(p10, nan=fallback_level)
        p50 = np.nan_to_num(p50, nan=fallback_level)
        p90 = np.nan_to_num(p90, nan=fallback_level)

    components: dict[str, np.ndarray] = {}
    notes = "同刻分位数画像（日类型 × 小时的经验分位数）"
    if tou_known is not None:
        tou_target = series.tou_price[issue_idx + 1 : issue_idx + 1 + horizon]
        tou_target = np.where(np.isfinite(tou_target), tou_target, np.nanmedian(tou_target))
        p10, p50, p90 = p10 + tou_target, p50 + tou_target, p90 + tou_target
        components["tou_price"] = tou_target
        components["market_spread_p50"] = p50 - tou_target
        notes = "目录电价 + 历史价差经验分位数（政策调价与节假日由目录价承接）"
    return PriceForecastOutput(
        ts=target_ts,
        p10=p10,
        p50=p50,
        p90=p90,
        method="price_diurnal_quantile",
        version="v1",
        notes=notes,
        components=components,
    )
