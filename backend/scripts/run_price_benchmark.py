"""GEFCom2014-P 电价预测基准实验：滚动 origin、15 个任务日、统一口径。

协议（严格，不许放松）：
- 对第 k 个任务日 D，**只允许使用 D 之前的数据**训练与取特征；
- 预测 D 当日 00:00–23:00 共 24 个小时价格，一次前向输出，不递归；
- 模型均在每个任务日重新拟合（rolling origin），不是"训一次跑 15 天"；
- 评价集合 = 15 天 × 24 小时 = 360 个点，逐点对齐。

对比对象：
1. `naive_daily`   昨日同时刻（最常用的行业基准）
2. `naive_weekly`  上周同时刻
3. `lear`          Lasso 自回归（Uniejewski/Weron 系，电价预测经典强基准）
4. `lgbm_quantile` 本文分位数梯度提升
5. `mqcnn`         MQ-CNN（Wen et al., NIPS 2017）
6. `mqrnn`         MQ-RNN（同论文 RNN 变体，July 复现移植，forking-sequences）
7. `gefcom_benchmark` 赛事官方基准给出的 99 分位数预测

2026-10-22 修正：MQ-CNN 此前误用全序列训练（泄漏任务日之后的数据），
已改为与其余模型一致的 origin 截断；docs/15 的 MQ-CNN 旧数字随之作废，
以本次修正后的重跑为准（见 docs/18 §5）。

用法::

    python scripts/run_price_benchmark.py --data ../../data/power/price/Price \
        --output data/forecast_experiments/price_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain.forecast.gefcom_price import (  # noqa: E402
    TASK_DAYS,
    load_gefcom_price,
    load_task_benchmark,
)
from app.domain.forecast.price_features import PriceSeries  # noqa: E402
from app.domain.forecast.price_metrics import summarize  # noqa: E402
from app.domain.forecast.price_model import QUANTILES  # noqa: E402

HORIZON = 24


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows 不提供 resource；不阻止离线实验。
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)


def _rolling_days(series: PriceSeries, start: str, count: int) -> list[pd.Timestamp]:
    """连续自然日；不补测试真值，不静默跳过缺失日或 DST 异常日。"""
    if count < 1:
        raise ValueError("rolling-days must be positive")
    first = pd.Timestamp(start, tz=series.ts.tz)
    if first != first.normalize():
        raise ValueError("rolling-start must be a date at midnight")
    days = list(pd.date_range(first, periods=count, freq="D"))
    for day in days:
        local_day = day.tz_localize("America/New_York") if day.tzinfo is None else day
        local_grid = pd.date_range(local_day, periods=HORIZON, freq="h")
        if (len({stamp.date() for stamp in local_grid}) != 1
                or len({stamp.hour for stamp in local_grid}) != 24):
            raise ValueError(f"evaluation day must have 24 distinct hours: {day.date()}")
        grid = pd.date_range(day, periods=HORIZON, freq="h")
        pos = series.ts.get_indexer(grid)
        if (pos < 0).any() or not np.isfinite(series.price[pos]).all():
            raise ValueError(f"missing evaluation actuals for {day.date()}")
    return days


def _forecast_window(series: PriceSeries, origin_idx: int, train_days: int) -> tuple[PriceSeries, int]:
    """物理截断到目标日末尾，并屏蔽发报后的价格；未来负荷是日前已知量。"""
    start = max(0, origin_idx + 1 - train_days * 24) if train_days else 0
    end = origin_idx + HORIZON + 1
    price = series.price[start:end].copy()
    issue = origin_idx - start
    price[issue + 1 :] = np.nan
    values = {}
    for name in ("load", "load_total", "temperature", "ghi", "tou_price"):
        value = getattr(series, name)
        values[name] = None if value is None else value[start:end].copy()
    event = None if series.event is None else {k: v[start:end].copy() for k, v in series.event.items()}
    return PriceSeries(ts=series.ts[start:end], price=price, event=event,
                       event_calendar=series.event_calendar, event_timezone=series.event_timezone,
                       event_available_at=series.event_available_at, **values), issue


def _day_matrix(series: PriceSeries) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """(天数, 24) 的价格/分区负荷/系统负荷矩阵，按自然日切。"""
    frame = pd.DataFrame(
        {
            "ts": series.ts,
            "price": series.price,
            "load": series.load,
            "load_total": series.load_total,
        }
    )
    frame["day"] = frame["ts"].dt.normalize()
    frame["hour"] = frame["ts"].dt.hour
    pivots = {}
    for col in ("price", "load", "load_total"):
        pivots[col] = frame.pivot_table(index="day", columns="hour", values=col, aggfunc="first")
    days = pivots["price"].index
    out = []
    for col in ("price", "load", "load_total"):
        block = pivots[col].reindex(columns=range(24)).to_numpy(dtype="float64")
        out.append(block)
    return days, out[0], out[1], out[2]


def _lear_forecast(
    days: pd.DatetimeIndex,
    price: np.ndarray,
    load: np.ndarray,
    target_day: pd.Timestamp,
    alpha: float = 0.02,
) -> np.ndarray:
    """LEAR：对每个小时单独建 Lasso 自回归，特征为前 7 天同刻与邻刻价格 + 当日负荷。"""
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    target_pos = int(np.searchsorted(days.values, np.datetime64(target_day)))
    if target_pos >= len(days) or days[target_pos] != target_day:
        raise ValueError(f"target day {target_day.date()} not in day index")

    usable = []
    for d in range(7, target_pos):
        if np.isfinite(price[d]).all() and np.isfinite(price[d - 7 : d]).all():
            usable.append(d)
    if len(usable) < 60:
        raise ValueError("not enough complete days for LEAR")

    prediction = np.full(24, np.nan)
    for hour in range(24):
        features, targets = [], []
        for d in usable:
            row = []
            for lag in range(1, 8):
                row.append(price[d - lag, hour])
            for lag in range(1, 8):
                row.append(price[d - lag, (hour - 1) % 24])
                row.append(price[d - lag, (hour + 1) % 24])
            row.append(load[d, hour] if np.isfinite(load[d, hour]) else 0.0)
            features.append(row)
            targets.append(price[d, hour])
        x = np.asarray(features, dtype="float64")
        y = np.asarray(targets, dtype="float64")
        model = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000))
        model.fit(x, y)
        row = []
        for lag in range(1, 8):
            row.append(price[target_pos - lag, hour])
        for lag in range(1, 8):
            row.append(price[target_pos - lag, (hour - 1) % 24])
            row.append(price[target_pos - lag, (hour + 1) % 24])
        row.append(load[target_pos, hour] if np.isfinite(load[target_pos, hour]) else 0.0)
        prediction[hour] = float(model.predict(np.asarray([row]))[0])
    return prediction


def _record(rows: list[dict], task: int, day: pd.Timestamp, model: str, actual, p10, p50, p90) -> None:
    day = day.tz_localize("America/New_York") if day.tzinfo is None else day
    for hour in range(HORIZON):
        rows.append(
            {
                "task": task,
                "day": day.date().isoformat(),
                "ts": (day + pd.Timedelta(hours=hour)).isoformat(),
                "model": model,
                "actual": float(actual[hour]),
                "p10": float(p10[hour]),
                "p50": float(p50[hour]),
                "p90": float(p90[hour]),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="GEFCom2014 Price/ 目录")
    parser.add_argument("--output", default="data/forecast_experiments/price_benchmark")
    parser.add_argument(
        "--models",
        default="naive_daily,naive_weekly,lear,lgbm_quantile,mqcnn,mqrnn,gefcom_benchmark",
    )
    parser.add_argument("--origin-stride", type=int, default=3, help="LGBM 训练 origin 抽样步长")
    parser.add_argument("--mqcnn-stride", type=int, default=6, help="MQ-CNN 训练 origin 抽样步长")
    parser.add_argument("--mqcnn-epochs", type=int, default=25)
    parser.add_argument("--mqrnn-stride", type=int, default=1, help="MQ-RNN 训练块抽样步长")
    parser.add_argument("--mqrnn-epochs", type=int, default=20)
    parser.add_argument("--lgbm-rounds", type=int, default=300)
    parser.add_argument("--cal-days", type=int, default=45, help="尾部留作保形校准集的天数，0 表示不校准")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="", help="只跑指定任务，如 1,15（调试用）")
    parser.add_argument("--rolling-start", default="", help="连续评估起日 YYYY-MM-DD")
    parser.add_argument("--rolling-days", type=int, default=0, help="连续评估天数，不能与 --tasks 混用")
    parser.add_argument("--train-days", type=int, default=0, help="训练历史天数（含校准），0 为扩展窗口")
    parser.add_argument("--price-scale", type=float, default=1.0,
                        help="源价格换算系数；T3 用 0.008 换算到元/kWh（情景汇率 8 元/美元）")
    parser.add_argument("--append", action="store_true", help="保留已有 predictions.csv 的其他模型")
    args = parser.parse_args()

    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    if bool(args.rolling_start) != bool(args.rolling_days) or args.rolling_days < 0:
        parser.error("--rolling-start and positive --rolling-days must be provided together")
    if args.rolling_start and (args.tasks or selected - {"naive_daily", "lgbm_quantile", "lgbm_calibrated"}):
        parser.error("rolling mode supports naive_daily/lgbm_quantile/lgbm_calibrated and no --tasks")
    if (args.train_days < 0 or args.cal_days < 0 or args.origin_stride < 1
            or args.lgbm_rounds < 1 or not np.isfinite(args.price_scale) or args.price_scale <= 0):
        parser.error("invalid training window, calibration days, stride, rounds or price scale")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    gefcom = load_gefcom_price(args.data)
    series = gefcom.series
    # 先检查原始真值，再前向填补历史中的短缺口，避免把缺失测试标签编成实测。
    rolling_days = _rolling_days(series, args.rolling_start, args.rolling_days) if args.rolling_start else None
    series.price = series.price * args.price_scale
    # 美东夏令时缺口（2013-03-10 02:00 不存在）：单点前向填充，只影响 3 月训练窗口，
    # 15 个任务日（6-7 月、12 月）的真实值不受影响
    for attr in ("price", "load", "load_total"):
        values = getattr(series, attr)
        if not np.isfinite(values).all():
            filled = pd.Series(values).ffill(limit=2).to_numpy(dtype="float64")
            n_filled = int((~np.isfinite(values)).sum() - (~np.isfinite(filled)).sum())
            setattr(series, attr, filled)
            print(f"filled {n_filled} DST-gap point(s) in {attr}", flush=True)
    # 保留官方任务编号：--tasks 只是筛选，不能重排，否则基准文件会错位
    task_items = list(enumerate(gefcom.task_days, start=1))
    if rolling_days is not None:
        task_items = list(enumerate(rolling_days, start=1))
    if args.tasks:
        keep = {int(t) for t in args.tasks.split(",")}
        task_items = [(i, d) for i, d in task_items if i in keep]

    days, price_matrix, load_matrix, _ = _day_matrix(series)
    rows: list[dict] = []
    log: list[dict] = []

    for task_index, day in task_items:
        day = pd.Timestamp(day)
        actual = np.asarray(
            series.price[
                series.ts.get_indexer(pd.date_range(day, periods=HORIZON, freq="h"))
            ],
            dtype="float64",
        )
        origin_idx = int(series.ts.get_loc(day) - 1)
        train_origins = np.arange(0, origin_idx - HORIZON + 1, args.origin_stride)
        t0 = time.time()

        # 1) 昨日同时刻
        if "naive_daily" in selected:
            prev = series.price[origin_idx - 23 : origin_idx + 1]
            _record(rows, task_index, day, "naive_daily", actual, prev, prev, prev)

        # 2) 上周同时刻
        if "naive_weekly" in selected:
            prev = series.price[origin_idx - 167 : origin_idx - 143]
            _record(rows, task_index, day, "naive_weekly", actual, prev, prev, prev)

        # 3) LEAR
        if "lear" in selected:
            try:
                pred = _lear_forecast(days, price_matrix, load_matrix, day)
                _record(rows, task_index, day, "lear", actual, pred, pred, pred)
            except ValueError as exc:
                print(f"  [task {task_index}] LEAR skipped: {exc}", flush=True)

        # 4) LightGBM 分位数
        if {"lgbm_quantile", "lgbm_calibrated"} & selected:
            try:
                from app.domain.forecast.price_model import LgbmQuantileForecaster

                # 两个变体共用同一份训练/校准切分，保证消融是配对的
                fit_series, fit_issue = _forecast_window(series, origin_idx, args.train_days)
                fit_origins = np.arange(0, fit_issue - HORIZON + 1, args.origin_stride)
                model = LgbmQuantileForecaster().fit(
                    fit_series,
                    horizon=HORIZON,
                    origins=fit_origins,
                    num_boost_round=args.lgbm_rounds,
                    seed=args.seed,
                    cal_days=args.cal_days,
                )
                if "lgbm_quantile" in selected:
                    raw = model.predict(fit_series, fit_issue, HORIZON, apply_calibration=False)
                    _record(
                        rows, task_index, day, "lgbm_quantile", actual,
                        raw.p10, raw.p50, raw.p90,
                    )
                if "lgbm_calibrated" in selected:
                    cal = model.predict(fit_series, fit_issue, HORIZON, apply_calibration=True)
                    _record(
                        rows, task_index, day, "lgbm_calibrated", actual,
                        cal.p10, cal.p50, cal.p90,
                    )
            except (ValueError, ImportError) as exc:
                if rolling_days is not None:
                    raise
                print(f"  [task {task_index}] lgbm skipped: {exc}", flush=True)

        # 5) MQ-CNN（截断到 origin：全序列喂入会泄漏任务日之后的数据，2026-10-22 修复）
        if "mqcnn" in selected:
            try:
                from app.domain.forecast.price_mqcnn import MqCnn

                model = MqCnn(enc_channels=3, future_features=7)
                model.fit(
                    series.ts[: origin_idx + 1],
                    series.price[: origin_idx + 1],
                    series.load[: origin_idx + 1],
                    horizon=HORIZON,
                    max_epochs=args.mqcnn_epochs,
                    seed=args.seed,
                    val_days=45,
                    origin_stride=max(1, args.mqcnn_stride),
                )
                forecast = model.predict(
                    series.ts[: origin_idx + 1],
                    series.price[: origin_idx + 1],
                    series.load[: origin_idx + 1],
                    origin_idx,
                )
                _record(
                    rows, task_index, day, "mqcnn", actual,
                    forecast.p10, forecast.p50, forecast.p90,
                )
            except (ValueError, ImportError) as exc:
                print(f"  [task {task_index}] mqcnn skipped: {exc}", flush=True)

        # 5b) MQ-RNN（July 的 arXiv:1711.11053 RNN 变体移植，同协议对决臂）
        if "mqrnn" in selected:
            try:
                from app.domain.forecast.price_mqrnn import MqRnn

                model = MqRnn(horizon=HORIZON)
                model.fit(
                    series.ts[: origin_idx + 1],
                    series.price[: origin_idx + 1],
                    None,
                    horizon=HORIZON,
                    max_epochs=args.mqrnn_epochs,
                    seed=args.seed,
                    val_days=45,
                    block_stride=max(1, args.mqrnn_stride),
                )
                forecast = model.predict(
                    series.ts[: origin_idx + 1],
                    series.price[: origin_idx + 1],
                    None,
                    origin_idx,
                )
                _record(
                    rows, task_index, day, "mqrnn", actual,
                    forecast.p10, forecast.p50, forecast.p90,
                )
            except (ValueError, ImportError, RuntimeError) as exc:
                print(f"  [task {task_index}] mqrnn skipped: {exc}", flush=True)

        # 6) 官方基准
        if "gefcom_benchmark" in selected:
            try:
                ts_bench, bench = load_task_benchmark(args.data, task_index)
                cols = list(bench.columns) if hasattr(bench, "columns") else None
                levels = np.asarray([round(0.01 * i, 2) for i in range(1, 100)])
                idx = [int(np.argmin(np.abs(levels - q))) for q in QUANTILES]
                q = bench[:, idx] * args.price_scale
                _record(rows, task_index, day, "gefcom_benchmark", actual, q[:, 0], q[:, 1], q[:, 2])
            except FileNotFoundError as exc:
                print(f"  [task {task_index}] benchmark skipped: {exc}", flush=True)

        elapsed = time.time() - t0
        log.append({"task": task_index, "day": day.date().isoformat(), "seconds": round(elapsed, 1)})
        print(f"[task {task_index:02d}] {day.date()} done in {elapsed:.1f}s", flush=True)

    predictions = pd.DataFrame(rows)
    predictions_path = out_dir / "predictions.csv"
    if args.append and predictions_path.exists():
        previous = pd.read_csv(predictions_path)
        # 同模型同名点以本次结果为准，其余保留（分批跑深度模型时不必重跑基准）
        previous = previous[~previous["model"].isin(predictions["model"].unique())]
        predictions = pd.concat([previous, predictions], ignore_index=True)
    # 源表只有壁钟时间。输出显式附加纽约时区；不支持的 DST 日直接报错。
    predictions["ts"] = [
        pd.Timestamp(ts).tz_localize("America/New_York").isoformat()
        if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).isoformat()
        for ts in predictions["ts"]
    ]
    predictions.to_csv(predictions_path, index=False, float_format="%.8f")

    metrics_rows = []
    for model, block in predictions.groupby("model"):
        baseline = predictions[predictions["model"] == "naive_daily"].sort_values("ts")["p50"].to_numpy()
        block = block.sort_values("ts")
        if len(baseline) != len(block):
            baseline = None
        q = block[["p10", "p50", "p90"]].to_numpy()
        stats = summarize(
            block["actual"].to_numpy(),
            block["p50"].to_numpy(),
            baseline=baseline,
            quantile_pred=q,
        )
        stats["model"] = model
        metrics_rows.append(stats)

    metrics = pd.DataFrame(metrics_rows).sort_values("pinball")
    metrics.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")

    per_task_rows = []
    for (model, day), block in predictions.groupby(["model", "day"]):
        block = block.sort_values("ts")
        q = block[["p10", "p50", "p90"]].to_numpy()
        stats = summarize(block["actual"].to_numpy(), block["p50"].to_numpy(), quantile_pred=q)
        per_task_rows.append({"model": model, "day": day, **stats})
    per_task = pd.DataFrame(per_task_rows)
    per_task.to_csv(out_dir / "metrics_by_task.csv", index=False, float_format="%.4f")

    meta = {
        "data": str(args.data),
        "tasks": len(task_items),
        "horizon": HORIZON,
        "points": int(len(predictions[predictions["model"] == metrics["model"].iloc[0]])),
        "models": sorted(selected),
        "seed": args.seed,
        "origin_stride": args.origin_stride,
        "rolling_start": args.rolling_start or None,
        "rolling_days": args.rolling_days,
        "train_days": args.train_days,
        "cal_days": args.cal_days,
        "lgbm_rounds": args.lgbm_rounds,
        "price_scale": args.price_scale,
        "price_unit": "CNY/kWh" if args.price_scale == 0.008 else "source_unit * price_scale",
        "timestamp_timezone": "America/New_York",
        "source_sha256": hashlib.sha256((Path(args.data) / "Task 15" / "Task15_P.csv").read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("numpy", "pandas", "lightgbm")
                     if pkg == "lightgbm" and {"lgbm_quantile", "lgbm_calibrated"} & selected or pkg != "lightgbm"},
        "peak_rss_mb": _peak_rss_mb(),
        "python_version": sys.version,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mqcnn_stride": args.mqcnn_stride,
        "mqrnn_stride": args.mqrnn_stride,
        "mqrnn_epochs": args.mqrnn_epochs,
        "elapsed_seconds": round(time.time() - started, 1),
        "task_log": log,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
