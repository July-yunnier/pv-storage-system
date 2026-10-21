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
6. `gefcom_benchmark` 赛事官方基准给出的 99 分位数预测

用法::

    python scripts/run_price_benchmark.py --data ../../data/power/price/Price \
        --output data/forecast_experiments/price_benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
        default="naive_daily,naive_weekly,lear,lgbm_quantile,mqcnn,gefcom_benchmark",
    )
    parser.add_argument("--origin-stride", type=int, default=3, help="LGBM 训练 origin 抽样步长")
    parser.add_argument("--mqcnn-stride", type=int, default=6, help="MQ-CNN 训练 origin 抽样步长")
    parser.add_argument("--mqcnn-epochs", type=int, default=25)
    parser.add_argument("--lgbm-rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", default="", help="只跑指定任务，如 1,15（调试用）")
    parser.add_argument("--append", action="store_true", help="保留已有 predictions.csv 的其他模型")
    args = parser.parse_args()

    selected = {m.strip() for m in args.models.split(",") if m.strip()}
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    gefcom = load_gefcom_price(args.data)
    series = gefcom.series
    # 保留官方任务编号：--tasks 只是筛选，不能重排，否则基准文件会错位
    task_items = list(enumerate(gefcom.task_days, start=1))
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
        if "lgbm_quantile" in selected:
            try:
                from app.domain.forecast.price_model import LgbmQuantileForecaster

                model = LgbmQuantileForecaster().fit(
                    series,
                    horizon=HORIZON,
                    origins=train_origins,
                    num_boost_round=args.lgbm_rounds,
                    seed=args.seed,
                )
                forecast = model.predict(series, origin_idx, HORIZON)
                _record(
                    rows, task_index, day, "lgbm_quantile", actual,
                    forecast.p10, forecast.p50, forecast.p90,
                )
            except (ValueError, ImportError) as exc:
                print(f"  [task {task_index}] lgbm skipped: {exc}", flush=True)

        # 5) MQ-CNN
        if "mqcnn" in selected:
            try:
                from app.domain.forecast.price_mqcnn import MqCnn

                model = MqCnn(enc_channels=3, future_features=7)
                model.fit(
                    series.ts,
                    series.price,
                    series.load,
                    horizon=HORIZON,
                    max_epochs=args.mqcnn_epochs,
                    seed=args.seed,
                    val_days=45,
                    origin_stride=max(1, args.mqcnn_stride),
                )
                forecast = model.predict(series.ts, series.price, series.load, origin_idx)
                _record(
                    rows, task_index, day, "mqcnn", actual,
                    forecast.p10, forecast.p50, forecast.p90,
                )
            except (ValueError, ImportError) as exc:
                print(f"  [task {task_index}] mqcnn skipped: {exc}", flush=True)

        # 6) 官方基准
        if "gefcom_benchmark" in selected:
            try:
                ts_bench, bench = load_task_benchmark(args.data, task_index)
                cols = list(bench.columns) if hasattr(bench, "columns") else None
                levels = np.asarray([round(0.01 * i, 2) for i in range(1, 100)])
                idx = [int(np.argmin(np.abs(levels - q))) for q in QUANTILES]
                q = bench[:, idx]
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
    predictions.to_csv(predictions_path, index=False, float_format="%.4f")

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
        "mqcnn_stride": args.mqcnn_stride,
        "elapsed_seconds": round(time.time() - started, 1),
        "task_log": log,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
