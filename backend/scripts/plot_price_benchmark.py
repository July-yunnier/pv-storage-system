"""把 `run_price_benchmark.py` 的结果画成报告图。

产出（默认写到同一输出目录）：
- `fig_metrics.png`：各模型的 MAE / pinball / 峰值命中对比
- `fig_curve_task15.png`：尖峰日（2013-12-17）的分位数曲线 vs 实测
- `fig_by_task.png`：逐任务日 MAE，看模型在哪些日子失效
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "lgbm_quantile": "LightGBM 分位数",
    "mqcnn": "MQ-CNN",
    "lear": "LEAR (Lasso AR)",
    "naive_daily": "昨日同时刻",
    "naive_weekly": "上周同时刻 / 官方基准",
    "gefcom_benchmark": "GEFCom 官方基准",
}

COLORS = {
    "lgbm_quantile": "#cf1322",
    "mqcnn": "#e8722a",
    "lear": "#2f54eb",
    "naive_daily": "#8c8c8c",
    "naive_weekly": "#595959",
    "gefcom_benchmark": "#bfbfbf",
}


def _setup_font() -> None:
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
        except Exception:  # noqa: BLE001
            continue


def plot_metrics(metrics: pd.DataFrame, out: Path) -> None:
    frame = metrics.sort_values("pinball", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("mae", "MAE（元等价单位，越低越好）"),
        ("pinball", "Pinball 损失（0.1/0.5/0.9）"),
        ("peak_hit", "峰值时段命中率"),
    ]
    for ax, (key, title) in zip(axes, panels):
        values = frame[key].to_numpy()
        colors = [COLORS.get(m, "#8c8c8c") for m in frame["model"]]
        bars = ax.barh([LABELS.get(m, m) for m in frame["model"]], values, color=colors)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.2f}",
                    va="center", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("GEFCom2014-P 电价预测基准（15 任务日 × 24 小时 = 360 点）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_curve(predictions: pd.DataFrame, out: Path, day: str, models: tuple[str, ...]) -> None:
    block = predictions[predictions["day"] == day]
    if block.empty:
        return
    actual = block[block["model"] == block["model"].iloc[0]].sort_values("ts")["actual"].to_numpy()
    hours = np.arange(len(actual))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(hours, actual, "k-", linewidth=2.4, label="实测")
    for model in models:
        sub = block[block["model"] == model].sort_values("ts")
        if sub.empty:
            continue
        ax.plot(hours, sub["p50"].to_numpy(), color=COLORS.get(model, "#888"),
                linewidth=1.6, label=f"{LABELS.get(model, model)} P50")
    best = block[block["model"] == models[0]].sort_values("ts")
    if not best.empty:
        ax.fill_between(hours, best["p10"].to_numpy(), best["p90"].to_numpy(),
                        color=COLORS.get(models[0], "#cf1322"), alpha=0.15, label="P10–P90")
    ax.set_xlabel("小时")
    ax.set_ylabel("电价（$/MWh）")
    ax.set_title(f"任务日 {day} 的分位数预测 vs 实测")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_by_task(per_task: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for model, block in per_task.groupby("model"):
        block = block.sort_values("day")
        ax.plot(block["day"], block["mae"], marker="o", markersize=4,
                color=COLORS.get(model, "#888"), label=LABELS.get(model, model))
    ax.set_ylabel("MAE")
    ax.set_title("逐任务日 MAE：模型失效发生在哪些日子")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/forecast_experiments/price_benchmark")
    parser.add_argument("--day", default="2013-12-17")
    args = parser.parse_args()

    _setup_font()
    root = Path(args.dir)
    metrics = pd.read_csv(root / "metrics.csv")
    predictions = pd.read_csv(root / "predictions.csv")
    per_task = pd.read_csv(root / "metrics_by_task.csv")

    plot_metrics(metrics, root / "fig_metrics.png")
    plot_curve(predictions, root / "fig_curve_task15.png", args.day,
               ("lgbm_quantile", "mqcnn", "lear", "naive_daily"))
    plot_by_task(per_task, root / "fig_by_task.png")
    print(f"figures written to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
