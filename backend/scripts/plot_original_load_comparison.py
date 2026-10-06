"""Plot a fixed, timestamp-aligned original-model comparison without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    results = pd.read_csv(args.output / "load_comparison.csv")
    plots = args.output / "plots"
    plots.mkdir(exist_ok=True)
    aliases = {"Original_load_GEFCom_adapter": "Original adapter", "LSTM_baseline": "LSTM quantile",
               "TCN_baseline": "TCN quantile", "TCN_optimized": "TCN optimized"}
    for h in (1, 4):
        rows = results[results.horizon == h]
        labels = [aliases.get(v, v).replace(" ", "\n", 1) for v in rows.variant]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, metric in zip(axes.flat, ("MAE", "RMSE", "Pinball", "interval_coverage_80")):
            ax.bar(np.arange(len(rows)), rows[metric], color=["#ab3546", "#747d83", "#747d83", "#147d82", "#4c966d", "#365a8c"])
            ax.set_xticks(np.arange(len(rows)), labels, fontsize=9)
            ax.set_title(metric); ax.grid(axis="y", alpha=.2)
            if metric == "interval_coverage_80":
                ax.axhline(.8, color="#444444", linestyle="--", linewidth=1)
                ax.set_ylim(0, 1)
        fig.suptitle(f"GEFCom LOAD {h}h: exploratory comparison, not main experiment", fontsize=14)
        fig.savefig(plots / f"load_{h}h_metrics.png", dpi=150); plt.close(fig)
        original = pd.read_csv(args.output / f"load/{h}h/predictions.csv", parse_dates=["target_ts"])
        # First test week is chosen by timestamp, never by model performance.
        week = original.target_ts < original.target_ts.iloc[0] + pd.Timedelta(days=7)
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        ax.plot(original.loc[week, "target_ts"], original.loc[week, "y_true"], label="Observed", color="#333333", linewidth=1.5)
        ax.plot(original.loc[week, "target_ts"], original.loc[week, "point"], label="Original adapter", color="#ab3546", linewidth=1)
        for label, color in (("LSTM_baseline", "#147d82"), ("TCN_optimized", "#365a8c")):
            p = pd.read_csv(args.previous / f"load/{h}h/evaluation/{label}/predictions.csv")
            ax.plot(original.loc[week, "target_ts"], p.loc[week, "p50"], label=aliases[label], color=color, linewidth=1)
        ax.set_title(f"{h}h: first chronological test week (fixed display window)")
        ax.set_ylabel("Load (source units)"); ax.legend(ncol=4); ax.grid(alpha=.2)
        fig.savefig(plots / f"load_{h}h_first_test_week.png", dpi=150); plt.close(fig)
    audit = []
    for h in (1, 4):
        p = pd.read_csv(args.output / f"load/{h}h/predictions.csv")
        m = json.loads((args.output / f"load/{h}h/metrics.json").read_text())
        error = p.point.to_numpy() - p.y_true.to_numpy()
        mae, rmse = float(np.abs(error).mean()), float(np.sqrt(np.mean(error**2)))
        if abs(mae-m["MAE"]) > 1e-5 or abs(rmse-m["RMSE"]) > 1e-5:
            raise ValueError("CSV and JSON metrics disagree")
        audit.append({"horizon": h, "n": len(p), "csv_json_metric_check": "passed"})
    (args.output / "metric_recomputation_audit.json").write_text(json.dumps(audit, indent=2))
    lines = ["## 实际效果", ""]
    for h in (1, 4):
        rows = results[results.horizon == h].set_index("variant")
        current = rows.loc["Original_load_GEFCom_adapter"]
        for name in ("Persistence", "LSTM_baseline", "TCN_optimized"):
            previous = rows.loc[name]
            mae_change = (current.MAE / previous.MAE - 1) * 100
            rmse_change = (current.RMSE / previous.RMSE - 1) * 100
            lines.append(f"- {h}h 原适配模型相对 {name}：MAE变化 {mae_change:+.2f}%，RMSE变化 {rmse_change:+.2f}%；负值代表误差降低。")
    lines += ["", "这些差异不能单独归因于注意力：网络宽度、损失、解码方式也不同，且没有等预算超参搜索。仿真域旧指标与真实域新指标也不能直接相减宣称改善。若结果较差，意味着该固定适配配置在此数据上较差，不证明所有注意力模型无效。", ""]
    report = args.output / "REPORT.md"
    text = report.read_text().split("## 实际效果", 1)[0]
    report.write_text(text.rstrip() + "\n\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
