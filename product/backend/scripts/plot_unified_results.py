"""Render load/price result tables and figures separately, no hidden selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_task(root: Path, task: str) -> None:
    result = pd.read_csv(root/f"{task}_results.csv")
    available = result[result.status == "complete"].copy()
    figs = root/task/"figures"
    figs.mkdir(parents=True, exist_ok=True)
    available.to_csv(figs/"metrics_table.csv", index=False)
    for horizon, rows in available.groupby("horizon"):
        full = rows[rows.variant == "full"]
        if full.empty:
            continue
        config = json.loads((root/task/f"{horizon}h"/f"{full.iloc[0].model}__full/config.json").read_text())
        cap_label = "train-max proxy" if "proxy" in config["Cap_rule"] else "user-specified Cap"
        unit = config["unit"]
        for metric in ("MAE", "RMSE", "1-MAE/Cap", "1-RMSE/Cap"):
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(full.model, full[metric], color="#187a82")
            ax.set_title(f"{task.upper()} {horizon}h chronological test: {metric}")
            if metric.startswith("1-"):
                ax.axhline(.96, color="#ba3a2e", ls="--", label=f"96% ({cap_label})")
                ax.legend(fontsize=8)
            ax.tick_params(axis="x", rotation=55)
            ax.set_ylabel(metric)
            fig.tight_layout()
            fig.savefig(figs/f"{horizon}h_{metric.replace('/', '_')}.png", dpi=160)
            plt.close(fig)
        # Representative model is chosen by validation error, not test scores.
        best = full.sort_values("validation_RMSE").iloc[0]
        pred = pd.read_csv(root/task/f"{horizon}h"/f"{best.model}__full/predictions.csv", parse_dates=["target_ts"])
        pred["abs_error"] = (pred.y_pred-pred.y_true).abs()
        for label, is_weekend in (("weekday", False), ("weekend", True)):
            eligible = pred[pred.target_ts.dt.dayofweek.ge(5) == is_weekend]
            if eligible.empty:
                continue
            day = eligible.target_ts.dt.floor("D").iloc[0]
            # Plot one actual contiguous day, never connect disjoint weekdays.
            segment = pred[pred.target_ts.dt.floor("D") == day]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(segment.target_ts, segment.y_true, color="#222222", label="actual")
            ax.plot(segment.target_ts, segment.y_pred, color="#187a82", label="forecast")
            ax.fill_between(segment.target_ts, segment.p10, segment.p90, alpha=.15, color="#187a82", label="empirical P10/P90")
            ax.set_title(f"{task} {horizon}h {best.model}: {label}, {day.date()}")
            ax.set_ylabel(unit)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(figs/f"{horizon}h_{label}_fit.png", dpi=160)
            plt.close(fig)
        daily = pred.set_index("target_ts").resample("D").abs_error.mean()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(daily.index, daily, color="#b5651d")
        ax.set_title(f"{task} {horizon}h {best.model}: daily MAE over test time")
        ax.set_ylabel(f"MAE ({unit})")
        fig.tight_layout()
        fig.savefig(figs/f"{horizon}h_error_over_time.png", dpi=160)
        plt.close(fig)
        daily_scores = pred.assign(day=pred.target_ts.dt.floor("D")).groupby("day").apply(
            lambda g: 1-np.sqrt(np.mean((g.y_true-g.y_pred)**2))/best.Cap,
            include_groups=False)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(daily_scores.index, daily_scores, color="#187a82")
        ax.axhline(.96, color="#ba3a2e", ls="--")
        ax.set_title(f"{task} {horizon}h daily 1-RMSE/Cap: pass rate {(daily_scores>=.96).mean():.1%} ({cap_label})")
        fig.tight_layout()
        fig.savefig(figs/f"{horizon}h_daily_pass_rate.png", dpi=160)
        plt.close(fig)
        daily_scores.rename("1-RMSE/Cap").to_csv(figs/f"{horizon}h_daily_scores.csv")
        ablation = rows[rows.model == "LightGBM"]
        if len(ablation) > 1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(ablation.variant, ablation.RMSE, color=["#187a82", "#b5651d", "#ba3a2e"][:len(ablation)])
            ax.set_title(f"{task} {horizon}h LightGBM ablation (matched sample IDs)")
            ax.set_ylabel("RMSE")
            fig.tight_layout()
            fig.savefig(figs/f"{horizon}h_ablation.png", dpi=160)
            plt.close(fig)
            ablation.to_csv(figs/f"{horizon}h_ablation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    for task in ("load", "price"):
        if (args.output/f"{task}_results.csv").exists():
            plot_task(args.output, task)


if __name__ == "__main__":
    main()
