from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.protocol import ROOT


def plot_dataset(dataset="skippd"):
    runs = sorted((ROOT / "runs" / dataset).glob("*/metrics.json"))
    rows = []
    for path in runs:
        x = json.loads(path.read_text())
        model = path.parent.name.replace("_residual_quantiles", " + val residual quantiles")
        for horizon, m in x.get("by_horizon", {}).items():
            rows.append({"model": model, "horizon": int(horizon), "MAE accuracy": m["one_minus_mae_over_cap"],
                         "RMSE accuracy": m["one_minus_rmse_over_cap"]})
    if not rows:
        return
    data = pd.DataFrame(rows)
    out = ROOT / "reports" / dataset
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for metric, marker in (("MAE accuracy", "o"), ("RMSE accuracy", "s")):
        for model, group in data.groupby("model"):
            group = group.sort_values("horizon")
            plt.plot(group.horizon, group[metric] * 100, marker=marker, label=f"{model} {metric}")
    plt.axhline(96, color="black", linestyle="--", linewidth=1, label="96% threshold")
    plt.xticks([15, 60, 240], ["15 min", "1 h", "4 h"])
    plt.ylabel("1 - error / Cap (%)")
    plt.xlabel("Forecast horizon")
    plt.ylim(0, 100)
    plt.grid(alpha=.2)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "baseline_accuracy_by_horizon.png", dpi=180)
    plt.close()

    weather = []
    for path in runs:
        x = json.loads(path.read_text())
        for horizon, m in x.get("by_horizon", {}).items():
            for label, values in m.get("weather", {}).items():
                if values.get("n", 0):
                    weather.append({"model": path.parent.name, "horizon": int(horizon), "weather": label,
                                    "RMSE accuracy": values["one_minus_rmse_over_cap"]})
    if weather:
        w = pd.DataFrame(weather)
        pivot = w.pivot_table(index=["weather", "horizon"], columns="model", values="RMSE accuracy")
        pivot.plot(kind="bar", figsize=(12, 5), ylim=(0, 1), grid=False)
        plt.axhline(.96, color="black", linestyle="--", linewidth=1)
        plt.ylabel("1 - RMSE / Cap")
        plt.xlabel("Weather label and horizon")
        plt.tight_layout()
        plt.savefig(out / "baseline_weather_rmse.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    plot_dataset()
